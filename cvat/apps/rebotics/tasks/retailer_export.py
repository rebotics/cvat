import hashlib
from typing import List
from io import BytesIO

import rq
import requests
import django_rq
from rq.job import Job, JobStatus
from django.db import transaction
from django.utils import timezone
from rebotics_sdk.providers import RetailerProvider

from cvat.apps.engine.models import Task, S3File
from cvat.rebotics.utils import save_ranges, load_ranges, add_to_ranges
from cvat.apps.engine.media_extractors import sort
from cvat.apps.engine.task import retry
from cvat.apps.engine.log import slogger
from cvat.rebotics.cache import default_cache

META_LAST_TASK_ID = 'last_task_id'
META_LAST_FRAME = 'last_frame'
META_SCAN_IDS = 'scan_ids'


class RetailerExportError(Exception):
    pass


def _sorted_files(task: Task, start_frame: int = 0) -> List[S3File]:
    return sort(
        task.data.s3_files.all(),
        sorting_method=task.data.sorting_method,
        func=lambda f: f.name
    )[start_frame:]


def _retailer_name(retailer_host: str):
    return retailer_host.split('//', 1)[1].split('.', 1)[0][:50]


def _task_hash(task_ids):
    return hashlib.md5('_'.join(str(i) for i in sorted(task_ids)).encode()).hexdigest()


def _timestamp():
    return int(timezone.now().timestamp() * 1000000)


@transaction.atomic
def _start_rq(task_ids, start_frame, retailer_name, username, store_id=None):
    rq_job: Job = rq.get_current_job()
    slogger.glob.info(f'Starting export {task_ids} to {retailer_name}, rq job: {rq_job.id}')

    retailer_auth = default_cache.get(f'{username}_{retailer_name}')
    if retailer_auth is None:
        raise RetailerExportError('Auth credentials are not found. '
                                  'Please, authenticate before exporting.')

    try:
        if store_id is None:
            store_id = retailer_auth['store_id']
    except KeyError:
        raise RetailerExportError('Store id is not provided and default store id is not found.'
                                  'Please, authenticate to set it.')

    retailer = RetailerProvider(retailer_auth['host'], token=retailer_auth['token'])
    tasks = Task.objects.filter(pk__in=task_ids).order_by('pk') \
        .select_related('data').prefetch_related('data__s3_files')

    scan_ids = []
    rq_job.meta[META_SCAN_IDS] = save_ranges(scan_ids)

    for task in tasks:
        frame = start_frame
        rq_job.meta[META_LAST_TASK_ID] = task.pk

        for file in _sorted_files(task, start_frame):
            slogger.glob.info(f'{task.pk} {frame} {file.name}')

            result = retailer.processing_upload_request(file.name)
            file_id = result['id']
            s3_dest = result['destination']

            with BytesIO(file.file.read()) as f:
                retry(requests.post, kwargs={
                    'url': s3_dest['url'],
                    'data': s3_dest['fields'],
                    'files': {'file': f},
                }, times=3, delay=10, factor=2)

            retailer.notify_processing_upload_finished(file_id)
            result = retailer.create_processing_action(store_id, files=[file_id])

            scan_ids.append(result['id'])

            rq_job.meta[META_LAST_FRAME] = frame
            rq_job.meta[META_SCAN_IDS] = add_to_ranges(rq_job.meta[META_SCAN_IDS], result['id'])
            rq_job.save_meta()

            frame += 1

        start_frame = 0

    return scan_ids


def start(task_ids, start_frame, retailer_name, user, store_id=None):
    queue = django_rq.get_queue('default')
    id_base = f'/api/retailer_export/{retailer_name}_{_task_hash(task_ids)}'

    for i in queue.get_job_ids() + queue.started_job_registry.get_job_ids():
        if i.startswith(id_base):
            return i

    rq_id = f'{id_base}_{_timestamp()}'
    queue.enqueue_call(_start_rq, args=(
        task_ids, start_frame, retailer_name, user.username, store_id,
    ), job_id=rq_id)
    return rq_id


def auth(retailer_host, username, password, user, verification_code=None, default_store_id=None):
    retailer = RetailerProvider(host=retailer_host)
    kwargs = {}
    if verification_code is not None:
        kwargs['verification_code'] = verification_code
    response = retailer.token_auth(username, password, **kwargs)
    token = response.json()['token']
    retailer_name = _retailer_name(retailer_host)
    default_cache.set(f'{user.username}_{retailer_name}',
                      {'token': token, 'host': retailer_host, 'store_id': default_store_id},
                      expire=60*60*24*90)  # 90 days in seconds
    return retailer_name


def check(rq_id):
    queue = django_rq.get_queue('default')
    job: Job = queue.fetch_job(rq_id)
    if job is None:
        return None

    status = job.get_status()
    return {
        'status': status,
        'scan_ids': job.return_value if status == JobStatus.FINISHED else load_ranges(job.meta.get(META_SCAN_IDS, '')),
        'last_task_id': job.meta.get(META_LAST_TASK_ID, None),
        'last_frame': job.meta.get(META_LAST_FRAME, None),
        'message': job.exc_info if status == JobStatus.FAILED else '',
    }
