import hashlib
from typing import List

import rq
import requests
import django_rq
from rq.job import Job, JobStatus
from django.db import transaction
from rebotics_sdk.providers import RetailerProvider

from cvat.apps.engine.models import Task, S3File
from cvat.rebotics.utils import save_ranges, load_ranges, add_to_ranges
from cvat.apps.engine.media_extractors import sort
from cvat.apps.engine.task import retry
from cvat.apps.engine.log import slogger
from io import BytesIO

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
    return retailer_host.split('//', 1)[1].split('.', 1)[0]


def _task_hash(task_ids):
    return hashlib.md5('_'.join(str(i) for i in task_ids).encode()).hexdigest()


@transaction.atomic
def _start_rq(task_ids, start_frame, retailer_host, retailer_token, store_id):
    tasks = Task.objects.filter(pk__in=task_ids).order_by('pk')\
        .select_related('data').prefetch_related('data__s3_files')
    rq_job: Job = rq.get_current_job()
    slogger.glob.info(f'Starting export {task_ids} to {retailer_host}, rq job: {rq_job.id}')

    retailer = RetailerProvider(retailer_host, token=retailer_token)
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


def start(task_ids, start_frame, retailer_host, retailer_token, store_id):
    queue = django_rq.get_queue('default')
    rq_id = f'/api/retailer_export/{_retailer_name(retailer_host)}/{_task_hash(task_ids)}'
    job: Job = queue.fetch_job(rq_id)

    if job is None or job.is_finished or job.is_failed:
        queue.enqueue_call(_start_rq, args=(
            task_ids, start_frame, retailer_host, retailer_token, store_id
        ), job_id=rq_id)
        return rq_id

    raise RetailerExportError(f'Export job to {retailer_host} for {task_ids} already exists.')


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
