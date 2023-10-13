import os
import shutil
import json
import random
import redis

from django.db.models import F
from django.db import transaction
from django.conf import settings

from cvat.apps.engine.models import Project, Task, LabeledShape, Label, AttributeSpec, Job, \
    S3File, Data, StorageMethodChoice, SortingMethod, RemoteFile, ModeChoice, LabeledShapeAttributeVal
from cvat.apps.engine.task import _move_data_to_s3, _validate_data, _validate_url, retry, \
    _get_manifest_frame_indexer, _get_task_segment_data, _count_files, _check_filename_collisions, \
    _validate_manifest
from cvat.apps.engine.media_extractors import sort
from cvat.apps.engine.log import slogger

logger = slogger.glob

# Engine task start #

# So the problem here is, that engine task processor
# assumes that it runs in a python-rq job
# and updates its status from time to time.
# We can't run rq job from shell, because worker
# process cannot import functions from it.
# The task is to write Bastion script to avoid deploying updates.
# So I had to copy task processing here and remove some
# unnecessary parts from it, including job status updating.

import itertools
import imghdr
import requests
from urllib import parse as urlparse, request as urlrequest

from cvat.apps.engine import models
from cvat.apps.engine.media_extractors import MEDIA_TYPES
from cvat.apps.engine.utils import av_scan_paths

from utils.dataset_manifest import CachedIndexManifestManager


# removed job status update
def _save_task_to_db(db_task, extractor):
    logger.info('Saving task to database')

    segment_step, segment_size, overlap = _get_task_segment_data(db_task, db_task.data.size)
    db_task.segment_size = segment_size
    db_task.overlap = overlap

    for start_frame in range(0, db_task.data.size, segment_step):
        stop_frame = min(start_frame + segment_size - 1, db_task.data.size - 1)

        logger.info("New segment for task #{}: start_frame = {}, \
            stop_frame = {}".format(db_task.id, start_frame, stop_frame))

        db_segment = models.Segment()
        db_segment.task = db_task
        db_segment.start_frame = start_frame
        db_segment.stop_frame = stop_frame
        db_segment.save()

        db_job = models.Job(segment=db_segment)
        db_job.save()

        job_path = db_job.get_dirname()
        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.makedirs(job_path)

        preview = extractor.get_preview(frame=start_frame)
        preview.save(db_job.get_preview_path())

    db_task.data.save()
    db_task.save()


# removed adding scan id
def fix_filename(filename, file_path):
    name, ext = os.path.splitext(filename)

    guess_ext = imghdr.what(file_path)
    if guess_ext is not None:
        guess_ext = '.' + guess_ext
        if guess_ext != ext:
            ext = guess_ext

    return f'{name}{ext}'


# removed job status update.
def _download_data(db_data: models.Data, upload_dir, rename_files=False):
    local_files = {}
    remote_files = db_data.remote_files.all()
    for file in remote_files:
        url = file.file
        _validate_url(url)
        logger.info("Downloading: {}".format(url))

        response = retry(requests.get, args=(url,), kwargs={'stream': True, 'timeout': 30}, times=5, delay=1, factor=2)
        if response.status_code == 200:
            response.raw.decode_content = True

            name = os.path.basename(urlrequest.url2pathname(urlparse.urlparse(url).path))
            output_path = os.path.join(upload_dir, name)
            with open(output_path, 'wb') as output_file:
                shutil.copyfileobj(response.raw, output_file)

            new_name = fix_filename(name, output_path)
            new_name = _check_filename_collisions(new_name, local_files, rename_files)
            if new_name != name:
                new_path = os.path.join(upload_dir, new_name)
                os.rename(output_path, new_path)
        else:
            logger.error(f'Failed to download {url}')
            logger.error(f'Status: {response.status_code}')
            logger.error(response.text)
            raise Exception("Failed to download " + url)

        local_files[new_name] = new_name
        file.meta['name'] = new_name
        file.save()
    return list(local_files.values())


# removed job status update and unnecessary parts (like video and 3d processing).
def _create_noatomic(db_task, data, rename_files=True):
    if isinstance(db_task, int):
        db_task = models.Task.objects.select_for_update().get(pk=db_task)

    logger.info("create task #{}".format(db_task.id))

    db_data = db_task.data
    upload_dir = db_data.get_upload_dirname()
    os.makedirs(upload_dir, exist_ok=True)

    if data['remote_files']:
        data['remote_files'] = _download_data(db_data, upload_dir, rename_files=rename_files)

    manifest_files = []
    media = _count_files(data, manifest_files)
    media, task_mode = _validate_data(media, manifest_files)

    manifest_file = _validate_manifest(manifest_files, upload_dir, False, None)
    if manifest_file and (not settings.USE_CACHE or db_data.storage_method != models.StorageMethodChoice.CACHE):
        raise Exception("File with meta information can be uploaded if 'Use cache' option is also selected")

    av_scan_paths(upload_dir)

    logger.info('Media files are being extracted...')

    db_images = []
    extractor = None
    manifest_index = _get_manifest_frame_indexer()

    for media_type, media_files in media.items():
        if media_files:
            if extractor is not None:
                raise Exception('Combined data types are not supported')
            source_paths = [os.path.join(upload_dir, f) for f in media_files]
            if manifest_file and data['sorting_method'] in {
                models.SortingMethod.RANDOM, models.SortingMethod.PREDEFINED
            }:
                raise Exception("It isn't supported to upload manifest file and use random sorting")

            details = {
                'source_path': source_paths,
                'step': db_data.get_frame_step(),
                'start': db_data.start_frame,
                'stop': data['stop_frame'],
            }
            if media_type in {'archive', 'zip', 'pdf'} and db_data.storage == models.StorageChoice.SHARE:
                details['extract_dir'] = db_data.get_upload_dirname()
                upload_dir = db_data.get_upload_dirname()
                db_data.storage = models.StorageChoice.LOCAL
            if media_type != 'video':
                details['sorting_method'] = data['sorting_method']
            extractor = MEDIA_TYPES[media_type]['extractor'](**details)

    db_task.mode = task_mode
    db_data.compressed_chunk_type = models.DataChoice.IMAGESET
    db_data.original_chunk_type = models.DataChoice.IMAGESET

    if db_data.chunk_size is None:
        w, h = extractor.get_image_size(0)
        area = h * w
        db_data.chunk_size = max(2, min(72, 36 * 1920 * 1080 // area))

    for media_type, media_files in media.items():
        if not media_files:
            continue

        # replace manifest file (e.g was uploaded 'subdir/manifest.jsonl' or 'some_manifest.jsonl')
        if manifest_file and not os.path.exists(db_data.get_manifest_path()):
            shutil.copyfile(os.path.join(upload_dir, manifest_file), db_data.get_manifest_path())
            if upload_dir != settings.SHARE_ROOT:
                os.remove(os.path.join(upload_dir, manifest_file))

        # images, archive, pdf
        db_data.size = len(extractor)
        manifest = CachedIndexManifestManager(db_data.get_manifest_path())
        if not manifest_file:
            manifest.link(
                sources=extractor.absolute_source_paths,
                meta={},  # no related images here
                data_dir=upload_dir,
                DIM_3D=False,
            )
            manifest.create()
        else:
            manifest.init_index()
        counter = itertools.count()
        for _, chunk_frames in itertools.groupby(extractor.frame_range, lambda x: next(counter) // db_data.chunk_size):
            chunk_paths = [(extractor.get_path(i), i) for i in chunk_frames]
            img_sizes = []

            for chunk_path, frame_id in chunk_paths:
                properties = manifest[manifest_index(frame_id)]

                # check mapping
                if not chunk_path.endswith(f"{properties['name']}{properties['extension']}"):
                    raise Exception('Incorrect file mapping to manifest content')
                resolution = (properties['width'], properties['height'])
                img_sizes.append(resolution)

            db_images.extend([
                models.Image(
                    data=db_data, path=os.path.relpath(path, upload_dir),
                    frame=frame, width=w, height=h
                ) for (path, frame), (w, h)
                in zip(chunk_paths, img_sizes)
            ])

    models.Image.objects.bulk_create(db_images)
    stop_frame = db_data.start_frame + (db_data.size - 1) * db_data.get_frame_step()
    if db_data.stop_frame > 0:
        stop_frame = min(db_data.stop_frame, stop_frame)
    db_data.stop_frame = stop_frame

    task_preview = extractor.get_preview(frame=0)
    task_preview.save(db_data.get_preview_path())

    logger.info("Found frames {} for Data #{}".format(db_data.size, db_data.id))

    _save_task_to_db(db_task, extractor)
    _move_data_to_s3(db_task, db_data)

#  Engine task end  #


def _rand_color():
    choices = '0123456789ABCDEF'
    color = '#'
    for _ in range(6):
        color += random.choice(choices)
    return color


def _prepare_labels(source_project: Project, target_project: Project, labels):
    source_labels = {
        label.name: label for label in
        source_project.label_set.filter(name__in=labels).prefetch_related('attributespec_set')
    }

    do_not_exist = []
    for key in labels:
        if key not in source_labels:
            do_not_exist.append(key)

    if len(do_not_exist) > 0:
        raise ValueError(f'Source project does not have the labels: {do_not_exist},'
                         f' while source project has: {list(source_labels.keys())}')

    target_labels = {
        label.name: label for label in
        target_project.label_set.prefetch_related('attributespec_set')
    }
    labels_map = {}
    for source, target in labels.items():
        source_label: Label = source_labels[source]

        if target in target_labels:
            target_label = target_labels[target]
        else:
            target_label = Label.objects.create(
                name=target,
                project=target_project,
                color=_rand_color(),
                type=source_label.type,
            )
            target_labels[target] = target_label

        source_specs = {spec.name: spec for spec in source_label.attributespec_set.all()}
        target_specs = {spec.name: spec for spec in target_label.attributespec_set.all()}

        specs_map = {}
        for name in source_specs:
            source_spec = source_specs[name]
            if name in target_specs:
                target_spec = target_specs[name]
            else:
                target_spec = AttributeSpec.objects.create(
                    label=target_label,
                    name=name,
                    mutable=source_spec.mutable,
                    input_type=source_spec.input_type,
                    default_value=source_spec.default_value,
                    values=source_spec.values,
                )
                target_specs[name] = target_spec
            specs_map[source_spec.pk] = target_spec.pk

        labels_map[source_label.pk] = {
            'pk': target_label.pk,
            'specs': specs_map,
        }

    return labels_map


def _get_files(task, labels, frame=0):
    files = sort(
        S3File.objects.filter(data_id=task['data_id']),
        sorting_method=task['sorting'],
        func=lambda f: f.file.name,
    )

    shapes = LabeledShape.objects \
        .annotate(task_id=F('job__segment__task_id')) \
        .filter(label_id__in=labels, task_id=task['pk'], frame__gte=frame) \
        .order_by('frame') \
        .prefetch_related('labeledshapeattributeval_set')

    result = []
    last = 0
    for i in range(1, len(shapes)):
        if shapes[i].frame != shapes[last].frame:
            result.append((
                files[shapes[last].frame],
                shapes[last:i],
            ))
            last = i
    result.append(shapes[last:])

    return result


@transaction.atomic
def _create_task(project: Project, data, labels, job_size, task_size, i):
    size = min(len(data), task_size)

    logger.info('Creating task')
    db_data = Data.objects.create(
        image_quality=70,
        storage_method=StorageMethodChoice.CACHE,
        size=size,
        stop_frame=size - 1,
        sorting_method=SortingMethod.LEXICOGRAPHICAL,
    )
    os.makedirs(db_data.get_upload_dirname(), exist_ok=True)

    task_name = f'Task {i}'

    task = Task.objects.create(
        project=project,
        data=db_data,
        name=task_name,
        owner=project.owner,
        organization=project.organization,
        mode=ModeChoice.ANNOTATION,
        segment_size=job_size,
    )

    shapes_map = {}
    files = []
    for file, shapes in data:
        files.append(RemoteFile(
            data=db_data,
            file=file.file.url,
            meta={'from_pk': file.pk}
        ))
        shapes_map[file.pk] = shapes

    files = RemoteFile.objects.bulk_create(files, batch_size=100)

    logger.info('Processing task')
    try:
        _create_noatomic(task.id, {
            'chunk_size': db_data.chunk_size,
            'size': db_data.size,
            'image_quality': db_data.image_quality,
            'start_frame': db_data.start_frame,
            'stop_frame': db_data.stop_frame,
            'frame_filter': db_data.frame_filter,
            'compressed_chunk_type': db_data.compressed_chunk_type,
            'original_chunk_type': db_data.original_chunk_type,
            'client_files': [],
            'server_files': [],
            'remote_files': [file.file for file in files],
            'use_zip_chunks': True,
            'use_cache': True,
            'copy_data': False,
            'storage_method': db_data.storage_method,
            'storage': db_data.storage,
            'sorting_method': db_data.sorting_method,
        }, rename_files=True)
    except Exception as e:
        shutil.rmtree(db_data.get_data_dirname())
        raise e

    logger.info('Creating annotations')
    files = sort(
        S3File.objects.filter(data=db_data),
        sorting_method=db_data.sorting_method,
        func=lambda f: f.meta['name'],
    )

    jobs = Job.objects.filter(segment__task_id=task.pk).order_by('pk')
    for frame, file in enumerate(files):
        shapes = {}
        attributes = []
        for shape in shapes_map[file.meta['from_pk']]:
            shapes[shape.pk] = LabeledShape(
                label_id=labels[shape.label_id]['pk'],
                job_id=jobs[frame // job_size].pk,
                frame=frame,
                group=shape.group,
                source=shape.source,
                type=shape.type,
                occluded=shape.occluded,
                outside=shape.outside,
                z_order=shape.z_order,
                points=shape.points,
                rotation=shape.rotation,
            )

            attributes += [LabeledShapeAttributeVal(
                spec_id=labels[shape.label_id]['specs'][attribute.spec_id],
                shape_id=attribute.shape_id,
                value=attribute.value,
            ) for attribute in shape.labeledshapeattributeval_set.all()]

        LabeledShape.objects.bulk_create(shapes, batch_size=100)
        for attribute in attributes:
            attribute.shape_id = shapes[attribute.shape_id].pk

        LabeledShapeAttributeVal.objects.bulk_create(attributes, batch_size=100)


def start(source_id, target_id, labels, job_size=50, n_jobs=10):
    """
    source_id - source project id
    target_id - target project id, must already exist
    labels - labels' name mapping
    job_size - number of images per target job
    n_jobs - number of jobs per target task
    batch_size - how much images to process at once.
    """

    if n_jobs < 1:
        raise ValueError('Task should have at least 1 job')
    if job_size < 1:
        raise ValueError('Job size should be positive integer')

    try:
        source_project = Project.objects.get(pk=source_id)
    except Project.DoesNotExist:
        raise ValueError('Source project does not exist')

    try:
        target_project = Project.objects.get(pk=target_id)
    except Project.DoesNotExist:
        raise ValueError('Target project does not exist')

    cache = redis.Redis.from_url(settings.REDIS_URL)
    cache_key = f'copy_prj_{source_id}_{target_id}'
    progress = cache.get(cache_key)
    if progress is None:
        progress = {
            'task_id': 0,
            'frame': 0,
            'task_n': 0,
        }
    else:
        progress = json.loads(progress)

    labels = _prepare_labels(source_project, target_project, labels)

    target_tasks = target_project.tasks\
        .filter(data__isnull=False, pk__gte=progress['task_id'])\
        .annotate(size='data__size', sorting='data__sorting_method')\
        .values('pk', 'data_id', 'size', 'sorting')

    size = 0
    files = []
    task_size = job_size * n_jobs
    task_n = progress['task_n']
    frame = progress['frame']
    for task in target_tasks:
        size += task['size']
        files += _get_files(task, labels, frame)
        frame = 0
        if size >= task_size:
            _create_task(target_project, files[:task_size], labels, job_size, task_size, task_n)
            size -= task_size
            files = files[task_size:]
            task_n += 1

            cache.set(cache_key, {
                'task_id': files[0][0].task_id,
                'frame': files[0][0].frame,
                'task_n': task_n
            })

    if size > 0:
        _create_task(files)

    cache.delete(cache_key)


def test_local():
    start(12, 1, {
        'Price tag': 'Price tag',
        'Box container': 'Product',
        'Product item': 'Product',
        'text_area': 'Text OCR',
        'Shelf': 'Shelf line',
        'Shelf edge': 'Shelf line'
    }, job_size=3, n_jobs=2)


# destination keys
# product, price_tag pegboard shelf_line edge_line virtual_line

def r3dev_to_flat_project():
    start(67, 1005, {
        'price_tag': 'price_tag',
        # 'Price tag': 'price_tag',
        # 'Box container': 'product',
        'Product item': 'product',
        'All UPC': 'product',
        'ShelfLine': 'shelf_line',
        'Pegboard': 'pegboard',
        'Shelf edge': 'edge_line',
    }, job_size=50, n_jobs=10)


def r3us_to_flat_project():
    start(26, 27, {
        'All PRICE TAGS': 'price_tag',
        'All UPC': 'product',
        'Price tag': 'price_tag',
        #'Location tag'
        'Shelf line': 'shelf_line',
        'Shelf edge': 'edge_line',

        'Product item': 'product',
        'Virtual line': 'virtual_line',
        # 'Box container': 'product',
        'Box container': 'box_container',
        'Pegboard': 'pegboard',
    }, job_size=50, n_jobs=10)


test_local()
