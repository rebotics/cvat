import random
from cvat.apps.engine.models import Project, Task, LabeledShape, Label, AttributeSpec,\
    S3File, Data, StorageMethodChoice, SortingMethod, RemoteFile, ModeChoice, LabeledShapeAttributeVal
from cvat.apps.engine.task import _create_noatomic
from django.db.models import F
from django.db import transaction
from cvat.apps.engine.media_extractors import sort
from cvat.apps.engine.log import slogger
import json
import redis
from django.conf import settings
import os
import shutil
import django_rq
from rq.job import JobStatus, Job as RqJob

logger = slogger.glob


def _rand_color():
    choices = '0123456789ABCDEF'
    color = '#'
    for _ in range(6):
        color += random.choice(choices)
    return color


def _prepare(source_id, target_id, labels, job_size=50, n_jobs=10):
    # validate arguments
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

    # create labels if needed
    source_labels = {
        label.name: label for label in
        source_project.label_set.filter(name__in=labels).prefetch_related('attributespec_set')
    }
    do_not_exist = []
    for key in labels:
        if key not in source_labels:
            do_not_exist.append(key)

    if len(do_not_exist) > 0:
        raise ValueError(f'Source project does not have the labels: {do_not_exist}')

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

    # this has label, frame and relation to task through job and segment
    shapes = LabeledShape.objects\
        .filter(label_id__in=labels_map)\
        .annotate(task_id=F('job__segment__task_id'))\
        .order_by('task_id', 'frame')\
        .prefetch_related('labeledshapeattributeval_set')
    serialized_shapes = {}
    for shape in shapes:
        serialized_shape = {
            'label_id': labels_map[shape.label_id]['pk'],
            'group': shape.group,
            'source': shape.source,
            'type': shape.type,
            'occluded': shape.occluded,
            'outside': shape.outside,
            'z_order': shape.z_order,
            'points': shape.points,
            'rotation': shape.rotation,
            'attributes': [{
                'spec_id': labels_map[shape.label_id]['specs'][attr.spec_id] ,
                'value': attr.value,
            } for attr in shape.labeledshapeattributeval_set.all()]
        }
        if shape.task_id in serialized_shapes:
            if shape.frame in serialized_shapes[shape.task_id]:
                serialized_shapes[shape.task_id][shape.frame].append(serialized_shape)
            else:
                serialized_shapes[shape.task_id][shape.frame] = [serialized_shape]
        else:
            serialized_shapes[shape.task_id] = {shape.frame: [serialized_shape]}

    # each task may have its own sorting for files, so frames will be different
    tasks = source_project.tasks.order_by('pk').select_related('data')

    min_quality = tasks[0].data.image_quality
    s3_files = []
    for task in tasks:
        if task.pk in serialized_shapes:
            task_files = sort(
                task.data.s3_files.all(),
                sorting_method=task.data.sorting_method,
                func=lambda f: f.file.name,
            )
            s3_files += [{
                'pk': task_files[frame].pk,
                'shapes': serialized_shapes[task.pk][frame]
            } for frame in serialized_shapes[task.pk]]

        if task.data.image_quality < min_quality:
            min_quality = task.data.image_quality

    return s3_files


@transaction.atomic
def _create_task(project: Project, data, job_size, task_size, i):
    size = min(len(data), task_size)
    data = {item['pk']: item['shapes'] for item in data}
    from_files = S3File.objects.filter(pk__in=data.keys())

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

    files = RemoteFile.objects.bulk_create([
        RemoteFile(
            data=db_data,
            file=file.file.url,
            meta={'from_pk': file.pk}
        )
        for file in from_files
    ], batch_size=100)

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
    files = S3File.objects.filter(data=db_data)
    for file in files:
        shapes = []
        attributes = []
        for shape_data in data[file.meta['from_pk']]:
            attributes_data = shape_data.pop('attributes')
            attributes.append([
                LabeledShapeAttributeVal(
                    spec_id=item['spec_id'],
                    shape_id=0,
                    value=item['value']
                ) for item in attributes_data
            ])
            shapes.append(LabeledShape(**shape_data))

        shapes = LabeledShape.objects.bulk_create(shapes, batch_size=200)
        flat_attributes = []
        for i, shape in enumerate(shapes):
            for attribute in attributes[i]:
                attribute.shape_id = shape.pk
                flat_attributes.append(attribute)
        LabeledShapeAttributeVal.objects.bulk_create(flat_attributes, batch_size=200)


def start(source_id, target_id, labels, job_size=50, n_jobs=10):
    meta_key = f'move_{source_id}_{target_id}_meta'
    data_key = f'move_{source_id}_{target_id}_data'

    cache = redis.Redis.from_url(settings.REDIS_URL)
    try:
        meta = cache.get(meta_key)
        if meta is None:
            logger.info('Starting new import')
            data = _prepare(source_id, target_id, labels)
            meta = {
                'current': 0,
                'total': len(data),
                'job_size': job_size,
                'n_jobs': n_jobs,
            }
            cache.set(meta_key, json.dumps(meta))
            cache.set(data_key, json.dumps(data))
        else:
            logger.info(f'Continue previous import')
            meta = json.loads(meta)
            data = json.loads(cache.get(data_key))

        start_frame = meta['current']
        total = meta['total']
        job_size = meta['job_size']
        task_size = job_size * meta['n_jobs']
        project = Project.objects.get(pk=source_id)

        logger.info(f'{start_frame:06d} / {start_frame:06d}')
        i = 0
        for frame in range(start_frame, total, task_size):
            print(data)
            _create_task(project, data[frame: frame + task_size], job_size, task_size, i)
            meta['current'] = frame
            cache.set(meta_key, json.dumps(meta))
            i += 1
            logger.info(f'{start_frame:06d} / {start_frame:06d}')

        cache.delete(meta_key)
        cache.delete(data_key)
    finally:
        cache.close()


# does not work, until file is imported.
# can not start this from shell.
def start_rq(source_id, target_id, labels, job_size, n_jobs):
    q = django_rq.get_queue('default')
    job_id = f'move_{source_id}_{target_id}_job'
    job: RqJob = q.fetch_job(job_id)

    if job is None or job.is_finished or job.is_failed:
        q.enqueue_call(
            start,
            args=(source_id, target_id, labels, job_size, n_jobs),
            job_id=job_id,
            timeout=172800,
        )
        logger.info('Enqueued move job.')
    else:
        logger.info('Move job is already in progress')


start(12, 1, {
    'Price tag': 'Price tag',
    'Box container': 'Product',
    'Product item': 'Product',
    'text_area': 'Text OCR',
    'Shelf': 'Shelf line',
    'Shelf edge': 'Shelf line'
}, job_size=3, n_jobs=2)
