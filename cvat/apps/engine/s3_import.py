import os

import cvat.apps.dataset_manager as dm
import cvat.apps.dataset_manager.views  # pylint: disable=unused-import
from cvat.apps.engine.log import slogger
from cvat.apps.engine.backup import _import_task, _import_project

from cvat.rebotics.s3_client import s3_client
from cvat.rebotics.utils import retry


class ImportType:
    TASK_ANNOTATIONS = 'task_annotations'
    JOB_ANNOTATIONS = 'job_annotations'
    PROJECT_DATASET = 'project_dataset'
    TASK_BACKUP = 'task_backup'
    PROJECT_BACKUP = 'project_backup'


_func_mapping = {
    ImportType.TASK_ANNOTATIONS: dm.task.import_task_annotations,
    ImportType.JOB_ANNOTATIONS: dm.task.import_job_annotations,
    ImportType.PROJECT_DATASET: dm.project.import_dataset_as_project,
    ImportType.TASK_BACKUP: _import_task,
    ImportType.PROJECT_BACKUP: _import_project,
}


def import_from_s3(obj_pk: int, s3_key: str, import_type: str, format_name: str):
    slogger.glob.info(f'Downloading {s3_key}')
    filename = retry(s3_client.download_to_temp, args=(s3_key,),
                     kwargs={'prefix': f'cvat_{import_type}_{obj_pk}_'},
                     times=5, delay=5, factor=2)
    try:
        import_func = _func_mapping.get(import_type)
        slogger.glob.info(f'Importing {import_type} {obj_pk}')
        return import_func(obj_pk, filename, format_name)
    finally:
        try:
            slogger.glob.info(f'Removing {filename}')
            os.remove(filename)
        except FileNotFoundError as e:
            slogger.glob.warn(e)

        try:
            slogger.glob.info(f'Deleting {s3_key}')
            s3_client.delete_object(s3_key)
        except Exception as e:
            slogger.glob.warn(e)


def restore_s3_backup(s3_key: str, import_type: ImportType, user_id: int, org_id: int):
    slogger.glob.info(f'Downloading {s3_key}')
    filename = retry(s3_client.download_to_temp, args=(s3_key,),
                     kwargs={'prefix': f'cvat_{import_type}_'},
                     times=5, delay=5, factor=2)
    try:
        import_func = _func_mapping.get(import_type)
        slogger.glob.info(f'Restoring {import_type}')
        return import_func(filename, user_id, org_id)
    finally:
        try:
            slogger.glob.info(f'Removing {filename}')
            os.remove(filename)
        except FileNotFoundError as e:
            slogger.glob.warn(e)

        try:
            slogger.glob.info(f'Deleting {s3_key}')
            s3_client.delete_object(s3_key)
        except Exception as e:
            slogger.glob.warn(e)
