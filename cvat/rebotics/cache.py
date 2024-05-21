import os
import redis
import json
from io import IOBase
from typing import Iterable
from botocore.exceptions import ClientError
from django.conf import settings

from cvat.rebotics.s3_client import s3_client


class CacheClient:
    def __init__(self, root=settings.CACHE_ROOT):
        self.root = root
        self._cache = redis.Redis.from_url(settings.REDIS_URL)

    def __del__(self):
        self._cache.close()

    def get(self, key, default=None, tag=False):
        data = self._cache.get(key)
        data = {'value': default} if data is None else json.loads(data)
        if tag:
            return data['value'], data.get('tag', None)
        return data['value']

    def set(self, key, value, expire=settings.CACHE_EXPIRE, tag=None):
        data = {'value': value}
        if tag:
            data['tag'] = tag
        return self._cache.set(key, json.dumps(data), ex=expire)

    def delete(self, key):
        return self._cache.delete(key)

    def __contains__(self, key):
        return key in self._cache


class S3CacheClient:
    """Can not work with default MediaCache
    Accepts items as io buff and dictionary of tags
    Return presigned urls to items and dictionary of tags.
    No files go through the server, just urls.
    """
    def get(self, key: str, tags: Iterable[str] | None,
            default: tuple[str, dict] | None = None) -> tuple[str, dict] | None:
        try:
            cache_key = self._key(key)
            if tags is None:
                s3_client.head_object(cache_key)
                tags = {}
            else:
                s3_tags = s3_client.get_tags(cache_key)
                tags = {t: s3_tags.get(t, None) for t in tags}

            return cache_key, tags
        except ClientError:
            return default

    def set(self, key: str, item: IOBase,
            tags: dict[str, str] | None = None) -> tuple[str, dict] | None:
        try:
            cache_key = self._key(key)
            s3_client.upload_from_io(item, cache_key)
            if tags is not None:
                s3_client.set_tags(cache_key, tags)

            return cache_key, tags
        except ClientError:
            return None

    def _key(self, key):
        return os.path.join(settings.S3_CACHE_ROOT, key)


default_cache = CacheClient()
s3_media_cache = S3CacheClient()
