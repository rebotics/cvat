import random
from enum import Enum
from django.conf import settings
from django.db import models
from django.utils.translation import ugettext_lazy as _


class InjectionError(AttributeError):
    pass


def injected_property(name: str, error_message: str = None):
    """
    Allows to explicitly mark injected properties on base classes instead of direct monkey patching.
    Replaces AttributeError with more concrete InjectionError on get and suppresses it on delete.
    """
    def inner(cls):
        if name.startswith('_'):
            raise ValueError('Name should not be protected, private or magic.')

        private_name = '_{}'.format(name)
        message = error_message
        if message is None:
            message = "{}'s {} is not set.".format(cls.__name__, name)

        @property
        def prop(obj):
            try:
                return getattr(obj, private_name)
            except AttributeError:
                raise InjectionError(message)

        @prop.setter
        def prop(obj, value):
            setattr(obj, private_name, value)

        @prop.deleter
        def prop(obj):
            try:
                delattr(obj, private_name)
            except AttributeError:
                pass

        setattr(cls, name, prop)

        return cls
    return inner


class ChoicesEnum(Enum):
    @classmethod
    def choices(cls):
        return (x.value for x in cls)


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


def setting(name, default=None):
    # obtain settings which may be not initialized.
    return getattr(settings, name, default)


class DateAwareModel(models.Model):
    date_created = models.DateTimeField(auto_now_add=True, verbose_name=_("Date created"))
    date_modified = models.DateTimeField(auto_now=True, verbose_name=_("Date modified"))
    date_display_format = "%Y-%m-%d %H:%M:%S"

    class Meta:
        abstract = True

    @property
    def date_created_display(self):
        return self.date_created.strftime(self.date_display_format)

    @property
    def date_modified_display(self):
        return self.date_modified.strftime(self.date_display_format)


def fix_coordinates(item, width, height):
    if item['lowerx'] > item['upperx']:
        item['lowerx'], item['upperx'] = item['upperx'], item['lowerx']
    if item['lowery'] > item['uppery']:
        item['lowery'], item['uppery'] = item['uppery'], item['lowery']
    if item['lowerx'] < 0:
        item['lowerx'] = 0
    if item['upperx'] > width:
        item['upperx'] = width
    if item['lowery'] < 0:
        item['lowery'] = 0
    if item['uppery'] > height:
        item['uppery'] = height


def rand_color():
    color = '#'
    for _ in range(6):
        color += random.choice('0123456789ABCDEF')
    return color


def save_ranges(data: list, ranges_sep=',', start_end_sep='-'):
    result = []
    if len(data) > 0:
        start = 0
        end = 0
        for i in range(1, len(data)):
            if data[i] - data[end] > 1:
                result.append(str(data[start]) if start == end else f'{data[start]}{start_end_sep}{data[end]}')
                start = i
            end = i
        result.append(str(data[start]) if start == end else f'{data[start]}{start_end_sep}{data[end]}')
    return ranges_sep.join(result)


def load_ranges(ranges: str, ranges_sep=',', start_end_sep='-'):
    result = []
    if len(ranges) > 0:
        for i in ranges.split(ranges_sep):
            if start_end_sep in i:
                start, end = i.split(start_end_sep)
                result += list(range(int(start), int(end) + 1))
            else:
                result.append(int(i))
    return result


def add_to_ranges(ranges: str, num: int, ranges_sep=',', start_end_sep='-'):
    if len(ranges) > 0:
        parts = ranges.rsplit(ranges_sep, 1)
        last = parts[-1]
        if start_end_sep in last:
            start, end = last.split(start_end_sep)
        else:
            start = end = last
        if num - int(end) > 1:
            parts.append(str(num))
        else:
            parts[-1] = f'{start}{start_end_sep}{num}'
        return ranges_sep.join(parts)
    return str(num)
