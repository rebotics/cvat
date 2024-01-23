from django.conf import settings
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from cvat.apps.organizations.models import Organization
from cvat.apps.engine.serializers import RqStatusSerializer


class _BaseIESerializer(serializers.Serializer):

    def create(self, validated_data):
        raise NotImplementedError('Creating export data is not allowed')

    def update(self, instance, validated_data):
        raise NotImplementedError('Updating export data is not allowed')


class _ImportAnnotationSerializer(_BaseIESerializer):
    lowerx = serializers.FloatField()
    lowery = serializers.FloatField()
    upperx = serializers.FloatField()
    uppery = serializers.FloatField()
    label = serializers.CharField(max_length=128)
    points = serializers.CharField(max_length=255, allow_null=True, default=None)
    type = serializers.CharField(max_length=255, allow_null=True, default=None)
    upc = serializers.CharField(max_length=128, allow_blank=True, allow_null=True, default=None)


class _ImportPriceTagSerializer(_BaseIESerializer):
    lowerx = serializers.FloatField(allow_null=True, default=None)
    lowery = serializers.FloatField(allow_null=True, default=None)
    upperx = serializers.FloatField(allow_null=True, default=None)
    uppery = serializers.FloatField(allow_null=True, default=None)
    label = serializers.CharField(max_length=128)
    points = serializers.CharField(max_length=255, allow_null=True, default=None)
    type = serializers.CharField(max_length=255, allow_null=True, default=None)
    upc = serializers.CharField(max_length=128, allow_blank=True, allow_null=True, default=None)


class _ImportImageSerializer(_BaseIESerializer):
    items = serializers.ListSerializer(child=_ImportAnnotationSerializer())
    image = serializers.URLField(allow_null=True, default=None)
    planogram_title = serializers.CharField(allow_null=True, default=None)
    processing_action_id = serializers.IntegerField(allow_null=True, default=None)
    price_tags = serializers.ListSerializer(child=_ImportPriceTagSerializer(),
                                            default=None, allow_null=True)


class ImportSerializer(_BaseIESerializer):
    image_quality = serializers.IntegerField(min_value=0, max_value=100, default=70)
    segment_size = serializers.IntegerField(min_value=0, default=0)
    workspace = serializers.CharField(max_length=16, allow_null=True, default=settings.IMPORT_WORKSPACE,
                                      help_text='Organization short name. Case sensitive.')
    export_by = serializers.CharField(allow_null=True, default=None)
    retailer_codename = serializers.CharField(allow_null=True, default=None)
    images = serializers.ListSerializer(child=_ImportImageSerializer())

    def validate_workspace(self, value):
        if value is None or Organization.objects.filter(slug=value).exists():
            return value
        raise ValidationError(f'Workspace "{value}" does not exist!')


class _ImportResponseImageSerializer(_BaseIESerializer):
    id = serializers.IntegerField()
    image = serializers.URLField()


class ImportResponseSerializer(_BaseIESerializer):
    task_id = serializers.IntegerField(allow_null=True, default=None)
    preview = serializers.URLField(allow_null=True, default=None)
    images = serializers.ListSerializer(child=_ImportResponseImageSerializer(),
                                        allow_null=True, default=None)
    status = RqStatusSerializer(allow_null=True, default=None)


class ExportSerializer(_BaseIESerializer):
    task_ids = serializers.ListSerializer(child=serializers.IntegerField(), allow_empty=False)
    start_frame = serializers.IntegerField(default=0)
    retailer_name = serializers.CharField(max_length=50)
    store_id = serializers.IntegerField()


class ExportResponseSerializer(_BaseIESerializer):
    rq_id = serializers.CharField(max_length=200)


class ExportAuthSerializer(_BaseIESerializer):
    retailer_host = serializers.CharField(max_length=200)
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(max_length=128)
    verification_code = serializers.CharField(max_length=20, allow_null=True, default=None)


class ExportAuthResponseSerializer(_BaseIESerializer):
    retailer_name = serializers.CharField(max_length=50)


class ExportCheckSerializer(_BaseIESerializer):
    status = serializers.CharField(max_length=20)
    scan_ids = serializers.ListSerializer(child=serializers.IntegerField())
    last_task_id = serializers.IntegerField(allow_null=True, default=None)
    last_frame = serializers.IntegerField(allow_null=True, default=None)
    message = serializers.CharField(max_length=3000, allow_blank=True)
