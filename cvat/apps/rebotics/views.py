from rest_framework.viewsets import GenericViewSet
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import action

from drf_spectacular.utils import extend_schema, extend_schema_view

from django.http.response import Http404, HttpResponsePermanentRedirect
from django.shortcuts import render
from django.conf import settings
from django.urls import resolve, Resolver404
from django.views.decorators.common import no_append_slash

from .authentication import RetailerAuthentication
from .serializers import ImportSerializer, ImportResponseSerializer, ExportSerializer, \
    ExportResponseSerializer, ExportAuthSerializer, ExportAuthResponseSerializer, ExportCheckSerializer
from cvat.apps.rebotics.tasks import retailer_import as import_api
from cvat.apps.rebotics.tasks import retailer_export as export_api


@extend_schema(tags=['retailer_import'])
@extend_schema_view(
    create=extend_schema(
        summary='Start dataset import from retailer instance. Returns task id to track import progress.',
        request=ImportSerializer,
        responses={
            '202': ImportResponseSerializer,
        },
    ),
    retrieve=extend_schema(
        summary='Check import status and progress. Returns images\' description if import is finished.',
        responses={
            '200': ImportResponseSerializer,
        },
    ),
)
class RetailerImportViewSet(GenericViewSet):
    authentication_classes = [RetailerAuthentication, ]
    permission_classes = [IsAuthenticated, ]

    def create(self, request, *args, **kwargs):
        serializer = ImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        task_id = import_api.create(serializer.data, request.user)
        serializer = ImportResponseSerializer(data={'task_id': task_id})
        serializer.is_valid(raise_exception=True)
        return Response(data=serializer.data, status=status.HTTP_202_ACCEPTED)

    def retrieve(self, request, *args, **kwargs):
        task_id = self.kwargs.get('pk')
        task_data = import_api.check(task_id)
        if task_data is None:
            raise Http404
        serializer = ImportResponseSerializer(data=task_data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)


@extend_schema(tags=['retailer_export'])
@extend_schema_view(
    create=extend_schema(
        summary='Start images export from selected tasks to retailer instance. '
                'Returns rq job id to track export progress.',
        request=ExportSerializer,
        responses={
            '202': ExportResponseSerializer,
        }
    ),
    auth=extend_schema(
        summary='Provide username and password for CVAT to obtain token from retailer instance. '
                'Username and password are not saved anywhere. Token is saved for 90 days. '
                'Returns detected retailer name.',
        request=ExportAuthSerializer,
        responses={
            '200': ExportAuthResponseSerializer,
        },
    ),
    retrieve=extend_schema(
        summary='Check export status and progress. Returns scan ids once export is finished.',
        responses={
            '200': ExportCheckSerializer,
        }
    ),
)
class RetailerExportViewSet(GenericViewSet):
    @action(methods=['POST'], detail=False, url_path=r'^auth/?$')
    def retailer_auth(self, request, *args, **kwargs):
        serializer = ExportAuthSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        retailer_name = export_api.auth(user=request.user, **serializer.validated_data)
        return Response({'retailer_name': retailer_name})

    def create(self, request, *args, **kwargs):
        serializer = ExportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        rq_id = export_api.start(user=request.user, **serializer.validated_data)
        serializer = ExportResponseSerializer(data={'rq_id': rq_id})
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data, status=status.HTTP_202_ACCEPTED)

    @action(methods=['GET'], detail=False, url_path=r'^(?P<retailer_name>\w+)/(?P<export_hash>\w+)/?$')
    def check_status(self, request, *args, **kwargs):
        rq_data = export_api.check(request.path)
        if rq_data is None:
            raise Http404
        serializer = ExportCheckSerializer(data=rq_data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data)


@no_append_slash
def index_view(request, url, *args, **kwargs):
    # same as django.contrib.admin.sites.AdminSite.catch_all_view
    # except returns index.html when url is not found.
    if settings.APPEND_SLASH and not url.endswith('/'):
        urlconf = getattr(request, 'urlconf', None)
        try:
            match = resolve('%s/' % request.path_info, urlconf)
        except Resolver404:
            pass
        else:
            if getattr(match.func, 'should_append_slash', True):
                return HttpResponsePermanentRedirect('%s/' % request.path)
    return render(request, 'index.html')
