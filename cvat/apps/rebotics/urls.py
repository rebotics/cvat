from rest_framework.routers import DefaultRouter

from .views import RetailerImportViewSet, RetailerExportViewSet

router = DefaultRouter(trailing_slash=False)
router.register('retailer_import', RetailerImportViewSet, basename='retailer_import')
router.register('retailer_export', RetailerExportViewSet, basename='retailer_export')

app_name = 'rebotics'

urlpatterns = router.urls
