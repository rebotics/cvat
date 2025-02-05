from rest_framework.routers import DefaultRouter

from .views import RetailerImportViewset


class AnySlashRouter(DefaultRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trailing_slash='/?'


router = AnySlashRouter()
router.register('retailer_import', RetailerImportViewset, basename='retailer_import_viewset')

app_name = 'rebotics'

urlpatterns = router.urls
