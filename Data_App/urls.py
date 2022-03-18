
from django.urls import path,include
from django.contrib.auth import views as auth_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from Data_App import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
        path('', views.home, name='home'), 
        path('uploadfiles', views.uploadfiles, name='uploadfiles'), 
        path('result', views.i, name='result'), 

]
# if settings.DEBUG:
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns()
