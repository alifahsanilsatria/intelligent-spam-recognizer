from django.conf.urls import url
from .views import index, main

urlpatterns = [
    # index
    url(r'^$', index, name='index'),
    url(r'^main/$', main, name='main')
]