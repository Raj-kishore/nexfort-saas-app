from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('your-name', views.get_name),
    path('thanks', views.thanks),
    path('train', views.train),

]