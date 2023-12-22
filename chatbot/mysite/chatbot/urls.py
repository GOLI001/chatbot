from django.urls import path
from .views import get_response

urlpatterns = [
    path('get_response/', get_response, name='get_response'),
]
