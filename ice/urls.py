from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('api/register/', views.register, name='register'),
    path('api/login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('history/', views.get_history, name='history'),
]