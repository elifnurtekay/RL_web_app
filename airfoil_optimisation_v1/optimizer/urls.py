from django.urls import path
from .views import DashboardView, optimize_airfoil

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('api/optimize/', optimize_airfoil, name='optimize-airfoil'),
]
