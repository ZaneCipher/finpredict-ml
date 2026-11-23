from django.urls import path
from . import views

urlpatterns = [
    path('', views.subir_datos, name='subir_datos'),
    path('resultados/', views.resultados_prediccion, name='resultados_prediccion'),
    path('api/predecir/', views.api_predecir, name='api_predecir'),
    path('exportar-csv/', views.exportar_historial_csv, name='exportar_historial_csv'),
]
