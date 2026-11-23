from django.db import models
from django.contrib.auth.models import User

class ModeloEntrenado(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    fecha = models.DateTimeField(auto_now_add=True)
    n_estudiantes = models.IntegerField()
    ruta_modelo = models.CharField(max_length=255)
    mae = models.FloatField()
    rmse = models.FloatField()
    r2 = models.FloatField()
    r2_pct = models.FloatField()
    tabla_coef = models.JSONField()
    matrices_info = models.JSONField()
    grafica_comparativa = models.TextField()
    grafica_importancia = models.TextField()

    class Meta:
        ordering = ['-fecha']

    def __str__(self):
        return f"Modelo {self.usuario.username} ({self.fecha:%Y-%m-%d %H:%M}) - R²={self.r2}"

class HistorialPrediccion(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE)
    fecha = models.DateTimeField(auto_now_add=True)
    x1 = models.FloatField(verbose_name="Horas de estudio")
    x2 = models.FloatField(verbose_name="Asistencia %")
    x3 = models.FloatField(verbose_name="Tareas %")
    x4 = models.FloatField(verbose_name="Nota previa")
    prediccion = models.FloatField(verbose_name="Calificación predicha")

    class Meta:
        ordering = ['-fecha']

    def __str__(self):
        return f"{self.usuario.username} - {self.prediccion} ({self.fecha:%Y-%m-%d %H:%M})"
