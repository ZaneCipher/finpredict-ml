import csv
import json
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from .forms import UploadCSVForm, PrediccionForm
from .models import ModeloEntrenado, HistorialPrediccion
from .services.data_service import cargar_y_validar_csv
from .services.ml_service import entrenar_y_guardar_modelo, predecir_con_modelo

@login_required
def subir_datos(request):
    datos_html = None
    error_msg = None
    n_registros = None

    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            df, error_msg = cargar_y_validar_csv(form.cleaned_data['archivo_csv'])
            if not error_msg:
                datos_html = df.to_html(
                    classes='table table-dark table-hover align-middle text-center mb-0',
                    index=False
                )
                datos_modelo = entrenar_y_guardar_modelo(df, request.user.id)
                ModeloEntrenado.objects.create(usuario=request.user, **datos_modelo)
                n_registros = len(df)
        else:
            error_msg = "Debes seleccionar un archivo CSV válido."

    modelo_listo = ModeloEntrenado.objects.filter(usuario=request.user).exists()
    return render(request, 'prediccion/subir_datos.html', {
        'datos_html': datos_html,
        'error_msg': error_msg,
        'n_registros': n_registros,
        'modelo_listo': modelo_listo,
    })

@login_required
def resultados_prediccion(request):
    modelo_db = ModeloEntrenado.objects.filter(usuario=request.user).first()
    if not modelo_db:
        return redirect('subir_datos')

    historial = HistorialPrediccion.objects.filter(usuario=request.user)[:15]
    return render(request, 'prediccion/resultados.html', {
        'res': modelo_db,
        'historial': historial,
    })

@login_required
@require_POST
def api_predecir(request):
    modelo_db = ModeloEntrenado.objects.filter(usuario=request.user).first()
    if not modelo_db:
        return JsonResponse({'error': 'No hay un modelo entrenado.'}, status=400)

    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'JSON inválido.'}, status=400)

    form = PrediccionForm(payload)
    if not form.is_valid():
        return JsonResponse({'error': 'Verifica que todos los montos y cantidades sean números mayores o iguales a 0.'}, status=400)

    d = form.cleaned_data
    prediccion, aviso_rango = predecir_con_modelo(
        modelo_db.ruta_modelo, d['x1'], d['x2'], d['x3'], d['x4']
    )

    costos_totales = d['x1'] + d['x2']
    utilidad = round(prediccion - costos_totales, 2)

    registro = HistorialPrediccion.objects.create(
        usuario=request.user,
        x1=d['x1'], x2=d['x2'], x3=d['x3'], x4=d['x4'],
        prediccion=prediccion
    )

    return JsonResponse({
        'prediccion': prediccion,
        'utilidad': utilidad,
        'rentable': utilidad > 0,
        'aviso_rango': aviso_rango,
        'registro': {
            'fecha': registro.fecha.strftime('%Y-%m-%d %H:%M'),
            'x1': registro.x1,
            'x2': registro.x2,
            'x3': registro.x3,
            'x4': registro.x4,
            'prediccion': registro.prediccion,
        }
    })

@login_required
def exportar_historial_csv(request):
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="historial_financiero.csv"'

    writer = csv.writer(response)
    writer.writerow(['Fecha', 'Usuario', 'X1_Marketing', 'X2_GastosOp', 'X3_Clientes', 'X4_VentasPrev', 'Ingreso_Proyectado'])

    for r in HistorialPrediccion.objects.filter(usuario=request.user):
        writer.writerow([r.fecha.strftime('%Y-%m-%d %H:%M'), r.usuario.username, r.x1, r.x2, r.x3, r.x4, r.prediccion])
    return response
