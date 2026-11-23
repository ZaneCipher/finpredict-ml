import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .math_service import calcular_ecuacion_normal
from .chart_service import generar_grafica_comparativa, generar_grafica_importancia

def entrenar_y_guardar_modelo(df, user_id):
    X = df[['X1', 'X2', 'X3', 'X4']]
    Y = df['Y']

    modelo = LinearRegression()
    modelo.fit(X, Y)
    Y_pred = modelo.predict(X)

    mae = mean_absolute_error(Y, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    r2 = r2_score(Y, Y_pred)

    beta, matrices_info = calcular_ecuacion_normal(X, Y)

    etiquetas = [
        ('β₀ (Ingreso Base / Intercepto)', modelo.intercept_, beta[0]),
        ('β₁ — Inversión Marketing $ (X1)', modelo.coef_[0], beta[1]),
        ('β₂ — Gastos Operativos $ (X2)', modelo.coef_[1], beta[2]),
        ('β₃ — Clientes Activos (X3)', modelo.coef_[2], beta[3]),
        ('β₄ — Ventas Mes Anterior $ (X4)', modelo.coef_[3], beta[4]),
    ]
    tabla_coef = [
        {'nombre': n, 'sklearn': round(float(s), 4), 'numpy': round(float(b), 4)}
        for n, s, b in etiquetas
    ]

    os.makedirs('media/modelos', exist_ok=True)
    ruta_modelo = f'media/modelos/modelo_user_{user_id}.joblib'
    joblib.dump(modelo, ruta_modelo)

    return {
        'n_estudiantes': len(df),
        'ruta_modelo': ruta_modelo,
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'r2': round(float(r2), 4),
        'r2_pct': round(float(r2) * 100, 2),
        'tabla_coef': tabla_coef,
        'matrices_info': matrices_info,
        'grafica_comparativa': generar_grafica_comparativa(Y, Y_pred),
        'grafica_importancia': generar_grafica_importancia(modelo.coef_),
    }

def predecir_con_modelo(ruta_modelo, x1, x2, x3, x4):
    modelo = joblib.load(ruta_modelo)
    nuevos_datos = pd.DataFrame([[x1, x2, x3, x4]], columns=['X1', 'X2', 'X3', 'X4'])
    pred_raw = float(modelo.predict(nuevos_datos)[0])

    aviso_rango = None
    if pred_raw < 0:
        aviso_rango = f"El modelo lineal calculó ${round(pred_raw, 2)}; se ajustó a $0.00 como piso de ingresos."
        pred = 0.0
    else:
        pred = round(pred_raw, 2)

    return pred, aviso_rango
