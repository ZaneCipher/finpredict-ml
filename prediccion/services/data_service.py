import pandas as pd

COLUMNAS_ESPERADAS = ['X1', 'X2', 'X3', 'X4', 'Y']

def cargar_y_validar_csv(archivo):
    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        return None, f"No se pudo leer el CSV: {e}"

    if not all(col in df.columns for col in COLUMNAS_ESPERADAS):
        return None, "El archivo CSV debe contener exactamente las columnas: X1, X2, X3, X4, Y."

    df = df[COLUMNAS_ESPERADAS].dropna()
    if len(df) <= 5:
        return None, "El CSV necesita al menos 6 estudiantes válidos para entrenar el modelo."

    return df, None
