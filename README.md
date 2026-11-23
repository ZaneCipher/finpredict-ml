# FinPredict ML — Sistema de Predicción Financiera y Transformaciones Lineales 2D

**FinPredict ML** es un sistema computacional desarrollado en Python que predice los **ingresos mensuales** de una empresa mediante **Regresión Lineal Múltiple**, valida matemáticamente sus coeficientes a través de la **Ecuación Normal** de Álgebra Lineal y demuestra de forma interactiva las **transformaciones lineales en 2D** usando coordenadas homogéneas.

El proyecto cuenta con una arquitectura modular basada en una **Capa de Servicios (`services/`)** que alimenta simultáneamente dos interfaces:
1. **Aplicación de Escritorio (`app_escritorio.py`):** Interfaz gráfica moderna en **Tkinter / CustomTkinter** con predicción financiera y simulador de transformaciones 2D.
2. **Plataforma Web (`Django`):** Sistema con autenticación de usuarios, persistencia del modelo en disco (`joblib`), historial en SQLite, predicciones asíncronas (AJAX) y exportación a CSV.

---

## Fundamento Matemático

### 1. Modelo Predictivo (Regresión Lineal Múltiple)
El sistema estima el ingreso mensual ($\hat{Y}$) a partir de cuatro variables financieras:
* $X_1$: Inversión en Marketing (USD)
* $X_2$: Gastos Operativos (USD)
* $X_3$: Clientes Activos (Cantidad)
* $X_4$: Ventas del Mes Anterior (USD)

$$\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4$$

Los parámetros óptimos $\beta$ se entrenan con **Scikit-Learn** y se verifican manualmente con **NumPy** resolviendo la **Ecuación Normal**:

$$\beta = (X^T X)^{-1} X^T Y$$

*(Ante posible colinealidad o matriz singular, el sistema aplica automáticamente la pseudoinversa de Moore-Penrose).*

El desempeño del modelo se evalúa mediante $MAE$, $RMSE$ y el coeficiente de determinación $R^2$. Además, se calcula la **utilidad neta estimada** como $\hat{Y} - (X_1 + X_2)$ para determinar la rentabilidad proyectada.

### 2. Transformaciones Lineales en 2D
Para unificar la **traslación**, **escalado**, **rotación** y **sesgado** en una sola multiplicación matricial, se emplean **coordenadas homogéneas** y matrices $3 \times 3$:

$$M = T(t_x, t_y) \cdot R(\theta) \cdot Sh(sh_x, sh_y) \cdot S(s_x, s_y)$$

---

## Estructura del Proyecto

```text
finpredict-ml/
├── app_escritorio.py             # Interfaz gráfica de escritorio (Tkinter / CustomTkinter)
├── datos_financieros.csv         # Dataset ficticio y anonimizado (15 periodos)
├── manage.py                     # Gestor de comandos de Django
├── requirements.txt              # Dependencias del proyecto
├── core/                         # Configuración principal de Django (settings, urls)
└── prediccion/
    ├── models.py                 # Tablas SQLite: ModeloEntrenado e HistorialPrediccion
    ├── forms.py                  # Validación de archivos CSV y rangos numéricos
    ├── views.py                  # Controladores web y endpoint JSON (/api/predecir/)
    ├── templates/                # Interfaz web responsiva (Bootstrap 5)
    └── services/                 # Capa de Servicios reutilizable
        ├── data_service.py       # Lectura y validación de CSV (Pandas)
        ├── math_service.py       # Ecuación Normal y matrices paso a paso (NumPy)
        ├── ml_service.py         # Entrenamiento, métricas y persistencia (Scikit-Learn + Joblib)
        ├── chart_service.py      # Gráficas en memoria Base64 (Matplotlib)
        └── transform_service.py  # Matrices homogéneas 3x3 para transformaciones 2D
```

---

## Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/ZaneCipher/finpredict-ml.git
cd finpredict-ml
```

### 2. Crear y activar el entorno virtual
En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```
*(Nota para usuarios de Linux/Arch: si falta Tkinter a nivel de sistema, instálalo con `sudo pacman -S tk` o `sudo apt install python3-tk`).*

---

## Cómo Ejecutar el Sistema

### Opción A: Aplicación de Escritorio (Tkinter + Transformaciones 2D)
```bash
python app_escritorio.py
```
* Al abrirse, carga automáticamente `datos_financieros.csv`, entrena el modelo y muestra las métricas, gráfica comparativa y coeficientes $\beta$.
* En la pestaña **2. Transformaciones Lineales 2D** puedes usar los botones rápidos o los deslizadores para probar traslación, escalado, rotación y sesgado en tiempo real.

### Opción B: Plataforma Web (Django)
1. Aplicar migraciones de la base de datos SQLite:
   ```bash
   python manage.py migrate
   ```
2. Crear un usuario administrador para iniciar sesión:
   ```bash
   python manage.py createsuperuser
   ```
3. Iniciar el servidor local:
   ```bash
   python manage.py runserver
   ```
4. Abrir en el navegador `http://127.0.0.1:8000/`, iniciar sesión y subir `datos_financieros.csv`.

---

## Datos y Ética
El archivo `datos_financieros.csv` contiene registros completamente **ficticios y anonimizados** con fines exclusivamente académicos y de demostración computacional.

---

## Autor
**Luis Alberto Chura Zegarra**  
Carrera de Ingeniería de Sistemas — Universidad Adventista de Bolivia (UAB)