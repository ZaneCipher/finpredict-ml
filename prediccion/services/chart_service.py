import io
import urllib.parse
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    uri = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close(fig)
    return uri

def generar_grafica_comparativa(Y, Y_pred):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#0b0f19')

    indices = np.arange(1, len(Y) + 1)
    ax.plot(indices, Y.values, label='Ingresos Reales (Y)', color='#22d3ee', marker='o', linewidth=2, markersize=6)
    ax.plot(indices, Y_pred, label='Ingresos Predichos (Ŷ)', color='#e879f9', marker='x', linestyle='--', linewidth=2, markersize=7)
    ax.set_title('Ingresos Mensuales Reales vs. Predichos ($)', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Mes / Periodo #', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Ingresos ($ USD)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(color='#1e293b', linestyle=':', linewidth=1)
    ax.legend(facecolor='#111827', edgecolor='#334155', labelcolor='#f8fafc')
    return _fig_to_uri(fig)

def generar_grafica_importancia(coefs):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#0b0f19')

    nombres = ['Marketing $ (X1)', 'Gastos Op. $ (X2)', 'Clientes Act. (X3)', 'Ventas Prev. $ (X4)']
    valores = [float(c) for c in coefs]
    colores = ['#22d3ee' if v >= 0 else '#f43f5e' for v in valores]

    bars = ax.barh(nombres, valores, color=colores, height=0.55)
    ax.axvline(0, color='#475569', linewidth=1)
    ax.set_title('Impacto de cada Variable Financiera (Coeficientes β)', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Retorno marginal ($ por unidad)', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='#cbd5e1')
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.grid(axis='x', color='#1e293b', linestyle=':', linewidth=1)

    for bar, val in zip(bars, valores):
        ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right',
                color='#f8fafc', fontsize=9, fontweight='bold')
    return _fig_to_uri(fig)
