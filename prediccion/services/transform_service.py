import numpy as np

# Figura base 2D (una flecha/casa asimétrica para notar bien rotación y sesgado)
FIGURA_BASE = np.array([
    [0, 0, 1],
    [2, 0, 1],
    [2, 2, 1],
    [1, 3, 1],
    [0, 2, 1],
    [0, 0, 1],
], dtype=float).T  # Matriz 3 x N

def matriz_traslacion(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

def matriz_escalado(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=float)

def matriz_rotacion(grados):
    rad = np.radians(grados)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)

def matriz_sesgado(shx, shy):
    return np.array([
        [1,   shx, 0],
        [shy, 1,   0],
        [0,   0,   1]
    ], dtype=float)

def aplicar_transformaciones(tx=0, ty=0, sx=1, sy=1, angulo=0, shx=0, shy=0):
    T = matriz_traslacion(tx, ty)
    R = matriz_rotacion(angulo)
    Sh = matriz_sesgado(shx, shy)
    S = matriz_escalado(sx, sy)

    # Composición matricial
    M = T @ R @ Sh @ S
    transformada = M @ FIGURA_BASE
    return FIGURA_BASE, transformada, M
