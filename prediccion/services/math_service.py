import numpy as np

def calcular_ecuacion_normal(X, Y):
    X_mat = np.c_[np.ones(X.shape[0]), X]
    Y_vec = Y.values
    X_T = X_mat.T
    X_T_X = X_T.dot(X_mat)

    metodo = "Inversa estándar (np.linalg.inv)"
    try:
        X_T_X_inv = np.linalg.inv(X_T_X)
    except np.linalg.LinAlgError:
        X_T_X_inv = np.linalg.pinv(X_T_X)
        metodo = "Pseudoinversa de Moore-Penrose (np.linalg.pinv) por colinealidad"

    beta = X_T_X_inv.dot(X_T).dot(Y_vec)

    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    matrices_info = {
        'metodo': metodo,
        'X_shape': f"{X_mat.shape[0]} × {X_mat.shape[1]}",
        'X_preview': np.array2string(X_mat[:6]),
        'XT_shape': f"{X_T.shape[0]} × {X_T.shape[1]}",
        'XT_preview': np.array2string(X_T[:, :6]),
        'XTX_shape': f"{X_T_X.shape[0]} × {X_T_X.shape[1]}",
        'XTX_str': np.array2string(X_T_X),
        'XTX_inv_str': np.array2string(X_T_X_inv),
    }
    return beta, matrices_info
