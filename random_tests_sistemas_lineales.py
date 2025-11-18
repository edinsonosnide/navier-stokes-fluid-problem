import numpy as np

# ==========================
# Utilidades
# ==========================
def is_square(A):
    """Devuelve True si A es cuadrada."""
    A = np.asarray(A)
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def is_diag_dominant(A, strict=False):
    """
    Verifica si A es diagonalmente dominante por filas.
    - strict=False: |a_ii| >= sum_{j!=i} |a_ij|
    - strict=True : |a_ii| >  sum_{j!=i} |a_ij|
    """
    A = np.asarray(A, dtype=float)
    if not is_square(A):
        raise ValueError("A debe ser cuadrada.")
    for i in range(A.shape[0]):
        diag = abs(A[i, i])
        rest = np.sum(np.abs(A[i, :])) - diag
        if (strict and not (diag > rest)) or (not strict and not (diag >= rest)):
            return False
    return True

def _validate_inputs(A, b, x0):
    """
    Chequeos básicos de forma y diagonal no nula (requerido por los tres esquemas).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if not is_square(A):
        raise ValueError("A debe ser cuadrada.")
    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError("Dimensión de b incompatible con A.")
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(n)
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("Hay ceros (o casi ceros) en la diagonal de A.")
    return A, b, x

def _stopped(x_new, x_old, tol):
    """
    Criterio de paro: ||x^{(k+1)} - x^{(k)}||_∞ < tol
    (puedes cambiarlo por residuo si lo prefieres).
    """
    return np.linalg.norm(x_new - x_old, ord=np.inf) < tol


# ==========================
# Jacobi (n x n)
# ==========================
def jacobi(A, b, x0=None, tol=1e-8, max_iter=500, return_history=False):
    r"""
    Método de Jacobi para Ax=b.

    Fórmulas matriciales:
    ---------------------
    Descomposición: A = D + L + U, con:
      - D: diagonal(A)
      - L: parte estrictamente inferior
      - U: parte estrictamente superior

    **Iteración de Jacobi:**
        x^{(k+1)} = D^{-1} [ b - (L + U) x^{(k)} ]

    Implementación:
    ---------------
    - Calculamos D, R := L + U = A - D
    - Usamos D_inv @ (b - R @ x) para actualizar x.
    """
    A, b, x = _validate_inputs(A, b, x0)
    D = np.diag(np.diag(A))       # D
    R = A - D                     # L + U
    D_inv = np.diag(1.0 / np.diag(D))  # D^{-1}

    if return_history:
        history = [x.copy()]

    for k in range(1, max_iter + 1):
        # x^{(k+1)} = D^{-1} (b - (L+U) x^{(k)})
        x_new = D_inv @ (b - R @ x)

        if _stopped(x_new, x, tol):
            if return_history:
                history.append(x_new.copy())
                return x_new, k, history
            return x_new, k
        x = x_new
        if return_history:
            history.append(x.copy())

    if return_history:
        return x, max_iter, history
    return x, max_iter


# ==========================
# Gauss–Seidel (n x n)
# ==========================
def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=500, return_history=False):
    r"""
    Método de Gauss–Seidel para Ax=b.

    Fórmulas matriciales:
    ---------------------
    Descomposición: A = D + L + U

    **Forma compacta:**
        (D + L) x^{(k+1)} = b - U x^{(k)}
        x^{(k+1)} = (D + L)^{-1} [ b - U x^{(k)} ]

    Implementación:
    ---------------
    - No invertimos (D+L) explícitamente. En cambio, actualizamos componente a componente:
        Para i=1..n:
          suma = Σ_j a_{ij} x_j  - a_{ii} x_i
          x_i^{(k+1)} = (b_i - suma)/a_{ii}
      donde para j<i ya usamos x^{(k+1)} (valores “nuevos”), y para j>i usamos x^{(k)}.
    """
    A, b, x = _validate_inputs(A, b, x0)
    n = A.shape[0]

    if return_history:
        history = [x.copy()]

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            # suma = (A[i,:]·x) - a_ii x_i
            # Nota: x ya contiene valores "nuevos" en índices < i
            suma = A[i, :].dot(x) - A[i, i] * x[i]
            x[i] = (b[i] - suma) / A[i, i]

        if _stopped(x, x_old, tol):
            if return_history:
                history.append(x.copy())
                return x, k, history
            return x, k
        if return_history:
            history.append(x.copy())

    if return_history:
        return x, max_iter, history
    return x, max_iter


# ==========================
# SOR (n x n)
# ==========================
def sor(A, b, omega=1.0, x0=None, tol=1e-8, max_iter=500, return_history=False):
    r"""
    Método SOR (Successive Over-Relaxation) para Ax=b.

    Parámetro:
    ----------
    - omega (ω) en (0,2). ω=1 => Gauss–Seidel.

    Fórmulas matriciales:
    ---------------------
    Descomposición: A = D + L + U

    **Forma compacta de SOR:**
        (D + ωL) x^{(k+1)} = ω b - [ ωU + (ω - 1) D ] x^{(k)}
        x^{(k+1)} = (D + ωL)^{-1} [ ω b - (ωU + (ω - 1) D) x^{(k)} ]

    Implementación (componente a componente):
    -----------------------------------------
    - Análogo a Gauss–Seidel, pero combinamos el valor anterior x_i con el
      nuevo "candidato" (b_i - Σ_{j≠i} a_{ij} x_j)/a_{ii} usando ω:
        x_i ← (1 - ω) x_i + (ω / a_{ii}) (b_i - Σ_{j≠i} a_{ij} x_j)
      donde Σ usa x_j ya actualizados para j<i y antiguos para j≥i.
    """
    if not (0 < omega < 2):
        raise ValueError("omega debe estar en (0,2).")

    A, b, x = _validate_inputs(A, b, x0)
    n = A.shape[0]

    if return_history:
        history = [x.copy()]

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            # sigma = Σ_{j≠i} a_ij x_j  (con x_j nuevos si j<i, antiguos si j>i)
            sigma = A[i, :].dot(x) - A[i, i] * x[i]
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)

        if _stopped(x, x_old, tol):
            if return_history:
                history.append(x.copy())
                return x, k, history
            return x, k
        if return_history:
            history.append(x.copy())

    if return_history:
        return x, max_iter, history
    return x, max_iter


# ==========================
# Ejemplo rápido (diagonalmente dominante)
# ==========================
if __name__ == "__main__":
    # Construimos una A diagonalmente dominante para garantizar buena convergencia.
    rng = np.random.default_rng(0)
    n = 5
    A = rng.uniform(-1, 1, size=(n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0  # fuerza dominancia diagonal
    b = rng.uniform(-3, 3, size=n)
    x0 = np.zeros(n)

    print("Dominancia diagonal:", is_diag_dominant(A))
    x_j, it_j = jacobi(A, b, x0=x0, tol=1e-10, max_iter=2000)
    x_gs, it_gs = gauss_seidel(A, b, x0=x0, tol=1e-10, max_iter=2000)
    x_sor, it_sor = sor(A, b, omega=1.2, x0=x0, tol=1e-10, max_iter=2000)

    print(f"Jacobi:        iter={it_j},   ||Ax-b||={np.linalg.norm(A@x_j - b):.3e}")
    print(f"Gauss-Seidel:  iter={it_gs},  ||Ax-b||={np.linalg.norm(A@x_gs - b):.3e}")
    print(f"SOR (ω=1.2):   iter={it_sor}, ||Ax-b||={np.linalg.norm(A@x_sor - b):.3e}")
