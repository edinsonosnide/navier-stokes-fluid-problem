import copy
import matplotlib.pyplot as plt
from .utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time


import numpy as np

def analyze_matrix_for_gauss_seidel(A):
    """
    Análisis compacto para Gauss–Seidel.
    Convergencia ⇔ ρ(G_GS) < 1, donde
        G_GS = -(D+L)^{-1} U
    y A = D + L + U (D: diagonal, L: estrictamente inferior, U: estrictamente superior).

    También son condiciones suficientes:
      • A simétrica definida positiva (SPD) → converge.
      • A diagonalmente dominante por filas (estricta, o débil con ≥1 estricta) → converge.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Gauss–Seidel requiere A cuadrada.")

    # 1) Chequeos rápidos
    is_symmetric = np.allclose(A, A.T, atol=1e-10)
    diag = np.diag(A)
    has_zero_on_diagonal = np.any(np.isclose(diag, 0.0))

    # SPD por Cholesky (garantía fuerte)
    try:
        np.linalg.cholesky(A)
        is_spd = True
    except np.linalg.LinAlgError:
        is_spd = False

    # 2) Dominancia diagonal por filas (suficiente)
    absA = np.abs(A)
    abs_diag = np.abs(diag)
    row_offdiag_sum = np.sum(absA, axis=1) - abs_diag
    dd_strict = np.all(abs_diag > row_offdiag_sum)
    dd_weak_one_strict = np.all(abs_diag >= row_offdiag_sum) and np.any(abs_diag > row_offdiag_sum)

    print("---- Análisis para Gauss–Seidel ----")
    print(f"Simétrica: {is_symmetric}")
    print(f"Definida positiva (Cholesky): {is_spd}")
    print(f"Ceros en la diagonal: {has_zero_on_diagonal}")
    print(f"Dominancia diagonal por filas (estricta): {dd_strict}")
    print(f"Dominancia diagonal por filas (débil con ≥1 estricta): {dd_weak_one_strict}")

    if has_zero_on_diagonal:
        print("❌ Gauss–Seidel problemático: (D+L) puede ser singular (a_{ii}=0).")
        return

    # 3) Matriz de iteración de GS: G_GS = -(D+L)^{-1} U
    D_plus_L = np.tril(A)        # D + L
    U = A - D_plus_L             # U
    try:
        # Usamos solve para no invertir explícitamente (más estable, mismo resultado analítico)
        # Equivalente a: G = -inv(D+L) @ U
        G = -np.linalg.solve(D_plus_L, U)
    except np.linalg.LinAlgError:
        print("⚠️ No se pudo factorizar (D+L). Revisa singularidad o reordenamiento de A.")
        return

    # 4) Indicadores: normas (cotas) y radio espectral exacto (si es computable)
    G_inf = np.max(np.sum(np.abs(G), axis=1))
    G_one = np.max(np.sum(np.abs(G), axis=0))
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(G))))
    except np.linalg.LinAlgError:
        spectral_radius = np.nan

    print(f"||G||_∞: {G_inf:.6e}")
    print(f"||G||_1: {G_one:.6e}")
    print(f"ρ(G) (exacto si eigs disponibles): {spectral_radius:.6e}")

    # 5) Veredicto simple
    if is_spd:
        print("✅ GS: A es SPD → Convergencia garantizada.")
    elif np.isfinite(spectral_radius) and spectral_radius < 1.0:
        print("✅ GS: criterio espectral cumple (ρ(G) < 1) → Convergencia esperada.")
    elif (G_inf < 1.0) or (G_one < 1.0) or dd_strict or dd_weak_one_strict:
        print("✅ GS: se cumple al menos una condición suficiente/cota (convergencia probable).")
    else:
        print("⚠️ GS: no hay garantía clara de convergencia (considera SOR o reordenar/escalar A).")



# ==========================
# Gauss-Seidel (n x n)
# ==========================
def gauss_seidel(A, b, x0, tolerance, max_iter=500, on_iter=None):
    r"""
    Gauss–Seidel en forma matricial:
        x^{(k)} = (D + L)^{-1} ( b - U x^{(k-1)} )
    con A = D + L + U.
    """
    A, b, x = _validate_inputs(A, b, x0)

    D_plus_L = np.tril(A)        # D + L (triangular inferior con diagonal)
    U = A - D_plus_L             # U (estrictamente superior)

    try:
        D_plus_L_inv = np.linalg.inv(D_plus_L)  # inversion explícita
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Gauss–Seidel: (D+L) es singular; no se puede invertir."
        ) from e

    t0 = time.perf_counter()
    elapsed = 0
    residual_norm = None

    for k in range(1, max_iter + 1):
        rhs = b - (U @ x)                    # b - U x^{(k-1)}
        x_new = D_plus_L_inv @ rhs           # (D+L)^{-1} rhs

        less_than_tolerance, residual_norm = _stopped(A, b, x_new, tolerance)
        elapsed = time.perf_counter() - t0

        if on_iter is not None:
            on_iter(k, x_new, residual_norm, elapsed)

        if less_than_tolerance:
            return x_new, k, residual_norm, elapsed

        x = x_new

    return x, max_iter, residual_norm, elapsed





if __name__ == "__main__":
    matrix_height = 8
    matrix_width = 80

    fluid_inlet_velocity = 1
    fluid_cells_initial_velocity = 0

    vorticity_or_v_i_j_y = 4

    h = 1 # number for wich the widht and hight is divided to get to matrix_width and matrix_height

    tolerance = 1e-5

    matrix_grid_coordinates, matrix_grid_naming, matrix_grid_initial_values = create_initial_matrixis(matrix_height, matrix_width, fluid_cells_initial_velocity, fluid_inlet_velocity)

    print("\n")
    for i in matrix_grid_coordinates:
        print(i)
    print("\n")
    
    print('Matrix cell clases')
    print('Matrix cell clases dim: ',f'{len(matrix_grid_naming),len(matrix_grid_naming[0])}')
    for i in matrix_grid_naming:
        print(i)
    print("\n")

    print('Initial matrix numeric values')
    print('Initial matrix numeric values dim: 'f'{len(matrix_grid_initial_values),len(matrix_grid_initial_values[0])}')
    for i in matrix_grid_initial_values:
        print(i)
    print("\n")

    vector_f_numeric = np.array(calculate_f_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y, h), dtype=float)

    print('Initial vector f')
    print('Initial vector f dim: ', len(vector_f_numeric))
    print(vector_f_numeric)
    print("\n")

    jacobian_numeric = np.array(calculate_jacobian_numeric_for_this_iteration(matrix_height,matrix_width, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y, h), dtype=float)

    print('Initial jacobian numeric')
    print('Initial jacobian numeric dim: ', f'{len(jacobian_numeric),len(jacobian_numeric[0])}')
    #for i in jacobian_numeric:
    #    print(i)
    print("\n")

    vector_x_initial = np.array([fluid_cells_initial_velocity for _ in range(0,len(vector_f_numeric))], dtype=float)
    print('Vector x initial')
    print('Vector x initial dim: ', len(vector_x_initial))
    print(vector_x_initial)
    print("\n")

    rhs = -vector_f_numeric

    # Crear el plotter vivo
    plotter = LiveGridPlotter(
        matrix_grid_naming=matrix_grid_naming,
        base_grid_2d=matrix_grid_initial_values,
        title="Matrix values"
    )

    analyze_matrix_for_gauss_seidel(jacobian_numeric)

    # Ejecutar Jacobi llamando al plotter en cada iteración
    x_j, it_j, residual_norm, elapsed = gauss_seidel(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        max_iter=2000,
        on_iter=plotter.update     # <- aquí la magia
    )

    print(f"gauss_seidel: iter={it_j}, || b - A * x_current ||_2={residual_norm}, time={elapsed:.2f}s")

    # Actualización final y mostrar bloqueante en último frame
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()