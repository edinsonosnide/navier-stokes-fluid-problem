import copy
import matplotlib.pyplot as plt
from .utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time

import numpy as np

def analyze_matrix_for_sor(A):
    """
    Analiza A para el método SOR (SIN sugerir valores de ω) y emite conclusiones.
    Reporta: simetría, SPD (Cholesky), ceros en la diagonal, dominancia diagonal por filas,
    y estabilidad de la iteración base de Gauss–Seidel (ω=1) vía la matriz G_GS = −(D+L)^{-1}U.
    """
    import numpy as np

    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("SOR requiere A cuadrada.")

    # 1) Propiedades estructurales
    is_symmetric = np.allclose(A, A.T, atol=1e-10)
    diag = np.diag(A)
    has_zero_on_diag = np.any(np.isclose(diag, 0.0))

    try:
        np.linalg.cholesky(A)
        is_spd = True
    except np.linalg.LinAlgError:
        is_spd = False

    absA = np.abs(A)
    abs_diag = np.abs(diag)
    row_off = np.sum(absA, axis=1) - abs_diag
    dd_strict = np.all(abs_diag > row_off)
    dd_weak_one_strict = np.all(abs_diag >= row_off) and np.any(abs_diag > row_off)

    print("---- Análisis de la matriz A para SOR ----")
    print(f"Simétrica: {is_symmetric}")
    print(f"Definida positiva (Cholesky): {is_spd}")
    print(f"Ceros en la diagonal: {has_zero_on_diag}")
    print(f"Dominancia diagonal por filas (estricta): {dd_strict}")
    print(f"Dominancia diagonal por filas (débil con ≥1 estricta): {dd_weak_one_strict}")

    # 2) Indicadores vía Gauss–Seidel (ω=1) como referencia de estabilidad
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=+1)

    G_inf = np.nan
    G_one = np.nan
    rho_G = np.nan
    gs_ok = False
    try:
        M = D + L  # (D+L)
        G_GS = -np.linalg.solve(M, U)  # evita inversa explícita
        G_inf = float(np.max(np.sum(np.abs(G_GS), axis=1)))
        G_one = float(np.max(np.sum(np.abs(G_GS), axis=0)))
        if n <= 1200:
            rho_G = float(np.max(np.abs(np.linalg.eigvals(G_GS))))
        gs_ok = ((G_inf < 1.0) or (G_one < 1.0) or (np.isfinite(rho_G) and rho_G < 1.0))
    except np.linalg.LinAlgError:
        print("⚠️ No se pudo formar G_GS: (D+L) es singular o mal condicionado.")

    print(f"||G_GS||_∞: {G_inf:.6e}" if np.isfinite(G_inf) else "||G_GS||_∞: no disponible")
    print(f"||G_GS||_1: {G_one:.6e}" if np.isfinite(G_one) else "||G_GS||_1: no disponible")
    print(f"ρ(G_GS): {rho_G:.6e}" if np.isfinite(rho_G) else "ρ(G_GS): no disponible")

    # 3) Conclusiones (sin recomendar ω)
    print("---- Conclusiones ----")
    if has_zero_on_diag:
        print("❌ Hay ceros en la diagonal: (D+L) puede ser singular; la aplicación directa de SOR/GS es problemática.")
    if is_spd:
        print("✅ A es SPD: SOR es teóricamente convergente para algún rango estándar de relajación (0<ω<2).")
    elif dd_strict or dd_weak_one_strict:
        print("✅ A presenta dominancia diagonal por filas: condiciones suficientes clásicas apoyan la convergencia de GS/SOR.")
    else:
        print("⚠️ A no es SPD ni claramente dominante: la convergencia de SOR depende fuertemente de la estructura de A.")

    if np.isfinite(G_inf) or np.isfinite(G_one) or np.isfinite(rho_G):
        if gs_ok:
            print("✅ La iteración base Gauss–Seidel (ω=1) se ve estable por normas/ρ(G_GS), lo cual es un buen indicio para SOR.")
        else:
            print("⚠️ La referencia Gauss–Seidel no mostró estabilidad clara (normas ≥1 y/o ρ(G_GS)≥1): SOR podría ser delicado.")


# ==========================
# SOR (n x n)
# ==========================
def sor(A, b, x0, tolerance, relaxation_parameter_omega=1.0, max_iter=500, on_iter=None):
    r"""
    SOR (Successive Over-Relaxation) usando la **ecuación matricial**:

        x^{(k)} = (D + ω L)^{-1} [ ω b − ( ω U + (ω − 1) D ) x^{(k−1)} ]

    donde A = D + L + U:
      - D: diagonal(A)
      - L: parte estrictamente inferior de A
      - U: parte estrictamente superior de A
      - ω: parámetro de relajación (ω=1 → Gauss–Seidel).

    Notas:
      - Implementación deliberadamente directa (no optimizada): se forma (D + ωL)^{-1} explícitamente,
        porque pediste usar la ecuación matricial tal cual.
      - Firma y retornos compatibles con tu `richardson(...)` para que puedas sustituirla sin tocar el resto:
          (x_final, iteraciones, residual_norm, elapsed_seconds).
    """
    # Normaliza/valida entradas y obtiene x inicial como en tus otros métodos
    A, b, x_current = _validate_inputs(A, b, x0)
    ω = float(relaxation_parameter_omega)

    # Descomposición A = D + L + U
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=+1)

    # (D + ωL)^{-1}
    D_plus_omega_L = D + ω * L
    try:
        D_plus_omega_L_inv = np.linalg.inv(D_plus_omega_L)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "SOR: (D + ωL) es singular; intenta reordenar el sistema o revisar ceros en la diagonal."
        ) from e

    start = time.perf_counter()
    residual_norm = None

    for k in range(1, max_iter + 1):
        # rhs = ω b − ( ω U + (ω − 1) D ) x^{(k−1)}
        rhs = ω * b - (ω * U + (ω - 1.0) * D) @ x_current

        # x^{(k)} = (D + ωL)^{-1} rhs
        x_next = D_plus_omega_L_inv @ rhs

        # Criterio de parada (reutiliza tu utilitario)
        converged, residual_norm = _stopped(A, b, x_next, tolerance)
        elapsed = time.perf_counter() - start

        # Callback opcional (p. ej., para LiveGridPlotter)
        if on_iter is not None:
            on_iter(k, x_next, residual_norm, elapsed)

        if converged:
            return x_next, k, residual_norm, elapsed

        x_current = x_next

    return x_current, max_iter, residual_norm, (time.perf_counter() - start)






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

    # Parámetro de relajación del método de sor.
    # Ajusta este valor si ves que converge muy lento (sube un poco) o diverge/oscila (bájalo).
    relaxation_parameter_omega = 0.8

    print(f'Relaxation parameter omega: {relaxation_parameter_omega}')

    analyze_matrix_for_sor(jacobian_numeric)

    x_j, it_j, residual_norm, elapsed = sor(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        relaxation_parameter_omega=relaxation_parameter_omega,
        max_iter=2000,
        on_iter=plotter.update  # <- visualización en vivo
    )

    print(
        f"SOR (ω={relaxation_parameter_omega}): "
        f"iter={it_j}, ||b - A·x||_2={residual_norm}, time={elapsed:.2f}s"
    )

    # Actualización final, mostrar bloqueante en el último frame
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()