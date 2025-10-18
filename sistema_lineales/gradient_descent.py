import copy
import matplotlib.pyplot as plt
from utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time

def analyze_matrix_for_gradient_descent(A):
    """
    Analiza A para **Gradiente Descendente (Steepest Descent)** aplicado a Ax=b con
    paso óptimo α_k = (r_kᵀ r_k)/(r_kᵀ A r_k). Este esquema requiere A **simétrica definida
    positiva (SPD)** para garantizar que r_kᵀ A r_k > 0 y convergencia.

    Reporta: simetría, SPD (Cholesky), dominancia diagonal por filas, y—si A es SPD—
    espectro, número de condición κ₂(A) y razón teórica de contracción en norma-A:
        q_SD = ((κ − 1)/(κ + 1))².
    """
    import numpy as np

    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Gradiente descendente requiere A cuadrada.")

    # 1) Propiedades estructurales
    is_symmetric = np.allclose(A, A.T, atol=1e-12)
    diag = np.diag(A)
    has_zero_on_diag = np.any(np.isclose(diag, 0.0))

    # Dominancia diagonal (condición suficiente útil en práctica, no necesaria)
    absA = np.abs(A)
    abs_diag = np.abs(diag)
    row_off = np.sum(absA, axis=1) - abs_diag
    dd_strict = np.all(abs_diag > row_off)
    dd_weak_one_strict = np.all(abs_diag >= row_off) and np.any(abs_diag > row_off)

    # 2) SPD por Cholesky (si no es simétrica, probar parte simétrica S como referencia)
    try:
        np.linalg.cholesky(A)
        is_spd = True
        spd_checked_matrix = A
        used_sym_part = False
    except np.linalg.LinAlgError:
        is_spd = False
        S = 0.5 * (A + A.T)  # parte simétrica
        try:
            np.linalg.cholesky(S)
            spd_checked_matrix = S
            used_sym_part = True
        except np.linalg.LinAlgError:
            spd_checked_matrix = None
            used_sym_part = False

    print("---- Análisis de la matriz A para Gradiente Descendente ----")
    print(f"Simétrica (A≈Aᵀ): {is_symmetric}")
    print(f"Ceros en la diagonal: {has_zero_on_diag}")
    print(f"Dominancia diagonal (estricta por filas): {dd_strict}")
    print(f"Dominancia diagonal (débil con ≥1 estricta): {dd_weak_one_strict}")

    if is_spd:
        print("✅ A es SPD (Cholesky en A): el paso óptimo está bien definido y hay convergencia.")
    elif spd_checked_matrix is not None:
        print("ℹ️ A no es SPD, pero su parte simétrica S=(A+Aᵀ)/2 es SPD.")
        print("   Ojo: GD con α_k=(rᵀr)/(rᵀAr) garantiza descenso/teoría usando A SPD;")
        print("   si usas A no simétrica, rᵀAr podría no ser >0 en todas las iteraciones.")
    else:
        print("❌ Ni A ni su parte simétrica S son SPD: GD con paso óptimo puede fallar (rᵀAr ≤ 0).")

    # 3) Espectro y condición si (algo) SPD disponible
    if spd_checked_matrix is not None:
        # Como spd_checked_matrix es simétrica, usar eigvalsh (más estable/rápido)
        eigs = np.linalg.eigvalsh(spd_checked_matrix)
        lam_min = float(np.min(eigs))
        lam_max = float(np.max(eigs))
        print(f"λ_min (S) = {lam_min:.6e},  λ_max (S) = {lam_max:.6e}")

        if lam_min > 0.0:
            kappa2 = lam_max / lam_min
            q_sd = ((kappa2 - 1.0) / (kappa2 + 1.0)) ** 2  # razón teórica de contracción en norma-A
            print(f"κ₂(S) = {kappa2:.6e}")
            print(f"Razón teórica de contracción (norma-A) para Steepest Descent: q_SD = {q_sd:.6e}")
            if used_sym_part and not is_spd:
                print("⚠️ Estas métricas usan S (no exactamente A). Úsalas como guía cualitativa.")
        else:
            print("⚠️ λ_min ≤ 0 en la matriz analizada: no hay SPD efectivo → GD no garantizado.")

    # 4) Conclusiones
    print("---- Conclusiones ----")
    if is_spd:
        print("✔️ Adecuado para GD con paso exacto: esperable convergencia; la rapidez depende de κ₂(A).")
    elif spd_checked_matrix is not None:
        print("⚠️ GD sobre A no simétrica: teoría estándar usa SPD; monitorea rᵀAr y residuo en práctica.")
    else:
        print("✖️ No apto para GD con paso α_k=(rᵀr)/(rᵀAr): considera preacondicionar/simetrizar o usar otro método.")


# ==========================
# Gradiente descendiente (n x n)
# ==========================
def gradient_descent(A, b, x0, tolerance, max_iter=500, on_iter=None):
    r"""
    Gradiente Descendente / Steepest Descent para Ax = b minimizando ϕ(x)=½ xᵀAx − bᵀx.

    Fórmulas (se aplican literalmente abajo con vectores columna):
      (1) r_k = b − A x_k
      (2) α_k = (r_kᵀ r_k) / (r_kᵀ A r_k)          ← usamos r_col.T @ r_col y r_col.T @ A @ r_col
      (3) x_{k+1} = x_k + α_k r_k
    """
    # Validación/normalización de entradas (misma utilidad que en tus otros métodos)
    A, b, current_solution_vector_x = _validate_inputs(A, b, x0)

    start_time_perf_counter_seconds = time.perf_counter()
    residual_norm_current_iteration = None

    # Parada temprana si x0 ya cumple
    has_converged, residual_norm_current_iteration = _stopped(
        A, b, current_solution_vector_x, tolerance
    )
    if has_converged:
        return current_solution_vector_x, 0, residual_norm_current_iteration, 0.0

    for iteration_index in range(1, max_iter + 1):
        # -------------------------------------------------------
        # (1) r_k = b − A x_k   (vector columna explícito para mostrar transpuestas)
        # -------------------------------------------------------
        residual_vector_r = b - (A @ current_solution_vector_x)
        r_col = residual_vector_r.reshape(-1, 1)  # (n,1)
        x_col = current_solution_vector_x.reshape(-1, 1)  # (n,1), sólo para claridad simétrica en (3)

        # -------------------------------------------------------
        # (2) α_k = (r_kᵀ r_k) / (r_kᵀ A r_k)
        #     Aquí *sí* usamos la transpuesta explícita (.T) sobre el (n,1):
        #       r_kᵀ r_k   ≡ (r_col.T @ r_col)[0,0]
        #       r_kᵀ A r_k ≡ (r_col.T @ A @ r_col)[0,0]
        #     (Para problemas con números COMPLEJOS, cambia a np.vdot:
        #       num = np.vdot(r, r); den = np.vdot(r, A@r) ).
        # -------------------------------------------------------
        numerator_rr = float((r_col.T @ r_col).item())
        denominator_rAr = float((r_col.T @ (A @ r_col)).item())

        if denominator_rAr > 0.0:
            gradient_descent_step_size_alpha = numerator_rr / denominator_rAr
            print(f'gradient_descent_step_size_alpha = {gradient_descent_step_size_alpha}')
        else:
            # Fallback conservador si A no es SPD o hay inestabilidad numérica
            gradient_descent_step_size_alpha = 1e-3
            print(f'A no es SPD o hay inestabilidad numérica: gradient_descent_step_size_alpha = {gradient_descent_step_size_alpha}')
            

        # -------------------------------------------------------
        # (3) x_{k+1} = x_k + α_k r_k   (volvemos a forma (n,) para mantener tu interfaz)
        # -------------------------------------------------------
        next_solution_vector_x = (x_col + gradient_descent_step_size_alpha * r_col).ravel()

        # Criterio de parada con el candidato x_{k+1}
        has_converged, residual_norm_current_iteration = _stopped(
            A, b, next_solution_vector_x, tolerance
        )
        elapsed_seconds_since_start = time.perf_counter() - start_time_perf_counter_seconds

        if on_iter is not None:
            on_iter(
                iteration_index,
                next_solution_vector_x,
                residual_norm_current_iteration,
                elapsed_seconds_since_start,
            )

        if has_converged:
            return next_solution_vector_x, iteration_index, residual_norm_current_iteration, elapsed_seconds_since_start

        # Avanzar a la siguiente iteración
        current_solution_vector_x = next_solution_vector_x

    # Máximo de iteraciones alcanzado
    return (
        current_solution_vector_x,
        max_iter,
        residual_norm_current_iteration,
        (time.perf_counter() - start_time_perf_counter_seconds),
    )







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

    analyze_matrix_for_gradient_descent(jacobian_numeric)

    x_j, it_j, residual_norm, elapsed = gradient_descent(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        max_iter=2000,
        on_iter=plotter.update  # <- visualización en vivo
    )

    print(
        f"Gradiente descendiente: "
        f"iter={it_j}, ||b - A·x||_2={residual_norm}, time={elapsed:.2f}s"
    )

    # Actualización final, mostrar bloqueante en el último frame
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()