import copy
import matplotlib.pyplot as plt
from utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time

def analyze_matrix_for_conjugate_gradient(A):
    """
    Analiza si **Conjugate Gradient (CG)** es apropiado para Ax=b y estima su
    velocidad teórica. CG **requiere A simétrica definida positiva (SPD)**.

    Reporta:
      - Simetría (A≈Aᵀ) y SPD (vía Cholesky).
      - Si es SPD: κ(A)=λ_max/λ_min y el factor de contracción por iteración:
            q_CG = ((√κ − 1)/(√κ + 1))^2
        (cota clásica del error en norma-A).
    """
    import numpy as np

    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("Conjugate Gradient requiere A cuadrada.")

    # 1) Propiedades básicas
    is_symmetric = np.allclose(A, A.T, atol=1e-12)

    # 2) SPD por Cholesky (criterio práctico)
    try:
        np.linalg.cholesky(A)
        is_spd = True
    except np.linalg.LinAlgError:
        is_spd = False

    print("---- Análisis para Conjugate Gradient ----")
    print(f"Simétrica (A≈Aᵀ): {is_symmetric}")
    print(f"Definida positiva (Cholesky en A): {is_spd}")

    if not is_symmetric or not is_spd:
        # Intento orientativo con la parte simétrica (solo informativo)
        S = 0.5 * (A + A.T)
        try:
            np.linalg.cholesky(S)
            is_S_spd = True
        except np.linalg.LinAlgError:
            is_S_spd = False

        if is_symmetric and not is_spd:
            print("❌ A es simétrica pero no SPD → CG no está garantizado (puede fallar).")
        elif not is_symmetric:
            print("❌ A no es simétrica → CG estándar no aplica.")
        if is_S_spd:
            eigs_S = np.linalg.eigvalsh(S)
            lam_min_S, lam_max_S = float(eigs_S[0]), float(eigs_S[-1])
            kappa_S = lam_max_S / lam_min_S if lam_min_S > 0 else np.inf
            q_cg_S = ((np.sqrt(kappa_S) - 1.0) / (np.sqrt(kappa_S) + 1.0)) ** 2 if np.isfinite(kappa_S) else np.nan
            print("ℹ️ La parte simétrica S=(A+Aᵀ)/2 es SPD (métricas orientativas):")
            print(f"   κ(S)={kappa_S:.6e},  q_CG(S)={q_cg_S:.6e}")
        print("Conclusión: No recomendable usar CG directo. Considere simetrizar/preacondicionar.")
        return

    # 3) Métricas espectrales (A es SPD → usar eigvalsh)
    eigs = np.linalg.eigvalsh(A)
    lam_min, lam_max = float(eigs[0]), float(eigs[-1])
    kappa = lam_max / lam_min
    q_cg = ((np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)) ** 2

    print(f"λ_min(A)={lam_min:.6e},  λ_max(A)={lam_max:.6e}")
    print(f"κ(A) = λ_max/λ_min = {kappa:.6e}")
    print(f"Factor teórico por iteración (cota clásica): q_CG = {q_cg:.6e}")
    print("Conclusión: ✅ A es SPD → CG debe converger; la rapidez mejora cuanto menor sea κ(A).")



# ==========================
# Gradiente descendiente (n x n)
# ==========================
def conjugate_gradient(A, b, x0, tolerance, max_iter=500, on_iter=None):
    r"""
    Gradiente Conjugado (CG) para resolver Ax = b con A **simétrica definida positiva (SPD)**.
    Se muestran explícitamente las transpuestas usando vectores columna (n,1) para que
    cada ecuación sea rastreable en el código.

    Ecuaciones del método (k = 0,1,2,...):
      r_0 = b − A x_0
      p_0 = r_0
      α_k = (r_kᵀ r_k) / (p_kᵀ A p_k)
      x_{k+1} = x_k + α_k p_k
      r_{k+1} = r_k − α_k A p_k
      β_{k+1} = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k)
      p_{k+1} = r_{k+1} + β_{k+1} p_k

    Retorna: (x_final, iteraciones_realizadas, residual_norm, elapsed_segundos)
    """
    # --- Validación/normalización de entradas ---
    A, b, x_k = _validate_inputs(A, b, x0)

    start_time = time.perf_counter()
    residual_norm = None

    # --- r_0 = b − A x_0  (vector columna para usar transpuestas explícitas) ---
    r_k = b - (A @ x_k)
    r_k_col = r_k.reshape(-1, 1)         # (n,1)
    x_k_col = x_k.reshape(-1, 1)         # (n,1)  (solo por simetría visual)
    rr_k = float((r_k_col.T @ r_k_col).item())  # r_kᵀ r_k (escalar)

    # Parada temprana si ya cumple tolerancia con x0
    converged, residual_norm = _stopped(A, b, x_k, tolerance)
    if converged:
        return x_k, 0, residual_norm, 0.0

    # --- p_0 = r_0 ---
    p_k_col = r_k_col.copy()

    for k in range(1, max_iter + 1):
        # --- Ap_k = A p_k ---
        Ap_k_col = (A @ p_k_col)

        # --- α_k = (r_kᵀ r_k) / (p_kᵀ A p_k) ---
        denom_pkApk = float((p_k_col.T @ Ap_k_col).item())  # p_kᵀ A p_k (escalar)

        if denom_pkApk <= 0.0:
            # Fallback (muy raro si A es SPD). Intentamos paso de SD: α = (rᵀr)/(rᵀAr).
            Ar_k_col = (A @ r_k_col)
            denom_rAr = float((r_k_col.T @ Ar_k_col).item())
            if denom_rAr > 0.0:
                alpha_k = rr_k / denom_rAr
            else:
                # Último recurso conservador para evitar explotar numéricamente.
                alpha_k = 1e-3
        else:
            alpha_k = rr_k / denom_pkApk

        # --- x_{k+1} = x_k + α_k p_k ---
        x_next_col = x_k_col + alpha_k * p_k_col

        # --- r_{k+1} = r_k − α_k A p_k ---
        r_next_col = r_k_col - alpha_k * Ap_k_col

        # Chequeo de parada con el candidato x_{k+1}
        x_next = x_next_col.ravel()
        converged, residual_norm = _stopped(A, b, x_next, tolerance)
        elapsed = time.perf_counter() - start_time

        if on_iter is not None:
            on_iter(k, x_next, residual_norm, elapsed)

        if converged:
            return x_next, k, residual_norm, elapsed

        # --- β_{k+1} = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k) ---
        rr_next = float((r_next_col.T @ r_next_col).item())
        if rr_k <= np.finfo(float).eps:
            # r_k prácticamente nulo → solución alcanzada
            return x_next, k, residual_norm, elapsed
        beta_next = rr_next / rr_k

        # --- p_{k+1} = r_{k+1} + β_{k+1} p_k ---
        p_next_col = r_next_col + beta_next * p_k_col

        # Avanzar iteración
        x_k_col = x_next_col
        r_k_col = r_next_col
        p_k_col = p_next_col
        rr_k = rr_next

    # Máximo de iteraciones alcanzado
    return x_k_col.ravel(), max_iter, residual_norm, (time.perf_counter() - start_time)








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

    analyze_matrix_for_conjugate_gradient(jacobian_numeric)

    x_j, it_j, residual_norm, elapsed = conjugate_gradient(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        max_iter=12000,
        on_iter=plotter.update  # <- visualización en vivo
    )

    print(
        f"Gradiente conjugado: "
        f"iter={it_j}, ||b - A·x||_2={residual_norm}, time={elapsed:.2f}s"
    )

    # Actualización final, mostrar bloqueante en el último frame
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()