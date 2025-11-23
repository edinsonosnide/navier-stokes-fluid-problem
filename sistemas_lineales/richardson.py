import copy
import matplotlib.pyplot as plt
from .utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time

import numpy as np

def analyze_matrix_for_richardson(A):
    A = np.array(A, dtype=float)
    
    # 1) ¿Es simétrica?
    is_symmetric = np.allclose(A, A.T, atol=1e-10)
    
    # 2) ¿Es definida positiva?
    try:
        np.linalg.cholesky(A)
        is_positive_definite = True
    except np.linalg.LinAlgError:
        is_positive_definite = False
    
    # 3) Autovalores
    eigenvalues = np.linalg.eigvals(A)
    
    print("---- Análisis de la matriz A ----")
    print(f"Simétrica: {is_symmetric}")
    print(f"Definida positiva: {is_positive_definite}")
    print(f"Autovalores reales (numero autovalores: {len(eigenvalues)}): {np.all(np.isreal(eigenvalues))}")
    #print(f"Autovalores ({len(eigenvalues)}): {eigenvalues}")
    
    if is_positive_definite:
        lambda_min = np.min(np.real(eigenvalues))
        lambda_max = np.max(np.real(eigenvalues))
        print(f"Rango espectral: λ_min={lambda_min:.3e}, λ_max={lambda_max:.3e}")
        print(f"Rango de ω válido: (0, {2/lambda_max:.3e})")
        print(f"ω óptimo ≈ {2/(lambda_min + lambda_max):.3e}")
    else:
        print("⚠️ La matriz no es SPD: usar ω pequeño (≈1e-3) y observa el comportamiento.")

# ==========================
# Richardson (n x n)
# ==========================
def richardson(A, b, x0, tolerance, relaxation_parameter_omega=0.5, max_iter=500, on_iter=None):
    r"""
    Método de Richardson para resolver Ax = b por iteración de residuo.

    Idea central:
    -------------
    A partir de una aproximación x^{(k)}, calculamos el residuo r^{(k)} = b - A x^{(k)}.
    Luego damos un paso en la dirección del residuo:
        x^{(k+1)} = x^{(k)} + ω * r^{(k)}
    donde ω > 0 es el parámetro de relajación (tamaño de paso).

    Notas prácticas sobre ω:
    ------------------------
    - Si ω es muy grande, puede divergir; si es muy pequeño, convergerá pero muy lento.
    - Para matrices SPD, una guía clásica es ω ≈ 2 / (λ_min + λ_max).
      Si no se conocen los autovalores, empieza con un valor pequeño (p. ej., 1e-3 a 1e-2)
      y ajústalo empíricamente.
    - Este reemplazo mantiene la misma interfaz general (A, b, x0, tolerance, max_iter, on_iter),
      añadiendo el parámetro 'relaxation_parameter_omega' para que puedas ajustarlo sin
      tocar el núcleo del algoritmo.

    Criterio de parada:
    -------------------
    - Se reutiliza _stopped(A, b, x_candidato, tolerance) para evaluar ||b - A x||_2 <= tolerance
      (o el criterio equivalente que tengas implementado en utils._stopped).

    Parámetros:
    -----------
    A : np.ndarray
        Matriz del sistema.
    b : np.ndarray
        Vector del lado derecho.
    x0 : np.ndarray
        Aproximación inicial.
    tolerance : float
        Tolerancia para detener la iteración (en norma del residuo).
    relaxation_parameter_omega : float, opcional
        Tamaño de paso ω del método de Richardson.
    max_iter : int, opcional
        Número máximo de iteraciones.
    on_iter : callable, opcional
        Callback con firma on_iter(k, x_actual, residual_norm, elapsed_segundos).

    Retorna:
    --------
    x_aproximado, iteraciones_realizadas, residual_norm, elapsed_segundos
    """

    # Normaliza y valida entradas (reutiliza tu utilitario actual).
    A, b, x_iterate_current = _validate_inputs(A, b, x0)

    start_time_perf_counter_seconds = time.perf_counter()
    elapsed_seconds_since_start = 0.0
    residual_norm_current_iteration = None

    for iteration_index in range(1, max_iter + 1):
        # 1) Calcula el residuo actual r^{(k)} = b - A x^{(k)}.
        residual_vector_current = b - (A @ x_iterate_current)

        # 2) Paso de Richardson: x^{(k+1)} = x^{(k)} + ω * r^{(k)}.
        x_iterate_next = x_iterate_current + (relaxation_parameter_omega * residual_vector_current)

        # 3) Evalúa criterio de parada con el candidato x^{(k+1)}.
        less_than_tolerance, residual_norm_current_iteration = _stopped(
            A, b, x_iterate_next, tolerance
        )

        # 4) Tiempo transcurrido.
        elapsed_seconds_since_start = time.perf_counter() - start_time_perf_counter_seconds

        # 5) Callback opcional (útil para visualización en vivo).
        if on_iter is not None:
            on_iter(iteration_index, x_iterate_next, residual_norm_current_iteration, elapsed_seconds_since_start)

        # 6) Revisa convergencia.
        if less_than_tolerance:
            return x_iterate_next, iteration_index, residual_norm_current_iteration, elapsed_seconds_since_start

        # 7) Avanza a la siguiente iteración.
        x_iterate_current = x_iterate_next

    # Si se alcanzó el máximo de iteraciones, devuelve el último estado.
    return x_iterate_current, max_iter, residual_norm_current_iteration, elapsed_seconds_since_start






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

    # Parámetro de relajación del método de Richardson.
    # Ajusta este valor si ves que converge muy lento (sube un poco) o diverge/oscila (bájalo).
    relaxation_parameter_omega = 0.5

    print(f'Relaxation parameter omega: {relaxation_parameter_omega}')

    analyze_matrix_for_richardson(jacobian_numeric)

    x_j, it_j, residual_norm, elapsed = richardson(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        relaxation_parameter_omega=relaxation_parameter_omega,
        max_iter=2000,
        on_iter=plotter.update  # <- visualización en vivo
    )

    print(
        f"Richardson (ω={relaxation_parameter_omega}): "
        f"iter={it_j}, ||b - A·x||_2={residual_norm}, time={elapsed:.2f}s"
    )

    # Actualización final, mostrar bloqueante en el último frame
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()