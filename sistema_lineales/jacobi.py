import copy
import matplotlib.pyplot as plt
from utils import LiveGridPlotter, _stopped, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, is_square, _validate_inputs, create_initial_matrixis
import numpy as np
import time

# ==========================
# Jacobi (n x n)
# ==========================
def jacobi(A, b, x0, tolerance, max_iter=500, on_iter=None):
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

    t0 = time.perf_counter()
    elapsed = 0
    residual_norm = None

    for k in range(1, max_iter + 1):
        # x^{(k+1)} = D^{-1} (b - (L+U) x^{(k)})
        x_new = D_inv @ (b - R @ x)

        less_than_tolerance, residual_norm = _stopped(A, b, x_new, tolerance)

        elapsed = time.perf_counter() - t0

        # Callback de usuario (por ejemplo, para actualizar gráfico)
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

    # Ejecutar Jacobi llamando al plotter en cada iteración
    x_j, it_j, residual_norm, elapsed = jacobi(
        A=jacobian_numeric,
        b=rhs,
        x0=vector_x_initial,
        tolerance=tolerance,
        max_iter=2000,
        on_iter=plotter.update     # <- aquí la magia
    )

    print(f"Jacobi: iter={it_j}, || b - A * x_current ||_2={residual_norm}, time={elapsed:.2f}s")

    # Actualización final (por si quieres asegurar último frame) y mostrar bloqueante
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()