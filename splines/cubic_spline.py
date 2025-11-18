import numpy as np
from scipy.interpolate import CubicSpline
from sistemas_lineales.utils import LiveGridPlotter, calculate_f_for_this_iteration, calculate_jacobian_numeric_for_this_iteration, create_initial_matrixis
from sistemas_lineales.sor import analyze_matrix_for_sor, sor

if __name__ == "__main__":
    matrix_height = 8
    matrix_width = 80

    fluid_inlet_velocity = 1
    fluid_cells_initial_velocity = 0

    vorticity_or_v_i_j_y = 4

    h = 1  # number for wich the widht and hight is divided to get to matrix_width and matrix_height

    tolerance = 1e-5

    matrix_grid_coordinates, matrix_grid_naming, matrix_grid_initial_values = create_initial_matrixis(
        matrix_height, matrix_width, fluid_cells_initial_velocity, fluid_inlet_velocity
    )

    print("\n")
    for i in matrix_grid_coordinates:
        print(i)
    print("\n")

    print('Matrix cell clases')
    print('Matrix cell clases dim: ', f'{len(matrix_grid_naming), len(matrix_grid_naming[0])}')
    for i in matrix_grid_naming:
        print(i)
    print("\n")

    print('Initial matrix numeric values')
    print('Initial matrix numeric values dim: ' f'{len(matrix_grid_initial_values), len(matrix_grid_initial_values[0])}')
    for i in matrix_grid_initial_values:
        print(i)
    print("\n")

    vector_f_numeric = np.array(
        calculate_f_for_this_iteration(
            matrix_height, matrix_width, matrix_grid_naming, matrix_grid_initial_values,
            vorticity_or_v_i_j_y, h
        ),
        dtype=float
    )

    print('Initial vector f')
    print('Initial vector f dim: ', len(vector_f_numeric))
    print(vector_f_numeric)
    print("\n")

    jacobian_numeric = np.array(
        calculate_jacobian_numeric_for_this_iteration(
            matrix_height, matrix_width, matrix_grid_naming, matrix_grid_initial_values,
            vorticity_or_v_i_j_y, h
        ),
        dtype=float
    )

    print('Initial jacobian numeric')
    print('Initial jacobian numeric dim: ', f'{len(jacobian_numeric), len(jacobian_numeric[0])}')
    print("\n")

    vector_x_initial = np.array(
        [fluid_cells_initial_velocity for _ in range(0, len(vector_f_numeric))],
        dtype=float
    )

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
        on_iter=plotter.update  # visualización en vivo
    )

    print(
        f"SOR (ω={relaxation_parameter_omega}): "
        f"iter={it_j}, ||b - A·x||_2={residual_norm}, time={elapsed:.2f}s"
    )

    # ============================================================
    # POST-PROCESADO: refinamiento con splines 2D para visualización
    # ============================================================

    import matplotlib.pyplot as plt

    # ============
    # Fase 0: reconstruir solución coarse 8x80
    # ============
    coarse_solution_matrix_values = np.array(matrix_grid_initial_values, dtype=float)

    fluid_index = 0
    for i in range(matrix_height):
        for j in range(matrix_width):
            if matrix_grid_naming[i][j] == 'FLUID':
                coarse_solution_matrix_values[i, j] = x_j[fluid_index]
                fluid_index += 1

    # ============
    # Fase 1: parámetros de la malla refinada
    # ============
    ref_n = 5

    fine_h = matrix_height * ref_n     # 40
    fine_w = matrix_width * ref_n      # 400

    intermediate_x_refined = np.zeros((matrix_height, fine_w), dtype=float)
    fine_solution_matrix_values = np.zeros((fine_h, fine_w), dtype=float)

    # ============
    # Fase 2: refinamiento horizontal por segmentos FLUID
    # ============
    for i in range(matrix_height):

        segments = []
        inside = False
        start = None

        for j in range(matrix_width):
            if matrix_grid_naming[i][j] == 'FLUID' and not inside:
                inside = True
                start = j
            elif matrix_grid_naming[i][j] != 'FLUID' and inside:
                segments.append((start, j - 1))
                inside = False

        if inside:
            segments.append((start, matrix_width - 1))

        for seg_start, seg_end in segments:
            length = seg_end - seg_start + 1

            coarse_cols = np.arange(seg_start, seg_end + 1, dtype=float)
            coarse_vals = coarse_solution_matrix_values[i, seg_start:seg_end + 1]

            fine_seg_start = seg_start * ref_n
            fine_seg_end = (seg_end + 1) * ref_n - 1

            if length == 1:
                v = coarse_vals[0]
                intermediate_x_refined[i, fine_seg_start: fine_seg_end + 1] = v

            else:
                spline = CubicSpline(coarse_cols, coarse_vals, bc_type='natural')

                total = fine_seg_end - fine_seg_start
                for J in range(fine_seg_start, fine_seg_end + 1):
                    t = (J - fine_seg_start) / total
                    xf = seg_start + t * (seg_end - seg_start)
                    intermediate_x_refined[i, J] = spline(xf)

    # ============
    # Fase 3: refinamiento vertical por segmentos FLUID
    # ============
    for J in range(fine_w):

        j_coarse = J // ref_n

        segments = []
        inside = False
        start = None

        for i in range(matrix_height):
            if matrix_grid_naming[i][j_coarse] == 'FLUID' and not inside:
                inside = True
                start = i
            elif matrix_grid_naming[i][j_coarse] != 'FLUID' and inside:
                segments.append((start, i - 1))
                inside = False

        if inside:
            segments.append((start, matrix_height - 1))

        for seg_start, seg_end in segments:
            length = seg_end - seg_start + 1

            coarse_rows = np.arange(seg_start, seg_end + 1, dtype=float)
            coarse_vals = intermediate_x_refined[seg_start:seg_end + 1, J]

            fine_seg_start = seg_start * ref_n
            fine_seg_end = (seg_end + 1) * ref_n - 1

            if length == 1:
                v = coarse_vals[0]
                fine_solution_matrix_values[fine_seg_start: fine_seg_end + 1, J] = v

            else:
                spline = CubicSpline(coarse_rows, coarse_vals, bc_type='natural')

                total = fine_seg_end - fine_seg_start
                for I in range(fine_seg_start, fine_seg_end + 1):
                    t = (I - fine_seg_start) / total
                    yf = seg_start + t * (seg_end - seg_start)
                    fine_solution_matrix_values[I, J] = spline(yf)

    # ============
    # Fase 4: generar geometría fina 40x400
    # ============
    (
        fine_coords,
        fine_naming,
        fine_initial
    ) = create_initial_matrixis(fine_h, fine_w, fluid_cells_initial_velocity, fluid_inlet_velocity, ref_n)

    final_plot_matrix_fine = np.zeros((fine_h, fine_w), dtype=float)

    for I in range(fine_h):
        for J in range(fine_w):
            cell = fine_naming[I][J]

            if cell == 'FLUID':
                final_plot_matrix_fine[I, J] = fine_solution_matrix_values[I, J]

            elif cell == 'VELO0':
                final_plot_matrix_fine[I, J] = fluid_inlet_velocity

            else:
                final_plot_matrix_fine[I, J] = 0.0

    # ============
    # Fase 5: gráficas (coarse + fine)
    # ============
    print("Tipos en columna 0 de fine_naming:", sorted(set(row[0] for row in fine_naming)))
    print("Columna 0 de final_plot_matrix_fine:", final_plot_matrix_fine[:, 0])
    # coarse 8×80
    plt.figure(figsize=(8, 3))
    plt.imshow(coarse_solution_matrix_values, origin='upper', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Valor de la variable física')
    plt.title('Solución coarse 8×80')
    plt.xlabel('x (coarse)')
    plt.ylabel('y (coarse)')
    plt.tight_layout()

    # fine 40×400
    plt.figure(figsize=(12, 4))
    plt.imshow(final_plot_matrix_fine, origin='upper', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Valor de la variable física')
    plt.title(f'Solución refinada {ref_n*matrix_height}x{ref_n*matrix_width} con splines en segmentos FLUID continuos')
    plt.xlabel('x (fine)')
    plt.ylabel('y (fine)')
    plt.tight_layout()

    plt.figure()
    plt.imshow(final_plot_matrix_fine[:, [0]], origin='upper', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Solo columna 0 de final_plot_matrix_fine")

    # ============
    # Mostrar todas las figuras abiertas
    # ============
    plotter.update(it_j, x_j, residual_norm, elapsed)
    plotter.show_blocking()

