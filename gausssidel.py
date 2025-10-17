import copy
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Utilidades
# ==========================

def is_square(A):
    """Devuelve True si A es cuadrada."""
    A = np.asarray(A)
    return A.ndim == 2 and A.shape[0] == A.shape[1]

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


def create_initial_matrixis():

    matrix_grid_coordinates = []
    matrix_grid_naming = []
    matrix_grid_to_show = []
    matrix_grid_initial_values = []
    
    # Create the matrix with clear coordinates
    for i in range(0,matrix_height):
        matrix_grid_coordinates.append([])
        for j in range(0,matrix_width):
            if (j == 0):
                matrix_grid_coordinates[i] = [(i,0)]
            else:
                matrix_grid_coordinates[i] = matrix_grid_coordinates[i] + [(i,j)]
    
    matrix_grid_naming = copy.deepcopy(matrix_grid_coordinates)
    matrix_grid_to_show = copy.deepcopy(matrix_grid_coordinates)

    # Fill the walls (OWALL), beams (BEAM1), and fluid cells (FLUID)
    for i in range(0,matrix_height):
        for j in range(0,matrix_width):
            if (i == 0 or i == matrix_height - 1 or j == matrix_width - 1): # Outer Walls
                matrix_grid_naming[i][j] = 'OWALL'
                matrix_grid_to_show[i][j] = 0
            elif ((i < matrix_height) and (i > int(matrix_height*0.4))) and ((j <  int(matrix_width*0.55)) and (j > int(matrix_width*0.43))): # Bottom beam
                matrix_grid_naming[i][j] = 'BEAM1'
                matrix_grid_to_show[i][j] = 1
            elif ((i < matrix_height*0.5) and (i > 0)) and ((j <  matrix_width - 1) and (j > int(matrix_width*0.87))): # Top beam
                matrix_grid_naming[i][j] = 'BEAM2'
                matrix_grid_to_show[i][j] = 2
            elif (i > 0 and i < matrix_height - 1 and j == 0):
                matrix_grid_naming[i][j] = 'VELO0'
                matrix_grid_to_show[i][j] = 3
            else:
                matrix_grid_naming[i][j] = 'FLUID'
                matrix_grid_to_show[i][j] = 4

    matrix_grid_initial_values = copy.deepcopy(matrix_grid_coordinates)

    # Create the initial value matrix, replacing with zero walls, beams and fluid, and with fluid_inlet_velocity VELO0
    for i in range(0,matrix_height):
        for j in range(0,matrix_width):
            if matrix_grid_naming[i][j] == 'FLUID':
                matrix_grid_initial_values[i][j] = fluid_cells_initial_velocity
            elif matrix_grid_naming[i][j] == 'VELO0':
                matrix_grid_initial_values[i][j] = fluid_inlet_velocity
            else:
                matrix_grid_initial_values[i][j] = 0
    
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["black", "orange", "orange", "green", "white"]
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors)
    plt.figure(figsize=(10,2))
    image = plt.imshow(matrix_grid_to_show, cmap=custom_cmap, origin='upper',interpolation='nearest')
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('matrix_grid_to_show')
    plt.tight_layout()
    #plt.show() # uncomment to show graphic

    return matrix_grid_coordinates, matrix_grid_naming, matrix_grid_initial_values

# Create the F vector. The F vector is unidemensional, it contains each result of each equation of each cell that is a fluid
# Each element of F is on the form: [1,2,1,2,1,2,...] of dimension 1XNUMER_OF_FLUID_CELLS
def calculate_f_for_this_iteration(matrix_grid_numeric_current_iteration):
    vector_f_numeric = []

    for i in range(0,matrix_height):
        for j in range(0,matrix_width):
            if matrix_grid_naming[i][j] == 'FLUID':
                accumulated = 0

                VALUE_CELL_CENTER = matrix_grid_numeric_current_iteration[i][j]
                VALUE_CELL_UP = matrix_grid_numeric_current_iteration[i - 1][j]
                VALUE_CELL_DOWN = matrix_grid_numeric_current_iteration[i + 1][j]
                VALUE_CELL_RIGHT = matrix_grid_numeric_current_iteration[i][j + 1]
                VALUE_CELL_LEFT = matrix_grid_numeric_current_iteration[i][j - 1]
                
                #for xi of F for this cell
                right_side_of_xi_equation = (1/4)*(VALUE_CELL_UP + VALUE_CELL_DOWN + VALUE_CELL_LEFT + VALUE_CELL_RIGHT - (h/2)*VALUE_CELL_CENTER*(VALUE_CELL_UP - VALUE_CELL_DOWN) - (h/2)*vorticity_or_v_i_j_y*(VALUE_CELL_RIGHT - VALUE_CELL_LEFT))
                left_side_of_xi_equation = VALUE_CELL_CENTER

                # como la funcion es: lado_derechoi,j​=1/4​(U+D+L+R−(h/2)​(U−D)−(h/2)​(vi,jy)​(R−L))
                # y para newton raphson hay que igual a 0 cada funcion, entonces cada xi de F queda =
                # 0 = lado_izquierdoi,j​ - 1/4​(U+D+L+R−(h/2)​(U−D)−(h/2)​(vi,jy)​(R−L))
                # o lo que es lo mismo pero por alguna razon no queda bien:
                # 0 = 1/4​(U+D+L+R−(h/2)​(U−D)−(h/2)​(vi,jy)​(R−L)) - lado_izquierdoi,j​
                xi = left_side_of_xi_equation - right_side_of_xi_equation

                vector_f_numeric.append(xi)

    return vector_f_numeric

def calculate_jacobian_numeric_for_this_iteration(matrix_grid_numeric_current_iteration):

    # Second I define a dict that maps each FLUID cell to a number, that number is the position of that cell in F for each row
    def fluid_cells_cords_k_and_i_j(matrix_height, matrix_width, matrix_grid_naming):
        k_cords = {}
        i_j_cords = {}
        counter = 0
        for i in range(0,matrix_height):
            for j in range(0,matrix_width):
                if matrix_grid_naming[i][j] == 'FLUID':
                    k_cords[(i,j)] = counter
                    i_j_cords[counter] = (i,j)
                    counter += 1
        return k_cords, i_j_cords

    k_cords, i_j_cords = fluid_cells_cords_k_and_i_j(matrix_height, matrix_width, matrix_grid_naming)

    '''
    print('k_cords: ')
    print(k_cords)
    print("\n")
    print('i_j_cords: ')
    print(i_j_cords)
    print("\n")
    
    '''

    jacobian_width = len(k_cords.keys())
    jacobian_heigth = len(k_cords.keys())
    jacobian_matrix = []

    # i is the cell
    # j is the variable
    # I need to convert somehow i to ij and j to ij
    # First I initialize the jacobian matrix to cero
    for i in range(0, jacobian_heigth):
        jacobian_matrix.append([])
        for j in range(0, jacobian_width):
            if (j == 0):
                jacobian_matrix[i] = [0]
            else:
                jacobian_matrix[i] = jacobian_matrix[i] + [0]

    for i in range (0,len(k_cords.keys())):
        UP_NEIGHBOUR_CLASS_CELL = matrix_grid_naming[i_j_cords[i][0] - 1][i_j_cords[i][1]]
        DOWN_NEIGHBOUR_CLASS_CELL = matrix_grid_naming[i_j_cords[i][0] + 1][i_j_cords[i][1]]
        LEFT_NEIGHBOUR_CLASS_CELL = matrix_grid_naming[i_j_cords[i][0]][i_j_cords[i][1] - 1]
        RIGHT_NEIGHBOUR_CLASS_CELL = matrix_grid_naming[i_j_cords[i][0]][i_j_cords[i][1] + 1]

        CURRENT_NEIGHBOUR_NUMERIC_VALUE_CELL = matrix_grid_numeric_current_iteration[i_j_cords[i][0]][i_j_cords[i][1]]
        UP_NEIGHBOUR_NUMERIC_VALUE_CELL = matrix_grid_numeric_current_iteration[i_j_cords[i][0] - 1][i_j_cords[i][1]]
        DOWN_NEIGHBOUR_NUMERIC_VALUE_CELL = matrix_grid_numeric_current_iteration[i_j_cords[i][0] + 1][i_j_cords[i][1]]
        LEFT_NEIGHBOUR_NUMERIC_VALUE_CELL = matrix_grid_numeric_current_iteration[i_j_cords[i][0]][i_j_cords[i][1] - 1]
        RIGHT_NEIGHBOUR_NUMERIC_VALUE_CELL = matrix_grid_numeric_current_iteration[i_j_cords[i][0]][i_j_cords[i][1] + 1]

        #convert each cordinate ix,jy of F into a number k
        k_i_j = k_cords[(i_j_cords[i][0],i_j_cords[i][1])]
        # with k_i_j, I know where the F variable is located
        jacobian_matrix[i][k_i_j] = 1 + (h/8)*(UP_NEIGHBOUR_NUMERIC_VALUE_CELL - DOWN_NEIGHBOUR_NUMERIC_VALUE_CELL)
        
        # now I set in the row i of the jacobian that is the same row of F, the value of the derivate of the up neighbor
        if  UP_NEIGHBOUR_CLASS_CELL == 'FLUID':
            k_i_minus_one_j = k_cords[(i_j_cords[i][0] - 1,i_j_cords[i][1])]
            jacobian_matrix[i][k_i_minus_one_j] = (-1/4)*(1-(h/2)*CURRENT_NEIGHBOUR_NUMERIC_VALUE_CELL)
        # now I set in the row i of the jacobian that is the same row of F, the value of the derivate of the down neighbor
        if  DOWN_NEIGHBOUR_CLASS_CELL == 'FLUID':
            k_i_plus_one_j = k_cords[(i_j_cords[i][0] + 1,i_j_cords[i][1])]
            jacobian_matrix[i][k_i_plus_one_j] = (-1/4)*(1+(h/2)*CURRENT_NEIGHBOUR_NUMERIC_VALUE_CELL)
        # now I set in the row i of the jacobian that is the same row of F, the value of the derivate of the left neighbor
        if  LEFT_NEIGHBOUR_CLASS_CELL == 'FLUID':
            k_i_j_minus_one = k_cords[(i_j_cords[i][0],i_j_cords[i][1] - 1)]
            jacobian_matrix[i][k_i_j_minus_one] = (-1/4)*(1+(h/2)*vorticity_or_v_i_j_y)
        # now I set in the row i of the jacobian that is the same row of F, the value of the derivate of the right neighbor
        if  RIGHT_NEIGHBOUR_CLASS_CELL== 'FLUID':
            k_i_j_plus_one = k_cords[(i_j_cords[i][0],i_j_cords[i][1] + 1)]
            jacobian_matrix[i][k_i_j_plus_one] = (-1/4)*(1-(h/2)*vorticity_or_v_i_j_y)

    return jacobian_matrix






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



import numpy as np

def spectral_radius_Tgs(A, max_iter=5000, tol=1e-10):
    A = np.asarray(A, float)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A,  1)
    M = D + L  # triangular inferior con diagonal

    n = A.shape[0]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)

    hist = []
    for k in range(1, max_iter+1):
        y = U @ x
        # resolver M z = y  (triangular inferior)
        z = np.linalg.solve(M, y)
        x_new = z / (np.linalg.norm(z) or 1.0)

        # Rayleigh “por operador”: usa norma de crecimiento como estimador
        est = np.linalg.norm(z) / (np.linalg.norm(x) or 1.0)
        hist.append(est)

        if np.linalg.norm(x_new - x) <= tol:
            return est, {"method":"power-operator","converged":True,"iters":k,"estimate_history":hist}
        x = x_new

    return hist[-1], {"method":"power-operator","converged":False,"iters":max_iter,"estimate_history":hist}



if __name__ == "__main__":
    matrix_height = 8
    matrix_width = 80

    fluid_inlet_velocity = 1
    fluid_cells_initial_velocity = 0

    vorticity_or_v_i_j_y = 4

    h = 1 # number for wich the widht and hight is divided to get to matrix_width and matrix_height

    matrix_grid_coordinates, matrix_grid_naming, matrix_grid_initial_values = create_initial_matrixis()

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

    vector_f_numeric = np.array(calculate_f_for_this_iteration(matrix_grid_initial_values), dtype=float)

    print('Initial vector f')
    print('Initial vector f dim: ', len(vector_f_numeric))
    print(vector_f_numeric)
    print("\n")

    jacobian_numeric = np.array(calculate_jacobian_numeric_for_this_iteration(matrix_grid_initial_values), dtype=float)

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

    # --- chequeos previos útiles ---
    diag = np.diag(jacobian_numeric)
    if np.any(~np.isfinite(jacobian_numeric)):
        raise ValueError("El jacobiano contiene NaN/Inf.")
    if np.any(np.isclose(diag, 0.0, atol=1e-14)):
        idx0 = np.where(np.isclose(diag, 0.0, atol=1e-14))[0]
        raise ValueError(f"Jacobian con diagonal ~0 en filas {idx0.tolist()}. min|diag|={np.min(np.abs(diag)):.3e}")

    print(f"min|diag(J)| = {np.min(np.abs(diag)):.3e}, max|diag(J)| = {np.max(np.abs(diag)):.3e}")

    print("Dominancia diagonal:", is_diag_dominant(jacobian_numeric))
    x_gs, it_gs = gauss_seidel(jacobian_numeric, rhs, x0=vector_x_initial, tol=1e-10, max_iter=2000)

    print(f"Gauss-Seidel:  iter={it_gs},  ||Ax-b||={np.linalg.norm(jacobian_numeric@x_gs - vector_f_numeric):.3e}")

    print(spectral_radius_Tgs(jacobian_numeric))

    def rho_T_GS_exact(A):
        A = np.asarray(A, float)
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A,  1)
        Tgs = -np.linalg.solve(D + L, U)      # T_GS = -(D+L)^{-1} U
        w = np.linalg.eigvals(Tgs)
        return float(np.max(np.abs(w)))
    
    print(rho_T_GS_exact(jacobian_numeric))

    diag = np.diag(jacobian_numeric)
    print("min|diag(J)|:", np.min(np.abs(diag)))
    print("||J||_inf:", np.linalg.norm(jacobian_numeric, ord=np.inf))
    print("cond(J) (2-norm, aprox):", np.linalg.cond(jacobian_numeric))
    
    #plt.ion()
    figure, axis = plt.subplots(figsize=(10, 2))
    image = axis.imshow(matrix_grid_initial_values, origin="upper", cmap='viridis', interpolation='nearest', animated=True)
    axis.set_aspect('equal', adjustable='box')
    title = axis.set_title("Valores de la matriz")
    axis.set_xlabel("x"); axis.set_ylabel("y")
    figure.tight_layout()

    final_grid = copy.deepcopy(matrix_grid_initial_values)
    counter = 0
    for i in range(0,matrix_height):
        for j in range(0,matrix_width):
            if matrix_grid_naming[i][j] == 'FLUID':
                final_grid[i][j] = x_gs[counter]
                counter += 1
    image.set_data(final_grid)

    plt.show()