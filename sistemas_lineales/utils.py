import copy
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

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





def _stopped(A: List[List[float]], b: List[float], x_current: List[float], tolerance: float) -> bool:
    """
    Criterio de paro por residuo (norma Euclidiana):
        || b - A * x_current ||_2 < tolerance

    Parámetros
    ----------
    A          : matriz del sistema (lista de listas), de tamaño m x n
    b          : vector de términos independientes, de tamaño m
    x_current  : aproximación actual de la solución, de tamaño n
    tolerance  : tolerancia de convergencia (ε)

    Retorna
    -------
    bool : True si la norma del residuo es menor que 'tolerance'; False en caso contrario.
    residual: Number
    """
    residual_norm_sq = 0.0  # Acumulará ||r||_2^2

    # Recorre cada fila i de A para construir (A x_current)_i y el residuo r_i = b_i - (A x)_i
    for i in range(len(A)):               # 1er bucle: filas de A
        Ax_i = 0.0
        row_i = A[i]

        for j in range(len(row_i)):       # 2do bucle: producto fila_i · x_current
            Ax_i += row_i[j] * x_current[j]

        r_i = b[i] - Ax_i                 # Residuo en la componente i
        residual_norm_sq += r_i * r_i     # Suma de cuadrados para la norma 2

    residual_norm = math.sqrt(residual_norm_sq)
    return residual_norm < tolerance, residual_norm


def create_initial_matrixis(matrix_height, matrix_width, fluid_cells_initial_velocity, fluid_inlet_velocity, ref_n = 1):

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
            if ((i < ref_n)  or i >= matrix_height - ref_n or j >= matrix_width - ref_n): # Outer Walls
                matrix_grid_naming[i][j] = 'OWALL'
                matrix_grid_to_show[i][j] = 0
            elif ((i < matrix_height) and (i > int(matrix_height*0.4))) and ((j <  int(matrix_width*0.55)) and (j > int(matrix_width*0.43))): # Bottom beam
                matrix_grid_naming[i][j] = 'BEAM1'
                matrix_grid_to_show[i][j] = 1
            elif ((i < matrix_height*0.5) and (i > 0)) and ((j <  matrix_width - ref_n) and (j > int(matrix_width*0.87))): # Top beam
                matrix_grid_naming[i][j] = 'BEAM2'
                matrix_grid_to_show[i][j] = 2
            elif (i > 0 and i < matrix_height - ref_n and j < ref_n):
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
def calculate_f_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_numeric_current_iteration, vorticity_or_v_i_j_y, h):
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

def calculate_jacobian_numeric_for_this_iteration(matrix_height,matrix_width, matrix_grid_naming, matrix_grid_numeric_current_iteration, vorticity_or_v_i_j_y, h):

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




import copy
import matplotlib.pyplot as plt

class LiveGridPlotter:
    """
    Actualiza una figura imshow en cada iteración con los valores de x en las celdas FLUID.
    """
    def __init__(self, matrix_grid_naming, base_grid_2d, title="Valores de la matriz"):
        self.matrix_grid_naming = matrix_grid_naming
        self.base_grid_2d = base_grid_2d
        self.title_text = title

        # Mapeo fila-columna de las celdas FLUID en orden fila mayor (i,j)
        self.fluid_positions = []
        for i in range(len(matrix_grid_naming)):
            for j in range(len(matrix_grid_naming[0])):
                if matrix_grid_naming[i][j] == 'FLUID':
                    self.fluid_positions.append((i, j))

        # Figura
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 2))
        self.im = self.ax.imshow(base_grid_2d, origin="upper", cmap='viridis', interpolation='nearest', animated=True)
        self.ax.set_aspect('equal', adjustable='box')
        self.title = self.ax.set_title(self.title_text)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, k, x_vec, residual_norm, elapsed):
        """
        k: iteración actual (int)
        x_vec: vector 1D con valores SOLO en celdas FLUID, en el mismo orden del ensamblaje
        residual_norm: norma del residuo (opcional), para mostrar en el título
        """
        grid = copy.deepcopy(self.base_grid_2d)
        idx = 0
        # Volcar x_vec sobre las posiciones FLUID
        for (i, j) in self.fluid_positions:
            grid[i][j] = x_vec[idx]
            idx += 1

        # Actualizar imagen y título
        self.im.set_data(grid)
        if residual_norm is not None:
            self.title.set_text(f"{self.title_text}  |  iter={k}  |  || b - A * x_current ||_2={residual_norm:.2e}  |  time={elapsed:.2f}s")
        else:
            self.title.set_text(f"{self.title_text}  |  iter={k}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0000000001)  # refresco no bloqueante

    def show_blocking(self):
        """Deja la ventana abierta de forma bloqueante (al final)."""
        plt.ioff()
        plt.show()
