import copy
import numpy as np
import matplotlib.pyplot as plt

def create_initial_matrixis(matrix_height, matrix_width, fluid_cells_initial_velocity, fluid_inlet_velocity):

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


def calculate_jacobian_numeric_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_numeric_current_iteration, vorticity_or_v_i_j_y, h):

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

def newton_rapshon(matrix_height, matrix_width, vector_x_initial, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y):
    matrix_grid_numeric_current_iteration = copy.deepcopy(matrix_grid_initial_values)
    vector_x_current_iteration = vector_x_initial
    vector_solution_next_iteration = []

    plt.ion()
    figure, axis = plt.subplots(figsize=(10, 2))
    image = axis.imshow(matrix_grid_numeric_current_iteration, origin="upper", cmap='viridis', interpolation='nearest', animated=True)
    axis.set_aspect('equal', adjustable='box')
    title = axis.set_title("Valores de la matriz")
    axis.set_xlabel("x"); axis.set_ylabel("y")
    figure.tight_layout()

    for current_iter in range(0,500):

        f_vector = calculate_f_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_numeric_current_iteration, vorticity_or_v_i_j_y, h)
        jacobian_matrix = calculate_jacobian_numeric_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_numeric_current_iteration, vorticity_or_v_i_j_y, h)
        
        #improvement maybe unnecesary - soft the graphic
        #jacobian_matrix_numpy = np.array(jacobian_matrix, dtype=float)
        #regularization_lambda = 5
        #regularized_jacobian = jacobian_matrix_numpy + regularization_lambda*np.eye(len(jacobian_matrix_numpy))
        #improvement maybe unnecesary - soft the graphic

        vector_solution_next_iteration = np.array(vector_x_current_iteration) - np.linalg.inv(np.array(jacobian_matrix)).dot(np.array(f_vector))


        counter = 0
        for i in range(0,matrix_height):
            for j in range(0,matrix_width):
                if matrix_grid_naming[i][j] == 'FLUID':
                    matrix_grid_numeric_current_iteration[i][j] = vector_solution_next_iteration[counter]
                    counter += 1





        # the algorithm should stop?
        delta_vector_step = np.array(vector_solution_next_iteration, dtype=float) - np.array(vector_x_current_iteration,dtype=float)
        tau_absolute = 1e-5
        step_norm_L2 = np.linalg.norm(delta_vector_step, ord=2)
        if step_norm_L2 < tau_absolute:
            print(f'Stop because of small step: {step_norm_L2:.8e} < {tau_absolute:.8e} (iter {current_iter+1})') 
            # update drawing
            image.set_data(matrix_grid_numeric_current_iteration)
            title.set_text(f'Valores de la matriz - iter (final) {current_iter+1} - Step norm delta {step_norm_L2:.8e} ')
            plt.pause(1)
            break
        else:
            print(f'step: {step_norm_L2:.8e}, (iter {current_iter+1})')
            # update drawing
            image.set_data(matrix_grid_numeric_current_iteration)
            title.set_text(f'Valores de la matriz - iter {current_iter+1} - Step norm delta {step_norm_L2:.8e} ')
            plt.pause(0.00001)


        # update vector x
        vector_x_current_iteration = vector_solution_next_iteration



    # disable interactive mode and display figures
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    matrix_height = 8
    matrix_width = 80

    fluid_inlet_velocity = 1
    fluid_cells_initial_velocity = 0

    vorticity_or_v_i_j_y = 4

    h = 1 # number for wich the widht and hight is divided to get to matrix_width and matrix_height

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

    vector_f_numeric = calculate_f_for_this_iteration(matrix_height, matrix_width, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y, h)

    print('Initial vector f')
    print('Initial vector f dim: ', len(vector_f_numeric))
    print(vector_f_numeric)
    print("\n")

    jacobian_numeric = calculate_jacobian_numeric_for_this_iteration(matrix_height,matrix_width, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y, h)

    print('Initial jacobian numeric')
    print('Initial jacobian numeric dim: ', f'{len(jacobian_numeric),len(jacobian_numeric[0])}')
    for i in jacobian_numeric:
        print(i)
    print("\n")

    vector_x_initial = [fluid_cells_initial_velocity for _ in range(0,len(vector_f_numeric))]
    print('Vector x initial')
    print('Vector x initial dim: ', len(vector_x_initial))
    print(vector_x_initial)
    print("\n")
    
    newton_rapshon = newton_rapshon(matrix_height, matrix_width, vector_x_initial, matrix_grid_naming, matrix_grid_initial_values, vorticity_or_v_i_j_y)
    print('Newton raphson process is finished')

