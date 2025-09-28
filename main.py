from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import math
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from mpl_toolkits.mplot3d import axes3d
from scipy import interpolate

Lx = 15 #Longitud de la malla en el eje x
Ly = 15 #Longitud de la malla en el eje y
#Viga 1
init = 4
alto1 = 4
ancho1 = 4
#Viga 2
init2 = 11
alto2 = 4
ancho2 = 4
h = 0.01 #longitud de los pasos
V0 = 1.5 # Velocidad inicial
omega = 0.3 #estabilizador

u = np.zeros((Lx,Ly), float) # arreglo para la funcion U
v = np.zeros((Lx,Ly), float) # arreglo para la funcion v

bu = np.zeros((Lx*Ly),float)
bv = np.zeros((Lx*Ly),float)

for i in range(Lx * Ly):
  bu[i] = h/8
  bv[i] = 0

u0 = np.array([15.0 for _ in range(Lx*Ly)], float)  #Valores iniciales de U
v0 = np.array([15.0 for _ in range(Lx*Ly)], float)  #Valores iniciales de V

def inicializar():
    for i in range(Lx):
        V = 0.3
        for j in range(Ly):
            u[i, j] = round(V - 0.001, 3)
            V = V - 0.001

inicializar()

#Rellenamos las matrices U y V (bloques de las celdas en 0)
def rellenar(m1, m2):
    for i in range (Lx-alto1, Lx):
        for j in range (init, init+ancho1):
            m1[i,j] = 0.
            m2[i,j] = 0.

    for i in range(0, alto2):
        for j in range(init2,init2+ancho2):
            m1[i,j] = 0.
            m2[i,j] = 0.

rellenar(u,v)

def gen_matriz_sis_lineal(n,u,v,h,tipo):
    a = 0
    b = 0
    c = 0
    d = 0
    matriz = []
    for i in range(1, (n ** 2) + 1):
        fila = []
        for j in range(1, (n ** 2) + 1):
            if (tipo==1):
                a = (h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                b = -(h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                c = (h / 8) * v[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                d = -(h / 8) * v[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
            else:
                a = (h / 8) * v[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                b = -(h / 8) * v[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                c = (h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
                d = -(h / 8) * u[math.ceil(i/n) - 1, math.ceil(j/n) - 1] - 1 / 4
            # Primera diagonal
            if (i == j):
                fila.append(1)

            # Segunda diagonal superior
            elif (
                    (i % n != 0 and i + 1 == j)):
                fila.append(a)

            # Segunda diagonal inferior
            elif (
                    ((i % n) - 1 != 0 and i == j + 1)):
                fila.append(b)

            # Tercera diagonal superior
            elif (i + n == j):
                fila.append(c)

            # Tercera diagonal inferior
            elif (i == j + n):
                fila.append(d)

            else:
                fila.append(0)

        matriz.append(fila)
    return matriz


def surfaceG(m, b, flag):
    n = len(u)

    for i in range(0, n):
        if flag == "u":
            b[i] = V0
        else:
            b[i] = 0
        for j in range(0, n * n):
            m[i][j] = 0
            if i == j:
                m[i][j] = 1

    if flag == "u":
        for i in range (0,n):
            for j in range (0,n*n):
                m[i][j] = 0
                if (j == i):
                    m[i][j] = 1
    return m, b


def InletF(m, b, flag):
    n = len(u)
    for i in range(0, n * n, n):
        if flag == "u":
            b[i] = V0
        else:
            b[i] = 0
        for j in range(0, n * n):
            m[i][j] = 0
            if i == j:
                m[i][j] = 1
    return m, b


def outlet(m, b, flag):
    n = len(u)
    for i in range(n - 1, n * n, n):
        b[i] = 0
        for j in range(0, n * n):
            m[i][j] = 0
            if i == j:
                m[i][j] = 1

    for i in range (n-1, n*n, n):
        for j in range(0, n*n):
            if (i+n == j):
                m[i][j] = 1
            if (i == j+n):
                m[i][j] == -1
    return m, b

def centerLine(m, b, flag):
    n = len(u)
    for i in range(n * n - n, n * n):
        b[i] = 0
        for j in range (0, n*n):
            m[i][j] = 0
            if i == j:
                m[i][j] = 1
    return m, b

def transformPairToOne(i, j):
    return (i+1) * Lx - (Lx - j)

def transform(j):
    return j%Lx

def llenar(m, b, flag):
    n = len(u)
    # Rellenar
    # Viga 1
    for i in range(Lx - alto1, Lx):
        for k in range(init, init + ancho1):
            for j in range(0, n * n):
                j2 = math.ceil(j / Lx) - 1
                m[transformPairToOne(i, k)][j] = 0
                if transformPairToOne(i, k) == j:
                    m[transformPairToOne(i, k)][j] = 1

    for i in range(Lx - alto1, Lx):
        for k in range(init, init + ancho1):
            b[transformPairToOne(i, k)] = 0

    # Viga 2
    for i in range(0, alto2):
        for k in range(init2, init2 + ancho2):
            for j in range(0, n * n):
                j2 = math.ceil(j / Ly) - 1
                m[transformPairToOne(i, k)][j] = 0
                if transformPairToOne(i, k) == j:
                    m[transformPairToOne(i, k)][j] = 1

    for i in range(0, alto2):
        for k in range(init2, init2 + ancho2):
            b[transformPairToOne(i, k)] = 0

def viga1(m, b, flag):
    n = len(u)

    # Pared izquierda viga 1
    for i in range(n - alto1, n):
        for j in range(0, n * n):
            if flag == "u":
                b[i * n + init - 1] = 0
            else:
                j2 = math.ceil(j / Ly) - 1
                b[i * n + init - 1] = -2 * (u[i][j2 - 1] - u[i][j2]) / h * h
    # Pared superior viga 1
    f = n - alto1 - 1
    for j in range(init, init + ancho1 + 1):
        if flag == "u":
            b[transformPairToOne(f, j)] = 0
        else:
            j2 = math.ceil(j / Ly) - 1
            i = n - alto1 - 1
            b[transformPairToOne(f, j)] = -2 * (u[i - 1][j] - u[i][j]) / h * h
    # Pared derecha viga 1
    for i in range(n - alto1 - 1, n):
        for j in range(0, n * n):
            if flag == "u":
                b[i * n + init + ancho1] = 0
            else:
                j2 = math.ceil(j / Ly) - 1
                if j2 == init + ancho1:
                    b[i * n + init + ancho1] = -2 * (u[i][j2 + 1] - u[i][j2]) / h * h

    if flag == "u":
        # Pared izquierda viga 1
        for i in range(n - alto1, n):
            for j in range(0, n * n):
                m[i * n + init - 1][j] = 0
                if i * n + init - 1 == j:
                    m[i * n + init - 1][j] = 1
        # Pared superior viga 1
        f = n - alto1 - 1
        for j in range(init, init + ancho1):
            for k in range(0, n * n):
                m[transformPairToOne(f, j)][k] = 0
                if transformPairToOne(f, j) == k:
                    m[transformPairToOne(f, j)][k] = 1
        # Pared derecha viga 1
        for i in range(n - alto1 - 1, n):
            for j in range(0, n * n):
                m[i * n + init + ancho1][j] = 0
                if i * n + init + ancho1 == j:
                    m[i * n + init + ancho1][j] = 1
    else:
        # Pared izquierda viga 1
        for i in range(n - alto1, n):
            for j in range(0, n * n):
                m[i * n + init - 1][j] = 0
                if i * n + init - 1 == j:
                    m[i * n + init - 1][j] = 1

        # Pared superior viga 1
        f = n - alto1 - 1
        for j in range(init, init + ancho1):
            for k in range(0, n * n):
                m[transformPairToOne(f, j)][k] = 0
                if transformPairToOne(f, j) == k:
                    m[transformPairToOne(f, j)][k] = 1
        # Pared derecha viga 1
        for i in range(n - alto1 - 1, n):
            for j in range(0, n * n):
                m[i * n + init + ancho1][j] = 0
                if i * n + init + ancho1 == j:
                    m[i * n + init + ancho1][j] = 1

def viga2(m, b, flag):
    n = len(u)
    # Pared izquierda viga 2
    for i in range(0, alto2 + 1):
        for j in range(0, n * n):
            if flag == "u":
                b[i * n + init2 - 1] = 0
            else:
                j2 = Ly - ancho2 - 1
                # print(i, j2)
                b[i * n + init2 - 1] = -2 * (u[i][j2 - 1] - u[i][j2]) / h * h
                break

    # Pared inferior viga 2
    f = alto2
    for j in range(init2, init2 + ancho2):
        if flag == "u":
            b[transformPairToOne(f, j)] = 0
        else:
            j2 = math.ceil(j / Ly) - 1
            i = alto2
            b[transformPairToOne(f, j)] = -2 * (u[i + 1][j] - u[i][j]) / h * h

    if flag == "u":
        # Pared izquierda viga 2
        for i in range(0, alto2 + 1):
            for j in range(0, n * n):
                m[i * n + init2 - 1][j] = 0
                if i * n + init2 - 1 == j:
                    m[i * n + init2 - 1][j] = 1
        # Pared inferior viga 2
        f = alto2
        for j in range(init2, init2 + ancho2):
            for k in range(0, n * n):
                m[transformPairToOne(f, j)][k] = 0
                if transformPairToOne(f, j) == k:
                    m[transformPairToOne(f, j)][k] = 1
    else:
        # Pared izquierda viga 2
        for i in range(0, alto2 + 1):
            for j in range(0, n * n):
                m[i * n + init2 - 1][j] = 0
                if i * n + init2 - 1 == j:
                    m[i * n + init2 - 1][j] = 1

        # Pared inferior viga 2
        f = alto2
        for j in range(init2, init2 + ancho2):
            for k in range(0, n * n):
                m[transformPairToOne(f, j)][k] = 0
                if transformPairToOne(f, j) == k:
                    m[transformPairToOne(f, j)][k] = 1

#Agregamos los métodos propuestos
def richardson(A,x,b,N):
    for i in range(N):
        r=b-np.dot(A,x)
        x = x + r
    return x

def Jacobi(A,x,b, error=1,tol=1e-5):
  A = np.array(A,dtype=float)
  b = np.array(b,dtype=float)
  x = np.array(x,dtype=float)
  tam = np.shape(A)
  n = tam[0]
  m = tam[1]
  diferencia = np.ones(n,dtype=float)
  xin = np.zeros(n,dtype=float)
  while (error > tol):
    for i in range(0,n,1):
      nuevo = b[i]
      for j in range(0,m,1):
        if (j!=i):
          nuevo = nuevo - A[i,j]*x[j]
      nuevo = nuevo/A[i,i]
      xin[i] = nuevo
      diferencia[i] = np.abs(nuevo-x[i])
      error = np.max(diferencia)
    x = np.copy(xin)
    #print("Error del método de Jacobi: ", error)
  return x

#Soluciona la parte no lineal del sistema de ecuaciones
def Newton_Raphson(A,x,b, error=1, tol=1e-1):
  xi = x.T
  for i in range(0,2):
    deltax = Jacobi(A,xi,b)
    xn = xi - deltax
    error = (np.linalg.norm(xn-xi)/np.linalg.norm(xn))
    xi = xn
    print ("El error de Newton_Raphson es: ", error)
  return xi

#Creamos las matrices con los jacobianos correspondientes
uJac = gen_matriz_sis_lineal(Lx, u, v, 1, 1)
vJac = gen_matriz_sis_lineal(Lx, u, v, 1, 2)

#asignamos las condiciones pertinentes
def condiciones(mu, mv, b1, b2):
    surfaceG(mu, b1, "u")
    surfaceG(mv, b2, "v")
    InletF(mu, b1, "u")
    InletF(mv, b2, "v")
    outlet(mu, b1, "u")
    outlet(mv, b2, "v")
    centerLine(mu, b1, "u")
    centerLine(mv, b2, "v")
    llenar(mu, b1, "u")
    llenar(mv, b2, "v")
    viga1(mu, b1, "u")
    viga1(mv, b2, "v")
    viga2(mu, b1, "u")
    viga2(mv, b2, "v")

condiciones(uJac, vJac, bu, bv)

for i in range(2):
    u0 = Newton_Raphson(uJac,u0,bu)
    v0 = Newton_Raphson(vJac,v0,bv)
    k = 0
    for i in range (0, Lx):
        for j in range (0, Ly):
            u[i,j] = u[i,j] + u0[k]
            v[i,j] = v[i,j] + v0[k]
            k += 1
    newUjac = gen_matriz_sis_lineal(Lx,u,v,1,1)
    newVjac = gen_matriz_sis_lineal(Lx,u,v,1,2)
    condiciones(newUjac,newVjac,bu,bv)

#Ampliamos el tamaño de la matriz, es decir, le realizamos una extension entre cada punto
def scalematrix(m, scale):
  r = np.zeros(((m.shape[0]-1)*scale+1, (m.shape[1]-1)*scale+1))
  for fil in range(m.shape[0]):
    for col in range(m.shape[1]-1):
      r[fil*scale, col*scale:(col+1)*scale+1] = np.linspace(m[fil,col], m[fil,col+1], scale+1)
  # Rellenar resto de ceros, interpolando entre elementos de las columnas
  for fil in range(m.shape[0]-1):
    for col in range(r.shape[1]):
      r[fil*scale:(fil+1)*scale + 1, col] = np.linspace(r[fil*scale,col], r[(fil+1)*scale, col], scale+1)
  return r

matriz1 = scalematrix(u,10)
matriz2 = scalematrix(v,5)

#Interpolación con el método de Lagrange
def Lagrange_interpolation(x,y,A):
  #Rellenamos los vectores
  xi = np.array([i for i in range(A.shape[0])])
  yi = np.array([j for j in range(A.shape[1])])

  px = np.array([1.0 for _ in range(A.shape[0])])
  py = np.array([1.0 for _ in range(A.shape[1])])

  #Creamos el polinomio de interpolacion
  for i in range(len(xi)):
    for j in range(len(yi)):
      if (i!=j):
        px[i] *= ((x-xi[j])/(xi[i]-xi[j]))
        py[i] *= ((y-yi[j])/(yi[i]-yi[j]))
  pol = 0
  #Evaluamos el polinomio
  for i in range(len(xi)):
    for j in range(len(yi)):
      pol += px[i] * (A[i,j]*py[j])
  return pol


#Interpolación usando splines cubicos
def Spline_interpolate(A,B):
  #Se crean los vectores
  xi = np.array([i for i in range(B.shape[0])])
  yi = np.array([j for j in range(B.shape[1])])
  #creamos la malla
  xold,yold = np.meshgrid(xi,yi)
  z = np.zeros((len(xi),len(yi)), dtype = float)
  #Creamos una especie de "copia" de la malla con el zoom
  for i in range(len(xi)):
    for j in range(len(yi)):
      z[i,j] = B[i,j]
  #creamos los splines
  f = interpolate.interp2d(xi,yi,z,kind='cubic')
  newz = f(xi,yi)
  #Graficamos los splines
  fig, axs = plt.subplots(1,2)
  axs[0].imshow(A)
  axs[0].set_title('Malla original')
  axs[1].imshow(newz)
  axs[1].set_title('Interpolación Spline')
  plt.show()

#D = Spline_interpolate(u,matriz1)
#E = Spline_interpolate(v,matriz2)

#En esta parte se procede a evaluar el polinomio creado en la funcion Lagrange pero ahora en los valores de la malla con zoom
#xn = np.array([i for i in range(matriz1.shape[0])])
#yn = np.array([j for j in range(matriz1.shape[1])])
#zu = np.zeros((len(xn),len(yn)), dtype = float)
#zv = np.zeros((len(xn),len(yn)), dtype = float)

#for i in range(len(xn)):
#    for j in range(len(yn)):
#        zu[i,j] =  Lagrange_interpolation(xn[i],yn[j],matriz1)  #Corresponde a la malla U
#        zv[i,j] =  Lagrange_interpolation(xn[i],yn[j],matriz2) #Corresponde a la malla V


fig, axs = plt.subplots(1,2)
axs[0].imshow(u)
axs[0].set_title('Malla original')
plt.show()

