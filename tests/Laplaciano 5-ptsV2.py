# -*- coding: utf-8 -*-
"""
Laplace 2D (estacionario) en malla 8x80 con obstáculos usando stencil 5 puntos.
Ecuación en fluido:  u_xx + u_yy = 0     (Laplaciano puro)

BCs:
  - Techo (y=0):       Neumann   du/dy = V0_TOP
  - Izquierda (x=0):   Neumann   du/dx = LEFT_gx
  - Derecha (x=Nx-1):  Neumann   du/dx = RIGHT_gx
  - Base (y=Ny-1):     Dirichlet u = BOT_u
  - Obstáculos:        Dirichlet u = 0 (pared interna via ghost)

Discretización:
  - Stencil 5 puntos:  uE + uW + uN + uS - 4 uP = 0, ajustado por ghosts/BCs.
  - Ensamblaje analítico de A (dispersa si SciPy disponible) y b.

Notas:
  - Neumann se maneja con nodo fantasma: p.ej. en la izquierda uW = uP - gx*dx, lo que suma +1 a aP
    y agrega término a b. Dirichlet (base y paredes internas) usan ghost uGhost = 2*ub - uP.
  - Como es lineal, resolvemos A u = b directamente (sin Newton).
"""

import numpy as np
import matplotlib.pyplot as plt

# SciPy disperso (opcional, recomendado)
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------------------
# 1) Parámetros de malla y BCs
# ---------------------------
Ny, Nx = 8, 80
dx = dy = 1.0
if dx <= 0 or dy <= 0:
    raise ValueError("dx y dy deben ser > 0.")

# Gradientes Neumann y valor Dirichlet
LEFT_gx  = 0.0     # du/dx en x=0
RIGHT_gx = 0.0     # du/dx en x=Nx-1
V0_TOP   = -1     # du/dy en y=0
BOT_u    = 0.0     # u en y=Ny-1

# ---------------------------
# 2) Máscara de fluido / obstáculo (1=fluido, 0=obstáculo)
# ---------------------------
FLUID_01 = [[1]*Nx for _ in range(Ny)]

# Obstáculo alto-derecha (demo)
for i in range(0, 4):
    for j in range(70, 80):
        FLUID_01[i][j] = 0

# Obstáculo bajo-centro (demo)
for i in range(4, 8):
    for j in range(34, 44):
        FLUID_01[i][j] = 0

fluid_mask = np.array(FLUID_01, dtype=bool)
if not fluid_mask.any():
    raise RuntimeError("No hay celdas de fluido.")
is_obstacle = ~fluid_mask

# ---------------------------
# 3) Mapeo 2D <-> 1D (solo fluido)
# ---------------------------
ij_to_k = -np.ones((Ny, Nx), dtype=int)
k_to_ij = []
for i in range(Ny):
    for j in range(Nx):
        if fluid_mask[i, j]:
            ij_to_k[i, j] = len(k_to_ij)
            k_to_ij.append((i, j))
k_to_ij = np.array(k_to_ij, dtype=int)
nunk = len(k_to_ij)

# ---------------------------
# 4) Ensamblaje analítico de A y b (stencil 5 puntos con ghosts)
# ---------------------------
def assemble_system():
    """
    Construye A, b para A u = b (Laplace=0) en nodos de fluido.
    Trata:
      - Vecino fluido: coef 1 hacia ese vecino.
      - Vecino obstáculo (Dirichlet u=0): ghost -> uGhost = -uP -> suma (-1) a aP.
      - Borde Neumann: uGhost = uP +/- g*h -> suma (+1) a aP y término a b.
      - Base Dirichlet u=BOT_u: uGhost = 2*BOT_u - uP -> contribuye (-1) a aP y 2*BOT_u a b.
    """
    # Reservas para matriz dispersa
    rows = []
    cols = []
    data = []
    b = np.zeros(nunk, dtype=float)

    def add_entry(row, col, val):
        rows.append(row); cols.append(col); data.append(val)

    for kk, (i, j) in enumerate(k_to_ij):
        aP = -4.0
        # West (i, j-1)
        if j-1 >= 0:
            if fluid_mask[i, j-1]:
                add_entry(kk, ij_to_k[i, j-1], 1.0)
            else:
                # obstáculo -> uW = -uP  => aP += (-1)
                aP += -1.0
        else:
            # Borde izquierdo Neumann: uW = uP - LEFT_gx*dx
            aP += +1.0
            b[kk] += -(LEFT_gx * dx)

        # East (i, j+1)
        if j+1 < Nx:
            if fluid_mask[i, j+1]:
                add_entry(kk, ij_to_k[i, j+1], 1.0)
            else:
                # obstáculo -> uE = -uP
                aP += -1.0
        else:
            # Borde derecho Neumann: uE = uP + RIGHT_gx*dx
            aP += +1.0
            b[kk] += +(RIGHT_gx * dx)

        # North (i-1, j)  (techo)
        if i-1 >= 0:
            if fluid_mask[i-1, j]:
                add_entry(kk, ij_to_k[i-1, j], 1.0)
            else:
                # obstáculo arriba -> uN = -uP
                aP += -1.0
        else:
            # Techo Neumann: uN = uP + V0_TOP*dy  (convención usada)
            aP += +1.0
            b[kk] += +(V0_TOP * dy)

        # South (i+1, j)  (base)
        if i+1 < Ny:
            if fluid_mask[i+1, j]:
                add_entry(kk, ij_to_k[i+1, j], 1.0)
            else:
                # obstáculo abajo -> uS = -uP
                aP += -1.0
        else:
            # Base Dirichlet u=BOT_u: uS = 2*BOT_u - uP  => contrib (-uP) a aP y +2*BOT_u a b
            aP += -1.0
            b[kk] += 2.0 * BOT_u

        # Diagonal
        add_entry(kk, kk, aP)

    if SCIPY_OK:
        A = sp.csr_matrix((data, (rows, cols)), shape=(nunk, nunk))
    else:
        # Fallback denso
        A = np.zeros((nunk, nunk), dtype=float)
        for r, c, v in zip(rows, cols, data):
            A[r, c] += v
    return A, b

A, b = assemble_system()

# ---------------------------
# 5) Resolver A u = b
# ---------------------------
if SCIPY_OK:
    u = spla.spsolve(A, b)
else:
    u = np.linalg.solve(A, b)

# ---------------------------
# 6) Reconstrucción y gráficos
# ---------------------------
U = np.full((Ny, Nx), np.nan, dtype=float)
for kk, (i, j) in enumerate(k_to_ij):
    U[i, j] = u[kk]

import matplotlib as mpl
cmap = mpl.cm.viridis.copy()
cmap.set_bad('k')  # obstáculos en negro

plt.figure(figsize=(12, 3.2))
im = plt.imshow(U, origin="upper", aspect="auto", cmap=cmap)
plt.title("Laplace 2D (5 puntos) con BCs y obstáculos (NaN = obstáculo)")
plt.xlabel("x (col)"); plt.ylabel("y (fila)")
plt.colorbar(im, label="u")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 1.8))
plt.imshow(fluid_mask, origin="upper", aspect="auto", cmap="gray")
plt.title("Máscara de fluido (blanco) vs obstáculo (negro)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()

print("nunk =", nunk)
