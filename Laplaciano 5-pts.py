# -*- coding: utf-8 -*-
"""
Malla 8x80 definida como array 2D:
  1 = fluido, 0 = obstáculo (u=0).
Solver: Laplaciano 5-pts con Neumann en x=0 y x=Nx-1, y Neumann (v0) en y.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------------------
# 1) Parámetros del problema
# ---------------------------
Ny, Nx = 8, 80
dx = dy = 1.0
v0 = 1.0
apply_top_neumann    = True    # y = 0
apply_bottom_neumann = True    # y = Ny-1

# ---------------------------------------------
# 2) MATRIZ 2D (LISTA DE LISTAS) DE LA MALLA
#    1 = FLUIDO, 0 = OBSTÁCULO
#    -> EDITA estas 8 filas para tu malla real
# ---------------------------------------------
FLUID_01 = [
#  j:  0 1 2 3 4 5 6 7 8 9 10 ...                                 ... 70 71 72 73 74 75 76 77 78 79
    [1]*80,                                                                                         # i=0
    [1]*80,                                                                                         # i=1
    [1]*80,                                                                                         # i=2
    [1]*80,                                                                                         # i=3
    [1]*80,                                                                                         # i=4
    [1]*80,                                                                                         # i=5
    [1]*80,                                                                                         # i=6
    [1]*80,                                                                                         # i=7
]

# --- EJEMPLO: marcar dos obstáculos (pon 0s en rangos) ---
#   Obstáculo 1: fila i=1..2, columnas j=24..33
for i in range(0, 4):
    for j in range(70, 80):
        FLUID_01[i][j] = 0
#   Obstáculo 2: fila i=4..5, columnas j=52..59
for i in range(4, 8):
    for j in range(34, 44):
        FLUID_01[i][j] = 0

# --- Validación de dimensiones ---
if len(FLUID_01) != Ny or any(len(row) != Nx for row in FLUID_01):
    raise ValueError(f"La malla debe ser {Ny}x{Nx}. Revisa FLUID_01.")

# Convertir a máscara booleana (True=fluido, False=obstáculo)
fluid_mask = np.array(FLUID_01, dtype=bool)
is_obstacle = ~fluid_mask

# ------------------------------------------------
# 3) Mapeo 2D <-> 1D solo en celdas de FLUIDO
# ------------------------------------------------
ij_to_k = -np.ones((Ny, Nx), dtype=int)
k_to_ij = []
for i in range(Ny):
    for j in range(Nx):
        if fluid_mask[i, j]:
            ij_to_k[i, j] = len(k_to_ij)
            k_to_ij.append((i, j))
k_to_ij = np.array(k_to_ij, dtype=int)
num_unknowns = len(k_to_ij)
if num_unknowns == 0:
    raise RuntimeError("No hay celdas de fluido.")

# Ancla para evitar singularidad con solo Neumann
anchor_k = 0

# -----------------------------------------------
# 4) Ensamblaje F(u)=0 y Jacobiano
# -----------------------------------------------
def build_F_J(u_vec):
    F = np.zeros(num_unknowns)
    rows, cols, data = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); data.append(v)

    for kk, (i, j) in enumerate(k_to_ij):
        if kk == anchor_k:
            F[kk] = u_vec[kk]
            add(kk, kk, 1.0)
            continue

        coeffP = -4.0
        rhs = 0.0

        # Oeste
        if j - 1 >= 0:
            if fluid_mask[i, j-1]:
                kW = ij_to_k[i, j-1]
                add(kk, kW, 1.0); F[kk] += u_vec[kW]
        else:
            coeffP += 1.0  # du/dx=0 en x=0

        # Este
        if j + 1 < Nx:
            if fluid_mask[i, j+1]:
                kE = ij_to_k[i, j+1]
                add(kk, kE, 1.0); F[kk] += u_vec[kE]
        else:
            coeffP += 1.0  # du/dx=0 en x=Nx-1

        # Norte (y=0)
        if i - 1 >= 0:
            if fluid_mask[i-1, j]:
                kN = ij_to_k[i-1, j]
                add(kk, kN, 1.0); F[kk] += u_vec[kN]
        else:
            if apply_top_neumann:
                coeffP += 1.0
                rhs += -v0 * dy

        # Sur (y=Ny-1)
        if i + 1 < Ny:
            if fluid_mask[i+1, j]:
                kS = ij_to_k[i+1, j]
                add(kk, kS, 1.0); F[kk] += u_vec[kS]
        else:
            if apply_bottom_neumann:
                coeffP += 1.0
                rhs += -v0 * dy

        F[kk] += coeffP * u_vec[kk] + rhs
        add(kk, kk, coeffP)

    if SCIPY_OK:
        J = sp.csr_matrix((data, (rows, cols)), shape=(num_unknowns, num_unknowns))
    else:
        J = np.zeros((num_unknowns, num_unknowns))
        for r, c, v in zip(rows, cols, data):
            J[r, c] += v
    return F, J

# ---------------------------
# 5) Newton–Raphson
# ---------------------------
u = np.zeros(num_unknowns)
tol = 1e-12
for it in range(20):
    F, J = build_F_J(u)
    res = np.linalg.norm(F, ord=np.inf)
    print(f"Iter {it:02d}  ||F||_inf = {res:.3e}")
    if res < tol:
        break
    if SCIPY_OK:
        delta = spla.spsolve(J, -F)
    else:
        delta = np.linalg.solve(J, -F)
    u += delta

# ---------------------------
# 6) Reconstrucción y plots
# ---------------------------
U = np.full((Ny, Nx), np.nan)
for kk, (i, j) in enumerate(k_to_ij):
    U[i, j] = u[kk]

plt.figure(figsize=(12, 3.2))
im = plt.imshow(U, origin="upper", aspect="auto", cmap="viridis")
plt.title("Campo u(x,y) (NaN = obstáculo)")
plt.xlabel("x (col)"); plt.ylabel("y (fila)")
plt.colorbar(im, label="u")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 1.8))
plt.imshow(fluid_mask, origin="upper", aspect="auto", cmap="gray_r")
plt.title("Máscara de fluido (blanco) vs obstáculo (negro)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()
