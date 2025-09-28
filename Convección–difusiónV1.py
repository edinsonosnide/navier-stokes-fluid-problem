# -*- coding: utf-8 -*-
"""
Convección + difusión estacionaria en 2D (malla 8x80) con obstáculos.
Ecuación: - (Vx*du/dx + Vy*du/dy) + nu * Laplaciano(u) = 0

BCs:
  - x=0, x=Nx-1: Neumann du/dx = 0
  - y=0, y=Ny-1: Neumann du/dy = v0 (activables)
  - Obstáculo: u = 0 implícito (no deslizamiento)
  - Se fija un "ancla" u=0 para evitar singularidad con Neumann puro

Convección:
  - Modo "burgers": Vx = u (no lineal); Vy = VY_CONST o un campo dado.
  - Modo "const":    Vx = VX_CONST, Vy = VY_CONST (lineal).

Método:
  - Newton–Raphson con Jacobiano numérico por diferencias finitas (robusto).
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------------------
# 1) Parámetros del problema
# ---------------------------
Ny, Nx = 8, 80
dx = dy = 1.0
nu = 1.0                 # difusividad/viscosidad efectiva
v0 = 1.0                 # gradiente impuesto en y (du/dy = v0) en las fronteras superior/inferior
apply_top_neumann    = True   # y = 0
apply_bottom_neumann = True   # y = Ny-1

# Convección: elige el modo
#   "burgers" => Vx = u (no lineal), Vy = VY_CONST
#   "const"   => Vx = VX_CONST, Vy = VY_CONST
VX_MODE  = "burgers"      # "burgers" o "const"
VX_CONST = 0.0
VY_CONST = 0.0

# ------------------------------------------------
# 2) Define la malla/obstáculos (array 2D 0/1)
#    1 = FLUIDO, 0 = OBSTÁCULO
# ------------------------------------------------
FLUID_01 = [
    [1]*80,  # i = 0
    [1]*80,  # i = 1
    [1]*80,  # i = 2
    [1]*80,  # i = 3
    [1]*80,  # i = 4
    [1]*80,  # i = 5
    [1]*80,  # i = 6
    [1]*80,  # i = 7
]

# Obstáculo 1: i=1..2, j=24..33
for i in range(0, 4):
    for j in range(70, 80):
        FLUID_01[i][j] = 0
# Obstáculo 2: i=4..5, j=52..59
for i in range(4, 8):
    for j in range(34, 44):
        FLUID_01[i][j] = 0

fluid_mask = np.array(FLUID_01, dtype=bool)
if not fluid_mask.any():
    raise RuntimeError("No hay celdas de fluido.")
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
nunk = len(k_to_ij)
if nunk == 0:
    raise RuntimeError("No hay incógnitas (fluido).")

anchor_k = 0  # fija u=0 en esta celda para evitar singularidad

# ------------------------------------------------
# 4) Utilidades: vecinos + ghost/BC handling
# ------------------------------------------------
def get_u_at(i, j, u_vec):
    """ Devuelve u en (i,j) manejando:
        - Obstáculo: u=0 (Dirichlet implícita)
        - Fuera del dominio: usa ghost por Neumann (ver más abajo)
        - Fluido: toma de u_vec con mapeo ij_to_k
    """
    # Dentro del dominio
    if 0 <= i < Ny and 0 <= j < Nx:
        if fluid_mask[i, j]:
            return u_vec[ij_to_k[i, j]]
        else:
            return 0.0  # obstáculo
    # Fuera del dominio -> ghost segun BC
    # Para x: du/dx = 0 -> u_ghost = uP
    if j < 0:   # a la izquierda (x=0)
        jP = 0
        if fluid_mask[i, jP]:
            return u_vec[ij_to_k[i, jP]]
        else:
            return 0.0
    if j >= Nx:  # a la derecha (x=Nx-1)
        jP = Nx-1
        if fluid_mask[i, jP]:
            return u_vec[ij_to_k[i, jP]]
        else:
            return 0.0
    # Para y: du/dy = v0 -> u_ghost = uP -/+ v0*dy
    if i < 0:   # arriba (y=0)
        iP = 0
        if not fluid_mask[iP, j]:
            return 0.0
        up = u_vec[ij_to_k[iP, j]]
        if apply_top_neumann:
            # uNghost = uP + v0*dy  (signo corregido)
            return up + v0 * dy
        else:
            # si no aplica Neumann, tratamos como pared: u=0
            return 0.0
    if i >= Ny:  # abajo (y=Ny-1)
        iP = Ny-1
        if not fluid_mask[iP, j]:
            return 0.0
        up = u_vec[ij_to_k[iP, j]]
        if apply_bottom_neumann:
            # uSghost = uP + v0*dy
            return up + v0 * dy
        else:
            return 0.0
    # debería estar cubierto todo
    return 0.0

def convective_velocities(u_vec):
    """ Devuelve campos Vx, Vy en todo el dominio (solo usamos en nodos de fluido). """
    if VX_MODE.lower() == "burgers":
        # Vx = u (no lineal)
        Vx = np.zeros((Ny, Nx))
        for i, j in k_to_ij:
            Vx[i, j] = u_vec[ij_to_k[i, j]]
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)
    else:
        Vx = np.full((Ny, Nx), VX_CONST, dtype=float)
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)
    return Vx, Vy

# ------------------------------------------------
# 5) Residual F(u) del sistema no lineal
# ------------------------------------------------
def build_residual(u_vec):
    """
    F_k = - (Vx*du/dx + Vy*du/dy) + nu * Laplaciano(u)
    con BCs y obstáculos incorporados vía get_u_at.
    """
    Vx, Vy = convective_velocities(u_vec)
    F = np.zeros(nunk)

    for kk, (i, j) in enumerate(k_to_ij):
        if kk == anchor_k:
            F[kk] = u_vec[kk]  # ancla u=0
            continue

        # Vecinos con ghosts/obstáculos
        uP = get_u_at(i, j, u_vec)
        uW = get_u_at(i, j-1, u_vec)
        uE = get_u_at(i, j+1, u_vec)
        uN = get_u_at(i-1, j, u_vec)
        uS = get_u_at(i+1, j, u_vec)

        # Gradientes centrados
        du_dx = (uE - uW) / (2.0 * dx)
        du_dy = (uS - uN) / (2.0 * dy)

        # Laplaciano 5-pt
        lap = (uE + uW + uN + uS - 4.0 * uP) / (dx*dy/dy)  # dx=dy -> /1 ; lo dejo explícito

        # Convección
        vx = Vx[i, j]
        vy = Vy[i, j]
        conv = vx * du_dx + vy * du_dy

        # Residual
        F[kk] = -conv + nu * lap

    return F

# ------------------------------------------------
# 6) Jacobiano numérico (diferencias finitas)
# ------------------------------------------------
def build_jacobian_numeric(u_vec, eps=1e-8):
    """
    Jacobiano J ~ dF/du (nunk x nunk) por diferencias finitas hacia adelante.
    Para 8x80 es viable; es claro y robusto.
    """
    F0 = build_residual(u_vec)
    J = np.zeros((nunk, nunk))
    for q in range(nunk):
        u_pert = u_vec.copy()
        du = eps * max(1.0, abs(u_vec[q]))
        u_pert[q] += du
        Fq = build_residual(u_pert)
        J[:, q] = (Fq - F0) / du
    return J

# ------------------------------------------------
# 7) Newton–Raphson con damping (opcional)
# ------------------------------------------------
def newton_solve(u0, max_iter=30, tol=1e-10, damping=True):
    u = u0.copy()
    for it in range(max_iter):
        F = build_residual(u)
        res = np.linalg.norm(F, ord=np.inf)
        print(f"Iter {it:02d}  ||F||_inf = {res:.3e}")
        if res < tol:
            break

        J = build_jacobian_numeric(u)
        rhs = -F

        # Resolver
        if SCIPY_OK:
            delta = spla.spsolve(J, rhs)
        else:
            delta = np.linalg.solve(J, rhs)

        # Paso con damping si conviene
        alpha = 1.0
        if damping:
            # backtracking simple
            for _ in range(10):
                u_try = u + alpha * delta
                r_try = np.linalg.norm(build_residual(u_try), ord=np.inf)
                if r_try <= (1 - 1e-3 * alpha) * res:
                    u = u_try
                    break
                alpha *= 0.5
            else:
                # si no mejora, acepta paso completo igualmente (para no estancarse)
                u = u + delta
        else:
            u = u + delta
    return u

# ------------------------------------------------
# 8) Ejecutar solver
# ------------------------------------------------
u0 = np.zeros(nunk)          # arranque
u  = newton_solve(u0, max_iter=25, tol=1e-10, damping=True)

# ------------------------------------------------
# 9) Reconstrucción y visualización
# ------------------------------------------------
U = np.full((Ny, Nx), np.nan)
for kk, (i, j) in enumerate(k_to_ij):
    U[i, j] = u[kk]

plt.figure(figsize=(12, 3.2))
im = plt.imshow(U, origin="upper", aspect="auto", cmap="viridis")
plt.title("Convección–difusión: u(x,y) (NaN = obstáculo)")
plt.xlabel("x (col)"); plt.ylabel("y (fila)")
plt.colorbar(im, label="u")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 1.8))
plt.imshow(fluid_mask, origin="upper", aspect="auto", cmap="gray_r")
plt.title("Máscara de fluido (blanco) vs obstáculo (negro)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()
