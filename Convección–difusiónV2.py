# -*- coding: utf-8 -*-
"""
Convección–difusión estacionaria 2D en malla 8x80 con obstáculos.
Ecuación:  - (Vx * du/dx + Vy * du/dy) + nu * (d2u/dx2 + d2u/dy2) = 0

BCs (según figura):
  - Surface G (y=0)     : Neumann  du/dy = V0_TOP
  - Inlet F   (x=0)     : Neumann  du/dx = 0
  - Outlet H  (x=Nx-1)  : Neumann  du/dx = 0
  - Base E–A  (y=Ny-1)  : Dirichlet u = 0
  - Obstáculos          : Dirichlet u = 0

Convección:
  - Modo "const"  : Vx=VX_CONST, Vy=VY_CONST (lineal)
  - Modo "burgers": Vx=u (no lineal), Vy=VY_CONST

Método: Newton–Raphson con Jacobiano numérico (robusto para 8×80).
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
nu = 0.20                 # difusividad/viscosidad

# Convección
VX_MODE  = "const"        # "const" o "burgers"
VX_CONST = 1.0            # usado si VX_MODE="const"
VY_CONST = 0.0

# Gradiente en el techo (Surface G)
V0_TOP = +1.0             # impone du/dy = V0_TOP en y=0

# ---------------------------
# 2) Máscara de fluido / obstáculo (1/0)
# ---------------------------
FLUID_01 = [[1]*Nx for _ in range(Ny)]

# >>> EDITA aquí tus obstáculos (ejemplos):
# obstáculo alto-derecha (demo)
for i in range(0, 4):
    for j in range(70, 80):
        FLUID_01[i][j] = 0
# obstáculo bajo-centro (demo)
for i in range(4, 8):
    for j in range(34, 44):
        FLUID_01[i][j] = 0

fluid_mask = np.array(FLUID_01, dtype=bool)
if not fluid_mask.any():
    raise RuntimeError("No hay celdas de fluido en la malla.")
is_obstacle = ~fluid_mask

# ---------------------------
# 3) Mapeo 2D <-> 1D en fluido
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
if nunk == 0:
    raise RuntimeError("No hay incógnitas (todo es obstáculo).")

# Hay Dirichlet en la base → NO necesitamos ancla (el sistema ya es no singular)
anchor_k = None

# ---------------------------
# 4) Ghosts (2º orden) para BCs
# ---------------------------
def ghost_dirichlet(uP, u_b):
    # segundo orden: u_ghost = 2 u_b - u_P
    return 2.0*u_b - uP

def ghost_neumann_left(uP, gx):
    # (uP - uGhost)/dx = gx  -> uGhost = uP - gx*dx
    return uP - gx*dx

def ghost_neumann_right(uP, gx):
    # (uGhost - uP)/dx = gx  -> uGhost = uP + gx*dx
    return uP + gx*dx

def ghost_neumann_vertical(uP, gy):
    # (uGhost - uP)/dy = gy  -> uGhost = uP + gy*dy
    return uP + gy*dy

# valores de borde (según figura)
LEFT_gx  = 0.0      # du/dx=0 en inlet
RIGHT_gx = 0.0      # du/dx=0 en outlet
TOP_gy   = V0_TOP   # du/dy=V0_TOP en techo
BOT_u    = 0.0      # u=0 en base

def get_u_at(i, j, u_vec):
    """u(i,j) considerando fluido/obstáculo y BCs via ghosts."""
    # Dentro del dominio
    if 0 <= i < Ny and 0 <= j < Nx:
        if fluid_mask[i, j]:
            return u_vec[ij_to_k[i, j]]
        else:
            return 0.0  # obstáculo: no deslizamiento

    # Fuera del dominio: aplicar BCs del lado correspondiente
    if j < 0:  # izquierda (x=0): Neumann du/dx=0
        iP, jP = i, 0
        uP = 0.0 if not (0 <= iP < Ny and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_left(uP, LEFT_gx)

    if j >= Nx:  # derecha (x=Nx-1): Neumann du/dx=0
        iP, jP = i, Nx-1
        uP = 0.0 if not (0 <= iP < Ny and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_right(uP, RIGHT_gx)

    if i < 0:  # techo (y=0): Neumann du/dy=V0_TOP
        iP, jP = 0, j
        uP = 0.0 if not (0 <= jP < Nx and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_vertical(uP, TOP_gy)

    if i >= Ny:  # base (y=Ny-1): Dirichlet u=0
        iP, jP = Ny-1, j
        uP = 0.0 if not (0 <= jP < Nx and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_dirichlet(uP, BOT_u)

    raise RuntimeError("Índice fantasma inesperado.")

def convective_velocities(u_vec):
    """Campos de velocidad convectiva (para el término advectivo)."""
    if VX_MODE.lower() == "burgers":
        Vx = np.zeros((Ny, Nx))
        for i, j in k_to_ij:
            Vx[i, j] = u_vec[ij_to_k[i, j]]
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)
    else:
        Vx = np.full((Ny, Nx), VX_CONST, dtype=float)
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)
    return Vx, Vy

# ---------------------------
# 5) Residual F(u)
# ---------------------------
def build_residual(u_vec):
    Vx, Vy = convective_velocities(u_vec)
    F = np.zeros(nunk)

    for kk, (i, j) in enumerate(k_to_ij):
        # (no ancla porque hay Dirichlet en la base)
        uP = get_u_at(i, j, u_vec)
        uW = get_u_at(i, j-1, u_vec)
        uE = get_u_at(i, j+1, u_vec)
        uN = get_u_at(i-1, j, u_vec)
        uS = get_u_at(i+1, j, u_vec)

        # Gradientes centrados
        du_dx = (uE - uW) / (2.0*dx)
        du_dy = (uS - uN) / (2.0*dy)

        # Laplaciano (d2u/dx2 + d2u/dy2)
        lap = (uE - 2.0*uP + uW)/(dx*dx) + (uN - 2.0*uP + uS)/(dy*dy)

        # Convección
        vx = Vx[i, j]; vy = Vy[i, j]
        conv = vx*du_dx + vy*du_dy

        F[kk] = -conv + nu*lap

    return F

# ---------------------------
# 6) Jacobiano numérico
# ---------------------------
def build_jacobian_numeric(u_vec, eps=1e-8):
    F0 = build_residual(u_vec)
    J = np.zeros((nunk, nunk))
    for q in range(nunk):
        u_pert = u_vec.copy()
        du = eps * max(1.0, abs(u_vec[q]))
        u_pert[q] += du
        Fq = build_residual(u_pert)
        J[:, q] = (Fq - F0) / du
    return J

# ---------------------------
# 7) Newton–Raphson (con backtracking)
# ---------------------------
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

        if SCIPY_OK:
            delta = spla.spsolve(J, rhs)
        else:
            delta = np.linalg.solve(J, rhs)

        alpha = 1.0
        if damping:
            for _ in range(10):
                u_try = u + alpha*delta
                r_try = np.linalg.norm(build_residual(u_try), ord=np.inf)
                if r_try <= (1 - 1e-3*alpha) * res:
                    u = u_try
                    break
                alpha *= 0.5
            else:
                u = u + delta
        else:
            u = u + delta
    return u

# ---------------------------
# 8) Ejecutar solver
# ---------------------------
# Arranque: algo suave compatible con base u=0 (p. ej., cero)
u0 = np.zeros(nunk)
u  = newton_solve(u0, max_iter=25, tol=1e-10, damping=True)

# ---------------------------
# 9) Reconstrucción y gráficos
# ---------------------------
U = np.full((Ny, Nx), np.nan)
for kk, (i, j) in enumerate(k_to_ij):
    U[i, j] = u[kk]

plt.figure(figsize=(12, 3.2))
im = plt.imshow(U, origin="upper", aspect="auto", cmap="viridis")
plt.title("Convección–difusión con BCs del esquema (NaN = obstáculo)")
plt.xlabel("x (col)"); plt.ylabel("y (fila)")
plt.colorbar(im, label="u")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 1.8))
plt.imshow(fluid_mask, origin="upper", aspect="auto", cmap="gray_r")
plt.title("Máscara de fluido (blanco) vs obstáculo (negro)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()
