# -*- coding: utf-8 -*-
"""
Convección–difusión estacionaria 2D en malla 8x80 con obstáculos.
Ecuación:  - (Vx * du/dx + Vy * du/dy) + nu * (d2u/dx2 + d2u/dy2) = 0

BCs (según figura):
  - Surface G (y=0)     : Neumann  du/dy = V0_TOP
  - Inlet F   (x=0)     : Neumann  du/dx = 0
  - Outlet H  (x=Nx-1)  : Neumann  du/dx = 0
  - Base E–A  (y=Ny-1)  : Dirichlet u = 0
  - Obstáculos          : Dirichlet u = 0 (no deslizamiento en pared interna mediante ghost)

Convección:
  - Modo "const"  : Vx=VX_CONST, Vy=VY_CONST (lineal)
  - Modo "burgers": Vx=u (no lineal), Vy=VY_CONST

Método: Newton–Raphson con Jacobiano numérico + backtracking robusto.
Malla chica (8×80) -> versión densa; para mallas grandes conviene Jacobiano analítico y/o Newton–Krylov.
"""

import numpy as np
import matplotlib.pyplot as plt

# SciPy es opcional; si está disponible, usamos su solver denso estable
try:
    import scipy.linalg as sla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------------------------
# 1) Parámetros del problema
# ---------------------------
Ny, Nx = 8, 80
dx = dy = 1.0
if dx <= 0 or dy <= 0:
    raise ValueError("dx y dy deben ser positivos.")
nu = 0.5  # difusividad/viscosidad > 0
if nu <= 0:
    raise ValueError("nu debe ser positivo.")

# Convección
VX_MODE  = "const"       # "const" o "burgers"
VX_CONST = 1.0           # usado si VX_MODE="const"
VY_CONST = 0.0

# Gradiente en el techo (Surface G)
V0_TOP = 1.0             # impone du/dy = V0_TOP en y=0

# Umbral de Péclet local para conmutar a upwind (recomendado: 1.0~2.0)
PE_UPWIND = 1.0

# ---------------------------
# 2) Máscara de fluido / obstáculo (1/0)
# ---------------------------
FLUID_01 = [[1]*Nx for _ in range(Ny)]

# >>> EDITA aquí tus obstáculos (ejemplos de demo coherentes con tu caso):
# obstáculo alto-derecha
for i in range(0, 4):
    for j in range(70, 80):
        FLUID_01[i][j] = 0
# obstáculo bajo-centro
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

# Con Dirichlet en la base el sistema no tiene modo nulo; no requerimos ancla.
anchor_k = None  # dejar preparado si en algún momento usas todo-Neumann

# ---------------------------
# 4) Ghosts (2º orden/1º orden) para BCs
# ---------------------------
def ghost_dirichlet(uP, u_b):
    # segundo orden: u_ghost = 2 u_b - u_P
    return 2.0*u_b - uP

def ghost_neumann_x(uP, gx, lado):
    # lado en {'left','right'}
    # (uP - uGhost)/dx = gx para left  -> uGhost = uP - gx*dx
    # (uGhost - uP)/dx = gx para right -> uGhost = uP + gx*dx
    if lado == 'left':
        return uP - gx*dx
    elif lado == 'right':
        return uP + gx*dx
    else:
        raise ValueError("lado inválido para ghost_neumann_x")

def ghost_neumann_y(uP, gy, lado):
    # lado en {'top','bottom'}
    # (uGhost - uP)/dy = gy -> uGhost = uP + gy*dy
    # usamos misma fórmula para consistencia; el signo lo define gy
    return uP + gy*dy

# valores de borde (según figura)
LEFT_gx  = 0.0      # du/dx=0 en inlet
RIGHT_gx = 0.0      # du/dx=0 en outlet
TOP_gy   = V0_TOP   # du/dy=V0_TOP en techo
BOT_u    = 0.0      # u=0 en base

def get_u_at(i, j, u_vec):
    """u(i,j) considerando fluido/obstáculo/bordes con ghosts.
    Nota: dentro del dominio, si es obstáculo, devuelve 0.0 (valor de pared).
          Los efectos de pared en stencil se tratan con ghost explícito en neighbor_with_obstacle_ghost.
    """
    # Dentro del dominio
    if 0 <= i < Ny and 0 <= j < Nx:
        if fluid_mask[i, j]:
            return u_vec[ij_to_k[i, j]]
        else:
            return 0.0  # obstáculo: valor en pared (u_wall=0)

    # Fuera del dominio: aplicar BCs del lado correspondiente
    if j < 0:  # izquierda (x=0): Neumann du/dx=0
        iP, jP = i, 0
        uP = 0.0 if not (0 <= iP < Ny and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_x(uP, LEFT_gx, 'left')

    if j >= Nx:  # derecha (x=Nx-1): Neumann du/dx=0
        iP, jP = i, Nx-1
        uP = 0.0 if not (0 <= iP < Ny and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_x(uP, RIGHT_gx, 'right')

    if i < 0:  # techo (y=0): Neumann du/dy=V0_TOP
        iP, jP = 0, j
        uP = 0.0 if not (0 <= jP < Nx and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_neumann_y(uP, TOP_gy, 'top')

    if i >= Ny:  # base (y=Ny-1): Dirichlet u=0
        iP, jP = Ny-1, j
        uP = 0.0 if not (0 <= jP < Nx and fluid_mask[iP, jP]) else u_vec[ij_to_k[iP, jP]]
        return ghost_dirichlet(uP, BOT_u)

    raise RuntimeError("Índice fantasma inesperado.")

def neighbor_with_obstacle_ghost(i, j, u_vec, direction):
    """Devuelve el valor vecino en la dirección indicada,
       aplicando ghost Dirichlet (u_wall=0) si hay obstáculo adyacente.
       direction in {'W','E','N','S'}.
    """
    uP = get_u_at(i, j, u_vec)
    di, dj = {'W':(0,-1),'E':(0,1),'N':(-1,0),'S':(1,0)}[direction]
    ii, jj = i+di, j+dj
    # Si el vecino está dentro y es obstáculo -> ghost en la cara
    if 0 <= ii < Ny and 0 <= jj < Nx and is_obstacle[ii, jj]:
        # pared interna Dirichlet u_wall=0 -> uGhost = 2*u_wall - uP = -uP
        return -uP
    # Si está fuera -> BC externa correspondiente
    return get_u_at(ii, jj, u_vec)

def convective_velocities(u_vec):
    """Campos de velocidad convectiva (para el término advectivo)."""
    if VX_MODE.lower() == "burgers":
        Vx = np.zeros((Ny, Nx), dtype=float)
        for i, j in k_to_ij:
            Vx[i, j] = u_vec[ij_to_k[i, j]]
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)
    else:
        Vx = np.full((Ny, Nx), VX_CONST, dtype=float)
        Vy = np.full((Ny, Nx), VY_CONST, dtype=float)

    # Por consistencia física, velocidad nula en sólidos
    Vx[is_obstacle] = 0.0
    Vy[is_obstacle] = 0.0
    return Vx, Vy

# ---------------------------
# 5) Residual F(u)
# ---------------------------
def build_residual(u_vec):
    Vx, Vy = convective_velocities(u_vec)
    F = np.zeros(nunk, dtype=float)

    for kk, (i, j) in enumerate(k_to_ij):
        # stencil con paredes internas como ghost
        uP = get_u_at(i, j, u_vec)
        uW = neighbor_with_obstacle_ghost(i, j, u_vec, 'W')
        uE = neighbor_with_obstacle_ghost(i, j, u_vec, 'E')
        uN = neighbor_with_obstacle_ghost(i, j, u_vec, 'N')
        uS = neighbor_with_obstacle_ghost(i, j, u_vec, 'S')

        # Laplaciano (d2u/dx2 + d2u/dy2) centrado
        lap = (uE - 2.0*uP + uW)/(dx*dx) + (uN - 2.0*uP + uS)/(dy*dy)

        # Convección con conmutación central/upwind por Péclet local
        vx = Vx[i, j]; vy = Vy[i, j]

        # X
        if abs(vx)*dx/nu > PE_UPWIND:
            du_dx = (uP - uW)/dx if vx >= 0.0 else (uE - uP)/dx
        else:
            du_dx = (uE - uW)/(2.0*dx)

        # Y
        if abs(vy)*dy/nu > PE_UPWIND:
            du_dy = (uP - uN)/dy if vy >= 0.0 else (uS - uP)/dy
        else:
            du_dy = (uS - uN)/(2.0*dy)

        conv = vx*du_dx + vy*du_dy
        F[kk] = -conv + nu*lap

    return F

# ---------------------------
# 6) Jacobiano numérico (denso)
# ---------------------------
def build_jacobian_numeric(u_vec, eps_base=1e-7):
    F0 = build_residual(u_vec)
    J = np.zeros((nunk, nunk), dtype=float)
    # escala global del problema para el incremento
    scale = max(1.0, np.linalg.norm(u_vec, ord=np.inf))
    du_mag = eps_base * scale

    for q in range(nunk):
        u_pert = u_vec.copy()
        # asegura incremento mínimo decente incluso si u[q] ~ 0
        du = max(du_mag, eps_base)
        u_pert[q] += du
        Fq = build_residual(u_pert)
        J[:, q] = (Fq - F0) / du
    return J

# ---------------------------
# 7) Newton–Raphson (con backtracking robusto)
# ---------------------------
def newton_solve(u0, max_iter=30, tol=1e-10, damping=True, verbose=True):
    u = u0.copy()
    for it in range(max_iter):
        F = build_residual(u)
        res = np.linalg.norm(F, ord=np.inf)
        if verbose:
            print(f"Iter {it:02d}  ||F||_inf = {res:.3e}")
        if res < tol:
            break

        J = build_jacobian_numeric(u)
        rhs = -F

        # Resolver lineal (denso)
        if SCIPY_OK:
            delta = sla.solve(J, rhs, assume_a='gen', check_finite=False)
        else:
            delta = np.linalg.solve(J, rhs)

        if damping:
            # Backtracking tipo Armijo con fallback al mejor intento
            alpha = 1.0
            accepted = False
            best_res = res
            best_u = u
            for _ in range(12):
                u_try = u + alpha*delta
                r_try = np.linalg.norm(build_residual(u_try), ord=np.inf)
                if r_try < best_res:
                    best_res, best_u = r_try, u_try
                # criterio Armijo suave
                if r_try <= (1 - 1e-3*alpha) * res:
                    u = u_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                # paso conservador al mejor intento probado
                u = best_u
        else:
            u = u + delta
    return u

# ---------------------------
# 8) Ejecutar solver
# ---------------------------
u0 = np.zeros(nunk, dtype=float)  # arranque suave compatible con u=0 en base
u  = newton_solve(u0, max_iter=25, tol=1e-10, damping=True, verbose=True)

# ---------------------------
# 9) Reconstrucción y gráficos
# ---------------------------
U = np.full((Ny, Nx), np.nan, dtype=float)
for kk, (i, j) in enumerate(k_to_ij):
    U[i, j] = u[kk]

# Colorear NaN (obstáculos) en negro para distinguirlos
import matplotlib as mpl
cmap = mpl.cm.viridis.copy()
cmap.set_bad('k')

plt.figure(figsize=(12, 3.2))
im = plt.imshow(U, origin="upper", aspect="auto", cmap=cmap)
plt.title("Convección–difusión con BCs del esquema (NaN = obstáculo)")
plt.xlabel("x (col)"); plt.ylabel("y (fila)")
plt.colorbar(im, label="u")
plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 1.8))
plt.imshow(fluid_mask, origin="upper", aspect="auto", cmap="gray")
plt.title("Máscara de fluido (blanco) vs obstáculo (negro)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout(); plt.show()

# (opcional) imprimir máscara para depurar
print(fluid_mask)
