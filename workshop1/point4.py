import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path

# ===== Área de trabajo (10x10 centrada en 0,0) =====
lim = 5
area_cuadrado = (2*lim)**2  # 100

# ===== Figura irregular (polígono) =====
n_vertices = 12
semilla = 43                # Cambia/comenta para variar la figura
np.random.seed(semilla)

# Generar vértices: ángulos ordenados + radios aleatorios acotados
ang = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
r_min, r_max = 1.2, lim * 0.9
rad = np.random.uniform(r_min, r_max, n_vertices)

x = rad * np.cos(ang)
y = rad * np.sin(ang)
pts = np.column_stack((x, y))  # (n_vertices, 2)

# ===== Área real por método Shoelace =====
def area_shoelace(puntos: np.ndarray) -> float:
    x = puntos[:, 0]
    y = puntos[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

area_real = area_shoelace(pts)

# ===== Monte Carlo (mismas muestras para estimar y para dibujar) =====
numero_muestras = 500  # <-- AJUSTA AQUÍ
x_mc = np.random.uniform(-lim, lim, size=numero_muestras)
y_mc = np.random.uniform(-lim, lim, size=numero_muestras)
muestra_mc = np.column_stack((x_mc, y_mc))

poly_path = Path(pts)
dentro = poly_path.contains_points(muestra_mc, radius=1e-12)  # incluye borde
area_mc = area_cuadrado * (np.sum(dentro) / numero_muestras)

# Conteo
puntos_dentro = int(np.sum(dentro))
puntos_fuera = numero_muestras - puntos_dentro

# ===== Gráfica =====
fig, ax = plt.subplots(figsize=(6, 6))

# Cuadrado 10x10 (borde)
marco = Rectangle((-lim, -lim), 2*lim, 2*lim, fill=False, linewidth=1.8)
ax.add_patch(marco)

# Polígono irregular
poly = Polygon(pts, closed=True, facecolor='tab:blue', alpha=0.25,
               edgecolor='tab:blue', linewidth=2)
ax.add_patch(poly)

# Puntos de Monte Carlo (los mismos que estiman el área)
ax.scatter(x_mc[dentro], y_mc[dentro], s=8, c='green', label="Dentro")
ax.scatter(x_mc[~dentro], y_mc[~dentro], s=8, c='red', marker='x', label="Fuera")

# Estética
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, color='k', linewidth=1)
ax.axvline(0, color='k', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")

# Título con áreas, muestras y conteos
ax.set_title(
    f"Área real ≈ {area_real:.3f} u² | Área Monte Carlo ≈ {area_mc:.3f} u² | "
    f"Muestras = {numero_muestras} | Dentro: {puntos_dentro} | Fuera: {puntos_fuera}"
)

ax.legend(loc="upper right")
plt.show()
