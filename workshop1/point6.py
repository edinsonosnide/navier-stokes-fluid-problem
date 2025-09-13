# anim_hiperesfera_iterativa_vol4d_acum_teor.py
# Interactivo (rotar/zoom con el mouse). Intervalo: 2000 ms entre iteraciones.
# En cada iteración:
#   - Agrega puntos (100 inicial, +1000 por iteración)
#   - Muestra TODOS los puntos acumulados (proyección 3D en (x1,x2,x3))
#   - Estima por Monte Carlo el volumen 4D de la bola (||x||_2 <= R) usando TODOS los puntos acumulados
#     muestreados uniformemente en el hipercubo [-l/2, l/2]^4 (hipervolumen = l^4)
#   - Imprime el valor teórico V(B^4) = (pi^2/2) * R^4
#   - (Croquis) Dibuja en otra ventana un 2D (x1,x2) de los puntos NUEVOS de cada iteración,
#     diferenciando cuáles están dentro vs fuera de la bola 4D.
#   - (Nuevo) Dibuja en una tercera ventana la PRECISIÓN a través del tiempo:
#     error relativo acumulado = |V_est_acum - V_teor| / V_teor, contra el número total de simulaciones.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- Parámetros ----------------
dim = 4        # espacio R^4
R = 1.0        # radio de la bola/hiperesfera (para el test ||x||_2 <= R)
l = 2          # lado del hipercubo de muestreo en cada eje de R^4: [-l/2, l/2]
iters = 10
initial = 10   # puntos en la primera iteración
add_each = 500 # puntos añadidos en cada iteración
rng = np.random.default_rng(7)

# Dominio de muestreo: hipercubo [-l/2, l/2]^4  ->  hipervolumen = l^4
vol_hipercubo = (l) ** 4

# Volumen teórico de la bola 4D (B^4): V = (pi^2/2) * R^4
vol4_teorico = (np.pi ** 2 / 2.0) * (R ** 4)

# ---------------- Estado ----------------
grupos = []                         # puntos por iteración (para colorear distinto en 3D)
acumulado = np.empty((0, dim))      # todos los puntos acumulados

# Historial para la ventana de precisión
hist_N = []
hist_relerr = []

# ---------------- Figuras ----------------
plt.ion()

# Figura 3D interactiva (proyección x1, x2, x3 de TODOS los puntos acumulados)
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

# Croquis 2D (solo puntos NUEVOS de la iteración) en el plano (x1, x2)
fig2d, ax2d = plt.subplots()
theta = np.linspace(0, 2*np.pi, 512)
circ_x = R * np.cos(theta)
circ_y = R * np.sin(theta)

# Ventana adicional de precisión a través del tiempo
figAcc, axAcc = plt.subplots()
axAcc.set_title("Precisión vs número de simulaciones (acumuladas)")
axAcc.set_xlabel("Número total de puntos (N)")
axAcc.set_ylabel("Error relativo acumulado  |V_est - V_teor| / V_teor")
axAcc.grid(True)

def actualizar(frame):
    """Se ejecuta una vez por iteración."""
    global acumulado

    # 1) Generar nuevos puntos uniformes en [-l/2, l/2]^4
    n_new = initial if frame == 0 else add_each
    nuevos = rng.uniform(-l/2, l/2, size=(n_new, dim))

    # 2) Acumular y guardar por grupo
    grupos.append(nuevos)
    acumulado = np.vstack([acumulado, nuevos])

    # 3) Estimación del volumen 4D con TODOS los puntos acumulados
    norms_total = np.linalg.norm(acumulado, axis=1)
    inside_total_4d = int(np.sum(norms_total <= R))
    N_total = acumulado.shape[0]
    vol4_est_acum = vol_hipercubo * (inside_total_4d / N_total)

    # 3b) Actualizar precisión (error relativo acumulado)
    rel_err = abs(vol4_est_acum - vol4_teorico) / vol4_teorico
    hist_N.append(N_total)
    hist_relerr.append(rel_err)

    # -------- Gráfico 3D (proyección x1, x2, x3) --------
    ax3d.cla()
    for i, G in enumerate(grupos):
        ax3d.scatter(G[:, 0], G[:, 1], G[:, 2], s=12, label=f"Iter {i+1} (+{G.shape[0]})")

    ax3d.set_title(
        f"Iter {frame+1}/{iters} | Total={N_total} | Dentro 4D={inside_total_4d}\n"
        f"Vol est acum≈{vol4_est_acum:.6f}  |  Vol teórico={vol4_teorico:.6f}"
    )
    ax3d.set_xlabel("x1"); ax3d.set_ylabel("x2"); ax3d.set_zlabel("x3")
    ax3d.set_xlim(-l/2, l/2); ax3d.set_ylim(-l/2, l/2); ax3d.set_zlim(-l/2, l/2)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.grid(True)
    ax3d.legend(loc="upper left", fontsize=8, frameon=True)

    # -------- Croquis 2D (solo puntos NUEVOS de esta iteración) --------
    ax2d.cla()
    ax2d.plot(circ_x, circ_y, linewidth=1)  # circunferencia de radio R (referencia 2D)

    x1, x2 = nuevos[:, 0], nuevos[:, 1]
    inside_new = np.linalg.norm(nuevos, axis=1) <= R  # pertenencia REAL en 4D
    outside_new = ~inside_new

    ax2d.scatter(x1[inside_new], x2[inside_new], s=10, label=f"Dentro (nuevos): {inside_new.sum()}")
    ax2d.scatter(x1[outside_new], x2[outside_new], s=10, marker='x', label=f"Fuera (nuevos): {outside_new.sum()}")

    ax2d.set_title(f"Croquis (x1, x2) – Iter {frame+1}: nuevos={n_new}")
    ax2d.set_xlabel("x1"); ax2d.set_ylabel("x2")
    ax2d.set_aspect('equal', 'box')
    ax2d.set_xlim(-l/2, l/2); ax2d.set_ylim(-l/2, l/2)
    ax2d.grid(True)
    ax2d.legend(loc="upper right", fontsize=8, frameon=True)

    # -------- Ventana de precisión (error relativo acumulado vs N_total) --------
    axAcc.cla()
    axAcc.plot(hist_N, hist_relerr, linewidth=1)
    axAcc.set_title("Precisión vs número de simulaciones (acumuladas)")
    axAcc.set_xlabel("Número total de puntos (N)")
    axAcc.set_ylabel("Error relativo acumulado  |V_est - V_teor| / V_teor")
    axAcc.grid(True)

    # Salidas por consola
    print(
        f"Iteración {frame+1}: N_total={N_total}, dentro_4D={inside_total_4d}, "
        f"Vol_est_acum={vol4_est_acum:.6f}, Vol_teórico={vol4_teorico:.6f}, "
        f"Error_relativo={rel_err:.6%}"
    )

    return []

# ---------------- Animación ----------------
anim = FuncAnimation(
    fig3d, actualizar, frames=iters, interval=2000, blit=False, repeat=False
)

plt.show(block=True)
