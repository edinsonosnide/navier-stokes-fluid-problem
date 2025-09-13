import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la elipse y del área
a, b = 5, 3
lim = 5                  # cuadrado [-5,5]x[-5,5]
area_cuadrado = (2*lim)**2  # 100

# Puntos aleatorios
numero_coordenadas_random = 500  # sube este número para más precisión
x_rand = np.random.uniform(-lim, lim, size=numero_coordenadas_random)
y_rand = np.random.uniform(-lim, lim, size=numero_coordenadas_random)
random_cordinates = np.column_stack((x_rand, y_rand))

# Clasificación: dentro de la elipse (x/a)^2 + (y/b)^2 <= 1
mask_dentro = (x_rand/a)**2 + (y_rand/b)**2 <= 1.0
puntos_dentro = np.sum(mask_dentro)
puntos_fuera = numero_coordenadas_random - puntos_dentro

# Estimación de pi vía Monte Carlo usando la elipse
p_hat = puntos_dentro / numero_coordenadas_random
pi_est = (area_cuadrado / (a*b)) * p_hat  # = (100/(a*b))*p_hat

# Error estándar e IC 95% (aprox) para pi_est
se_pi = (area_cuadrado / (a*b)) * np.sqrt(p_hat * (1 - p_hat) / numero_coordenadas_random)
ci_low, ci_high = pi_est - 1.96*se_pi, pi_est + 1.96*se_pi

# Curva de la elipse para dibujar
t = np.linspace(0, 2*np.pi, 400)
x_elip = a * np.cos(t)
y_elip = b * np.sin(t)

# Gráfica
fig, ax = plt.subplots(figsize=(6,6))

# Cuadrado de referencia
rect = plt.Rectangle((-lim, -lim), 2*lim, 2*lim, fill=False, linewidth=1.5)
ax.add_patch(rect)

# Elipse
ax.plot(x_elip, y_elip, linewidth=2, label=f"Elipse a={a}, b={b}")

# Puntos
ax.scatter(x_rand[mask_dentro], y_rand[mask_dentro], s=8, c='green', label="Dentro")
ax.scatter(x_rand[~mask_dentro], y_rand[~mask_dentro], s=8, c='red', marker='x', label="Fuera")

# Estética
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, linewidth=1); ax.axvline(0, linewidth=1)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="upper right")

# Título con resultados (sin imprimir en consola)
ax.set_title(
    f"π ≈ {pi_est:.6f}"
    f"  Dentro: {puntos_dentro}  Fuera: {puntos_fuera}  Total: {numero_coordenadas_random}"
)

plt.show()
