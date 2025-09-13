import numpy as np
import matplotlib.pyplot as plt

# Definir el intervalo
x = np.linspace(0, 1, 200)
y = x**2

# Definir número de coordenadas aleatorias
numero_coordenadas_random = 100

# Generar coordenadas aleatorias dentro del rectángulo [0,1] x [0,1]
x_rand = np.random.rand(numero_coordenadas_random)
y_rand = np.random.rand(numero_coordenadas_random)
random_cordinates = np.column_stack((x_rand, y_rand))

# Clasificación de puntos respecto a la curva
y_curve = x_rand**2
puntos_debajo = np.sum(y_rand <= y_curve)
puntos_arriba = np.sum(y_rand > y_curve)
puntos_totales = numero_coordenadas_random  # o puntos_debajo + puntos_arriba

# Áreas
area_real = 1/3
area_rectangulo = 1.0  # base=1, altura=1
area_montecarlo = (puntos_debajo / puntos_totales) * area_rectangulo

# Crear la figura
fig, ax = plt.subplots()

# Graficar la curva
ax.plot(x, y, label=r"$y = x^2$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Curva y = x² en [0,1] con puntos aleatorios")

# Graficar puntos debajo y arriba con diferente color
ax.scatter(x_rand[y_rand <= y_curve], y_rand[y_rand <= y_curve],
           color='green', marker='o', label="Debajo de la curva")
ax.scatter(x_rand[y_rand > y_curve], y_rand[y_rand > y_curve],
           color='red', marker='x', label="Arriba de la curva")

ax.legend()
ax.grid(True)

# Ajustar espacio para colocar el texto fuera del plot
fig.subplots_adjust(bottom=0.3)

# Línea 1: conteos
fig.text(0.5, 0.08,
         f"Puntos arriba: {puntos_arriba} | Puntos debajo: {puntos_debajo} | Total: {puntos_totales}",
         ha="center", fontsize=12, color="black")

# Línea 2: áreas (real y Monte Carlo)
fig.text(0.5, 0.04,
         f"Área real bajo la curva: {area_real:.6f} | Área Monte Carlo: {area_montecarlo:.6f}",
         ha="center", fontsize=12, color="black")

plt.show()
