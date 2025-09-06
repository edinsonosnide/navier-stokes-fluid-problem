import matplotlib.pyplot as plt

rows, cols = 4, 40

fig, ax = plt.subplots(figsize=(24, 3), dpi=200)  # más ancho y nítido
ax.axis('off')  # ocultar ejes

# Construir los textos de cada celda (fila, columna)
cell_text = [[f"{i},{j}" for j in range(cols)] for i in range(rows)]

# Crear la tabla
table = ax.table(
    cellText=cell_text,
    cellLoc='center',
    loc='center',
)

# Ajustar proporciones de celdas y fuente
table.scale(1, 1.6)              # alto de celdas
for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')  # bordes visibles
    cell.set_linewidth(0.5)

# Tamaño de letra: si se ve apretado, baja a 5 o sube a 7
for (i, j), cell in table.get_celld().items():
    cell.get_text().set_fontsize(6)

plt.tight_layout()
plt.show()
