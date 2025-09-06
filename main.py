import matplotlib.pyplot as plt
import numpy as np

# Grid dimensions
rows, cols = 40, 400
grid = np.zeros((rows, cols))

plt.figure(figsize=(20, 3))  # make it wide and not too tall
plt.imshow(grid, cmap="Greys", interpolation="none")

# Add grid lines
plt.xticks(np.arange(-0.5, cols, 1), [])
plt.yticks(np.arange(-0.5, rows, 1), [])
plt.grid(color='black', linewidth=0.2)

plt.show()
