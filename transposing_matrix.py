import numpy as np

a = np.array([1, 2, 3])      # forma (3,)
print(a, a.shape)            # [1 2 3] (3,)

a_col = a.reshape(-1, 1)     # NUEVO array, forma (3,1)
print(a_col, a_col.shape)
# [[1]
#  [2]
#  [3]] (3, 1)
