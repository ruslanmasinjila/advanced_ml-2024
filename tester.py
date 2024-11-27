import numpy as np
from pygam import LinearGAM, s
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle

# Assuming `list_shape_a`, `list_shape_b`, and `list_shape_result` are already defined

list_shape_a = None
list_shape_b = None
list_shape_result = None

# Load data from pickle files
with open('list_shape_a.pkl', 'rb') as f:
    list_shape_a = pickle.load(f)

with open('list_shape_b.pkl', 'rb') as f:
    list_shape_b = pickle.load(f)

with open('list_shape_result.pkl', 'rb') as f:
    list_shape_result = pickle.load(f)

# Convert to numpy arrays
list_shape_a = np.array(list_shape_a)
list_shape_b = np.array(list_shape_b)
list_shape_result = np.array(list_shape_result)



import numpy as np

# Example array with shape (1000, 230, 2)
arr = np.random.rand(1000, 230, 2)

# Reshape the array to (230000, 2)
arr_reshaped = arr.reshape(-1, 2)

# Print the shape of the reshaped array
print(arr_reshaped)  # Should print (230000, 2)