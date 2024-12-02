import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys


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


list_shape_a = np.array(list_shape_a)
list_shape_b = np.array(list_shape_b)
list_shape_result = np.array(list_shape_result)

index = 700

shape_a = list_shape_a[index].reshape(-1, 2)
shape_a_df = pd.DataFrame(shape_a, columns=['x_shape_a','y_shape_a'])

shape_b = list_shape_b[index].reshape(-1, 2)
shape_b_df = pd.DataFrame(shape_b, columns=['x_shape_b','y_shape_b'])

shape_results = list_shape_result[index].reshape(-1, 2)
shape_result_df = pd.DataFrame(shape_results, columns=['x_shape_result','y_shape_result'])

# Add another set of points with different color
plt.scatter(shape_a_df['x_shape_a'], shape_a_df['y_shape_a'], color='blue', s=10, label = "shape_a")

# Add another set of points with different color
plt.scatter(shape_b_df['x_shape_b'], shape_b_df['y_shape_b'], color='green', s=10, label = "shape_b")

# Add another set of points with different color
plt.scatter(shape_result_df['x_shape_result'], shape_result_df['y_shape_result'], color='red', s=10, label = "result")

# Set equal aspect ratio and other properties
plt.gca().set_aspect('equal', adjustable='box')
plt.title(f'Interpolation of two Shapes')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()