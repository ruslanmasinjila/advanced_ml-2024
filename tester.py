import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

augmneted_sensor_data = None
with open('augmented_sensor_data.pkl', 'rb') as f:
    augmneted_sensor_data = pickle.load(f)

x = augmneted_sensor_data[500]['x']
y = augmneted_sensor_data[500]['y']

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='blue', s=10, label='Data points')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Generated Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()