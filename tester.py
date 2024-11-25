import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

augmneted_sensor_data = None
with open('augmented_sensor_data.pkl', 'rb') as f:
    augmneted_sensor_data = pickle.load(f)



l = len(augmneted_sensor_data)
print(l)
shape_a = augmneted_sensor_data[:int(l/2)]
shape_b = augmneted_sensor_data[int(l/2):]

print(len(shape_a))
print(len(shape_b))
