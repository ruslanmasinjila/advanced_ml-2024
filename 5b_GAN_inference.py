import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


gan = None

gan = load_model('GAN.h5')

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

list_shape_a = np.array(list_shape_a[-100:])
list_shape_b = np.array(list_shape_b[-100:])
list_shape_result = np.array(list_shape_result[-100:])

mse_inference = []
actual_result = []
predicted_result = []



for i in range(len(list_shape_a)):
    shape_a = list_shape_a[i].reshape(-1, 2)
    shape_b = list_shape_b[i].reshape(-1, 2)
    shape_result = list_shape_result[i].reshape(-1, 2)

    shape_a_df = pd.DataFrame(shape_a, columns=['x_shape_a','y_shape_a'])
    shape_b_df = pd.DataFrame(shape_b, columns=['x_shape_b','y_shape_b'])
    shape_result_df = pd.DataFrame(shape_result, columns=['x_shape_result','y_shape_result'])

    joint_df = pd.concat([shape_a_df, shape_b_df, shape_result_df], axis=1)
    X_new = joint_df[['x_shape_a', 'y_shape_a', 'x_shape_b', 'y_shape_b']].values

    pred = gan.predict(X_new)


    pred_df = pd.DataFrame(pred, columns=['x_shape_result', 'y_shape_result'])


    for col in pred_df.columns:
        mse = ((pred_df[col] - shape_result_df[col]) ** 2).mean()
        mse_inference.append(mse)
    
    actual_result.append(shape_result_df)
    predicted_result.append(pred_df)


print(f'Average mse on 100 inferences: {np.mean(mse_inference)}')


index = 25

plt.figure(figsize=(8, 8))

# Plot the first set of points
plt.scatter(predicted_result[index]['x_shape_result'], predicted_result[index]['y_shape_result'], color='blue', s=10, label='Prediction')

# Add another set of points with different color
plt.scatter(actual_result[index]['x_shape_result'], actual_result[index]['y_shape_result'], color='red', s=10, label='actual_result')

# Set equal aspect ratio and other properties
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Actual VS Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()
