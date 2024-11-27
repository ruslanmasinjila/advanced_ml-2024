import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pygam import GAM, s, LinearGAM
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


gamYx = None
gamYy = None

# Load the saved model using pickle
with open('gamYx.pkl', 'rb') as f:
    gamYx = pickle.load(f)

# Load the saved model using pickle
with open('gamYy.pkl', 'rb') as f:
    gamYy = pickle.load(f)

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

rmse_inference = []
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

    Yx_pred_new = gamYx.predict(X_new)
    Yy_pred_new = gamYy.predict(X_new)

    pred_df = pd.DataFrame()
    pred_df['x_shape_result'] = Yx_pred_new
    pred_df['y_shape_result'] = Yy_pred_new

    for col in pred_df.columns:
        rmse = np.sqrt(((pred_df[col] - shape_result_df[col]) ** 2).mean())
        rmse_inference.append(rmse)
    
    actual_result.append(shape_result_df)
    predicted_result.append(pred_df)

print(f'Average RMSE on 100 inferences: {np.mean(rmse_inference)}')


index = 20

plt.figure(figsize=(8, 8))

# Plot the first set of points
plt.scatter(predicted_result[index]['x_shape_result'], predicted_result[index]['y_shape_result'], color='blue', s=10, label='Prediction')

# Add another set of points with different color
plt.scatter(actual_result[index]['x_shape_result'], actual_result[index]['y_shape_result'], color='red', s=10, label='actual_result')

# Set equal aspect ratio and other properties
plt.gca().set_aspect('equal', adjustable='box')
plt.title('ACtual VS Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()
