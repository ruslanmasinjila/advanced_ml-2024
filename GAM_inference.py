import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pygam import GAM, s, LinearGAM
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

list_shape_a = list_shape_a[:-100]
list_shape_b = list_shape_b[:-100]
list_shape_result = list_shape_result[:-100]


for i in range(len(list_shape_a)):
    shape_a = np.array(list_shape_a).reshape(-1, 2)
    shape_b = np.array(list_shape_b).reshape(-1, 2)
    shape_result = np.array(list_shape_result).reshape(-1, 2)

    shape_a_df = pd.DataFrame(shape_a, columns=['x_shape_a','y_shape_a'])
    shape_b_df = pd.DataFrame(shape_b, columns=['x_shape_b','y_shape_b'])
    shape_result_df = pd.DataFrame(shape_result, columns=['x_shape_result','y_shape_result'])

    joint_df = pd.concat([shape_a_df, shape_b_df, shape_result_df], axis=1)
    print(joint_df)
    break
'''
print(len(list_shape_a),len(list_shape_b),len(list_shape_result))

# Convert to numpy arrays


'''




