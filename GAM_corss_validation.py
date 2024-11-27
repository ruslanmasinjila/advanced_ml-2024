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

print(len(list_shape_a),len(list_shape_b),len(list_shape_result))

# Convert to numpy arrays
list_shape_a = np.array(list_shape_a).reshape(-1, 2)
list_shape_b = np.array(list_shape_b).reshape(-1, 2)
list_shape_result = np.array(list_shape_result).reshape(-1, 2)

list_shape_a_df = pd.DataFrame(list_shape_a, columns=['x_shape_a','y_shape_a'])
list_shape_b_df = pd.DataFrame(list_shape_b, columns=['x_shape_b','y_shape_b'])
list_shape_result_df = pd.DataFrame(list_shape_result, columns=['x_shape_result','y_shape_result'])

joint_df = pd.concat([list_shape_a_df, list_shape_b_df, list_shape_result_df], axis=1)



################################################################################################

X = joint_df[['x_shape_a', 'y_shape_a', 'x_shape_b', 'y_shape_b']].values
y_x = joint_df['x_shape_result'].values

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variable to store RMSE scores for each fold
rmse_scores = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_x[train_index], y_x[test_index]
    
    # Initialize the Generalized Additive Model
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature
    
    # Fit the model on the training data
    gam.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = gam.predict(X_test)
    
    # Compute Mean Squared Error for the fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Compute RMSE (square root of MSE)
    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)

# Calculate the average RMSE across all folds
average_rmse = sum(rmse_scores) / len(rmse_scores)

print(f'Average RMSE across 5 folds on y_x: {average_rmse}')

# Initialize the Generalized Additive Model
gamYx = LinearGAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature

# Train the model on the entire dataset
gamYx.fit(X, y_x)

# Save the trained model to a file using pickle
with open('gamYx.pkl', 'wb') as f:
    pickle.dump(gamYx, f)


################################################################################################

X   = joint_df[['x_shape_a', 'y_shape_a', 'x_shape_b', 'y_shape_b']].values
y_y = joint_df['y_shape_result'].values

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variable to store RMSE scores for each fold
rmse_scores = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_y[train_index], y_y[test_index]
    
    # Initialize the Generalized Additive Model
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature
    
    # Fit the model on the training data
    gam.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = gam.predict(X_test)
    
    # Compute Mean Squared Error for the fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Compute RMSE (square root of MSE)
    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)

# Calculate the average RMSE across all folds
average_rmse = sum(rmse_scores) / len(rmse_scores)

print(f'Average RMSE across 5 folds on y_y: {average_rmse}')

# Initialize the Generalized Additive Model
gamYy = LinearGAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature

# Train the model on the entire dataset
gamYy.fit(X, y_y)

# Save the trained model to a file using pickle
with open('gamYy.pkl', 'wb') as f:
    pickle.dump(gamYy, f)


###############################################################################################
