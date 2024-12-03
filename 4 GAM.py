import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pygam import GAM, s
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

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

# Variable to store mse scores for each fold
mse_scores = []

start_timeX = time.time()
# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_x[train_index], y_x[test_index]
    
    # Initialize the Generalized Additive Model
    gam = GAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature
    
    # Fit the model on the training data
    gam.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = gam.predict(X_test)
    
    # Compute Mean Squared Error for the fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Compute mse (square root of MSE)
    mse = np.sqrt(mse)
    mse_scores.append(mse)

end_timeX = time.time()

# Calculate the average mse across all folds
average_mse_y_x =   np.mean(mse_scores)
std_y_x         =   np.std(mse_scores)

print(f'Average mse across 5 folds on y_x: {average_mse_y_x}')
print(f'Standard Deviation in mse across 5 folds on y_x: {std_y_x}')

# Initialize the Generalized Additive Model
gamYx = GAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature

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

# Variable to store mse scores for each fold
mse_scores = []

start_timeY = time.time()
# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_y[train_index], y_y[test_index]
    
    # Initialize the Generalized Additive Model
    gam = GAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature
    
    # Fit the model on the training data
    gam.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = gam.predict(X_test)
    
    # Compute Mean Squared Error for the fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Compute mse (square root of MSE)
    #mse = np.sqrt(mse)
    mse_scores.append(mse)

end_timeY = time.time()

# Calculate the average mse across all folds
average_mse_y_y =   np.mean(mse_scores)
std_y_y         =   np.std(mse_scores)

print(f'Average mse across 5 folds on y_y: {average_mse_y_y}')
print(f'Standard Deviation in mse across 5 folds on y_y: {std_y_y}')

# Initialize the Generalized Additive Model
gamYy = GAM(s(0) + s(1) + s(2) + s(3))  # Using smooth terms for each feature

# Train the model on the entire dataset
gamYy.fit(X, y_y)

# Save the trained model to a file using pickle
with open('gamYy.pkl', 'wb') as f:
    pickle.dump(gamYy, f)


###############################################################################################
# INFERENCE
###############################################################################################


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

    Yx_pred_new = gamYx.predict(X_new)
    Yy_pred_new = gamYy.predict(X_new)

    pred_df = pd.DataFrame()
    pred_df['x_shape_result'] = Yx_pred_new
    pred_df['y_shape_result'] = Yy_pred_new

    for col in pred_df.columns:
        mse = ((pred_df[col] - shape_result_df[col]) ** 2).mean()
        mse_inference.append(mse)
    
    actual_result.append(shape_result_df)
    predicted_result.append(pred_df)

print(f'Average mse on 100 inferences: {np.mean(mse_inference)}')


'''

index = 30

plt.figure(figsize=(8, 8))

# Plot the first set of points
plt.scatter(predicted_result[index]['x_shape_result'], predicted_result[index]['y_shape_result'], color='blue', s=10, label='Prediction')

# Add another set of points with different color
plt.scatter(actual_result[index]['x_shape_result'], actual_result[index]['y_shape_result'], color='red', s=10, label='Actual Result')

# Set equal aspect ratio and other properties
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Actual VS Prediction for GAM')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

'''
