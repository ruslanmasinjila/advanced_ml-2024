import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
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


# Inputs and outputs
X = joint_df[['x_shape_a', 'y_shape_a', 'x_shape_b', 'y_shape_b']].values
y = joint_df[['x_shape_result', 'y_shape_result']].values

# Reshape inputs to fit CNN requirements
X_cnn = X.reshape(X.shape[0], 2, 2, 1)

# Model definition
def create_model():
    model = Sequential([
        InputLayer(input_shape=(2, 2, 1)),
        Conv2D(16, kernel_size=(2, 2), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2)  # Two outputs: x_shape_result and y_shape_result
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
    return model

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
cv_results = []

model = None

start_time = time.time()
for train_index, val_index in kf.split(X_cnn):
    print(f"Fold {fold}")
    X_train, X_val = X_cnn[train_index], X_cnn[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = create_model()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=8, verbose=1)
    
    val_loss, val_mse = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss}, Validation MSE: {val_mse}")
    cv_results.append((val_loss, val_mse))
    fold += 1
end_time = time.time()

# Average metrics over folds
avg_loss = np.mean([result[0] for result in cv_results])
avg_mse = np.mean([result[1] for result in cv_results])
std     = np.std([result[1] for result in cv_results])
print(f"Average Validation MSE for 5-fold cross validation: {avg_mse}")
print(f"Standard Deviation of  MSE for 5-fold cross validation: {std}")

model.save("CNN.h5")




#################################################################################
# INFERENCE
#################################################################################
cnn = None

cnn = load_model('CNN.h5')

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
    X_new = X_new.reshape(X_new.shape[0], 2, 2, 1)

    pred = cnn.predict(X_new)


    pred_df = pd.DataFrame(pred, columns=['x_shape_result', 'y_shape_result'])


    for col in pred_df.columns:
        mse = ((pred_df[col] - shape_result_df[col]) ** 2).mean()
        mse_inference.append(mse)
    
    actual_result.append(shape_result_df)
    predicted_result.append(pred_df)


print(f'Average MSE on 100 inferences: {np.mean(mse_inference)}')
