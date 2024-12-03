import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import time
import matplotlib.pyplot as plt

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


# Splitting inputs and outputs
inputs = joint_df[['x_shape_a', 'y_shape_a', 'x_shape_b', 'y_shape_b']].values
outputs = joint_df[['x_shape_result', 'y_shape_result']].values


# GAN components
def create_generator(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    return model

def create_discriminator(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan_model = Model(gan_input, gan_output)
    return gan_model

# Training and evaluating the GAN with K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

generator = None

start_time = time.time()
for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    X_train, X_test = inputs[train_idx], inputs[test_idx]
    y_train, y_test = outputs[train_idx], outputs[test_idx]

    # Create generator and discriminator
    generator = create_generator(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    discriminator = create_discriminator(input_dim=y_train.shape[1])
    
    # Compile discriminator
    discriminator.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create and compile GAN
    gan = create_gan(generator, discriminator)
    gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

    # Training GAN
    batch_size = 16
    epochs = 1
    for epoch in range(epochs):
        # Train discriminator
        real_data = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size, X_train.shape[1]))
        fake_data = generator.predict(noise)
        combined_data = np.concatenate([real_data, fake_data])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        discriminator_loss = discriminator.train_on_batch(combined_data, labels)

        # Train generator via GAN
        noise = np.random.normal(0, 1, (batch_size, X_train.shape[1]))
        misleading_targets = np.ones((batch_size, 1))
        gan_loss = gan.train_on_batch(noise, misleading_targets)

    # Evaluate the generator on the test set
    predictions = generator.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    results.append({'fold': fold + 1, 'MAE': mae, 'MSE': mse})
    print(f"Fold {fold + 1}: MAE={mae:.4f}, MSE={mse:.4f}")
end_time = time.time()

# Display cross-validation results
cv_results = pd.DataFrame(results)
print("\nCross-validation results:")
print(cv_results)
print("Mean MSE:", cv_results['MSE'].mean())
print("Standard Deviation in MSE:", cv_results['MSE'].std())


#################################################################################################
# INFERENCE
#################################################################################################


gan = None

gan = generator


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
