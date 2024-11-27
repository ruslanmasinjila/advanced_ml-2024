import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Example dataset setup
# Replace this with loading your actual DataFrame
data = {
    'x1': np.random.rand(100),
    'y1': np.random.rand(100),
    'x2': np.random.rand(100),
    'y2': np.random.rand(100),
    'o1': np.random.rand(100),
    'O2': np.random.rand(100),
}
df = pd.DataFrame(data)

# Splitting inputs and outputs
inputs = df[['x1', 'y1', 'x2', 'y2']].values
outputs = df[['o1', 'O2']].values

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

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    X_train, X_test = inputs[train_idx], inputs[test_idx]
    y_train, y_test = outputs[train_idx], outputs[test_idx]

    # Create generator and discriminator
    generator = create_generator(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    discriminator = create_discriminator(input_dim=y_train.shape[1])
    
    # Compile discriminator
    discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create and compile GAN
    gan = create_gan(generator, discriminator)
    gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

    # Training GAN
    batch_size = 16
    epochs = 1000
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

# Display cross-validation results
cv_results = pd.DataFrame(results)
print("\nCross-validation results:")
print(cv_results)
print("\nMean MAE:", cv_results['MAE'].mean())
print("Mean MSE:", cv_results['MSE'].mean())
