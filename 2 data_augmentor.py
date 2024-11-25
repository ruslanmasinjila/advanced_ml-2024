import pandas as pd
import numpy as np
import random
import pickle


def random_rotation(x, y, angle):
    """Rotate coordinates by a given angle."""
    radians = np.radians(angle)
    x_rot = x * np.cos(radians) - y * np.sin(radians)
    y_rot = x * np.sin(radians) + y * np.cos(radians)
    return x_rot, y_rot

def random_resize(x, y, scale_x, scale_y):
    """Resize coordinates by random scaling factors."""
    return x * scale_x, y * scale_y

def random_translation(x, y, shift_x, shift_y):
    """Translate coordinates by random shifts."""
    return x + shift_x, y + shift_y

def random_skew(x, y, skew_x, skew_y):
    """Apply skew transformation."""
    x_skew = x + skew_x * y
    y_skew = y + skew_y * x
    return x_skew, y_skew

def augment_dataframe(df):
    """Apply random transformations to a dataframe."""
    x, y = df['x'].values, df['y'].values

    # Random transformations
    angle = random.uniform(0, 360)  # Random angle in degrees
    scale_x, scale_y = random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)  # Random scaling factors
    shift_x, shift_y = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)  # Random translation shifts
    skew_x, skew_y = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)  # Random skew factors

    # Apply transformations
    x, y = random_rotation(x, y, angle)
    x, y = random_resize(x, y, scale_x, scale_y)
    x, y = random_translation(x, y, shift_x, shift_y)
    x, y = random_skew(x, y, skew_x, skew_y)

    # Create a new dataframe
    augmented_df = pd.DataFrame({'x': x, 'y': y})
    return augmented_df

# Load your original dataframe
base_sensor_model = pd.read_csv('base_sensor_model.csv')

# Generate 2000 augmented dataframes
############################################################################################## change back to 2000
augmented_sensor_data = [augment_dataframe(base_sensor_model) for _ in range(2000)]

# Save the augmented sensor data
with open('augmented_sensor_data.pkl', 'wb') as f:
    pickle.dump(augmented_sensor_data, f)
