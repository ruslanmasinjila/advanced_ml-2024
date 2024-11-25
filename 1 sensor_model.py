import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
r1, r2 = 1, 2          # Radial bounds
theta1, theta2 = 0, np.pi/2  # Angular bounds (in radians)

# Generate data points
# 50 points between r1 and r2 along theta1............................[1]
r_along_theta1 = np.linspace(r1, r2, 40)
theta_along_theta1 = np.full_like(r_along_theta1, theta1)

# 50 points between theta1 and theta2 along r2........................[2]
theta_along_r2 = np.linspace(theta1, theta2, 100)
r_along_r2 = np.full_like(theta_along_r2, r2)

# 50 points between r2 and r1 along theta2............................[3]
r_along_theta2 = np.linspace(r2, r1, 40)
theta_along_theta2 = np.full_like(r_along_theta2, theta2)

# 50 points between theta2 and theta1 along r1........................[4]
theta_along_r1 = np.linspace(theta2, theta1, 50)
r_along_r1 = np.full_like(theta_along_r1, r1)



# Combine all data points
r = np.concatenate([r_along_theta1, r_along_r2, r_along_theta2, r_along_r1 ])
theta = np.concatenate([theta_along_theta1, theta_along_r2, theta_along_theta2, theta_along_r1 ])

# Convert polar to Cartesian coordinates for visualization
x = r * np.cos(theta)
y = r * np.sin(theta)

base_sensor_model = pd.DataFrame()
base_sensor_model['x'] = x
base_sensor_model['y'] = y

base_sensor_model.to_csv('base_sensor_model.csv')

'''

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='blue', s=10, label='Data points')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Generated Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.legend()
plt.show()

'''