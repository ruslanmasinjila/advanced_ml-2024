
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pickle



# Function to interpolate between two polygons
def interpolate_shapes(shape_a, shape_b, weight_a=0.5):
    # Use shapely unary_union to merge the shapes and find a midpoint
    merged_shape = unary_union([shape_a, shape_b])

    # Create a series of points on the boundary of each polygon
    a_points = np.array(shape_a.exterior.xy).T
    b_points = np.array(shape_b.exterior.xy).T

    # Interpolate between points of the two shapes
    # Here, we interpolate each pair of points with respect to the size of the shapes
    interpolated_points = []

    # Interpolation weight based on the area size of the shapes
    smaller_shape = shape_a if shape_a.area <= shape_b.area else shape_b
    larger_shape  = shape_b if shape_b.area > shape_a.area else shape_a

    # Weight the interpolation based on the size difference
    weight = smaller_shape.area / (smaller_shape.area + larger_shape.area)

    num_points = min(len(a_points), len(b_points))  # Ensure equal number of points
    for i in range(num_points):
        # Linear interpolation between points of the two shapes
        if(smaller_shape == shape_a):
          interpolated_point = (1 - weight) * a_points[i] + weight * b_points[i]
        else:
          interpolated_point = weight * a_points[i] + (1-weight) * b_points[i]

        interpolated_points.append(interpolated_point)

    # Convert list of interpolated points back to a Polygon
    interpolated_points = np.array(interpolated_points)
    return Polygon(interpolated_points)


# Load the augmented Data of the Sensor
augmneted_sensor_data = None
with open('augmented_sensor_data.pkl', 'rb') as f:
    augmneted_sensor_data = pickle.load(f)


# Convert augmented points from dataframes to list of x,y coordinates
augmneted_sensor_data = [i.values for i in augmneted_sensor_data]

# Convert x,y coordinates to polygon
augmented_sensor_data = [Polygon(i) for i in augmneted_sensor_data]




# Divide the data into two equal parts
l = len(augmneted_sensor_data)

list_shape_a = augmneted_sensor_data[:int(l/2)]
list_shape_a = [Polygon(i) for i in list_shape_a]


list_shape_b = augmneted_sensor_data[int(l/2):]
list_shape_b = [Polygon(i) for i in list_shape_b]




# Interpolate the two shapes

list_shape_result = [
    interpolate_shapes(shape_a, shape_b) 
    for shape_a, shape_b in zip(list_shape_a, list_shape_b)
]

list_shape_a = [list(x.exterior.coords) for x in list_shape_a]
list_shape_b = [list(x.exterior.coords) for x in list_shape_b]
list_shape_result = [list(x.exterior.coords) for x in list_shape_result]


# Save the input and output shapes.
with open('list_shape_a.pkl', 'wb') as f:
    pickle.dump(list_shape_a, f)

with open('list_shape_b.pkl', 'wb') as f:
    pickle.dump(list_shape_b, f)

with open('list_shape_result.pkl', 'wb') as f:
    pickle.dump(list_shape_result, f)


'''
# Plot the results
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the first and second shapes
x_a, y_a = list_shape_a[0].exterior.xy
x_b, y_b = list_shape_b[0].exterior.xy
ax.fill(x_a, y_a, alpha=0.5, label="Shape A", color='red')
ax.fill(x_b, y_b, alpha=0.5, label="Shape B", color='blue')

# Plot the interpolated shape
x_res, y_res = list_shape_result[0].exterior.xy
ax.fill(x_res, y_res, alpha=0.5, label="Hybrid Shape", color='green')

# Set labels and title
ax.set_title("Fused Shape (Hybrid of A and B)")
ax.legend()

plt.show()
'''
