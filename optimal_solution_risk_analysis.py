import numpy as np
from scipy.optimize import linprog
import skfuzzy as fuzz  # For fuzzy logic operations
import pandas as pd
import tkinter as tk  # For creating GUI applications
from pandastable import Table  # For displaying tables in Tkinter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define matrices and vectors for the constraints and coefficients
A11 = np.array([[0.5, 0.3], [0.4, 0.3]])
A12 = np.array([[0.3], [0.2]])  
A21 = np.array([[0.3, 0.2]])
A22 = np.array([[0.2]])

b1 = np.array([50])
b2 = np.array([150])

B11 = np.array([0.6])
B12 = np.array([0.5])
B2 = np.array([0.4])
E1 = E2 = 1  # Scalars representing unity in economic equations

# Define domains for fuzzy membership functions for cost coefficients
C11_domain = np.arange(0.1, 0.8, 0.05)
C12_domain = np.arange(0.1, 0.8, 0.05)
C2_domain = np.arange(0.1, 0.8, 0.05)

# Generate fuzzy membership functions for cost coefficients
C11_values = fuzz.trapmf(C11_domain, [0.3, 0.4, 0.6, 0.85])
C12_values = fuzz.trapmf(C12_domain, [0.35, 0.45, 0.7, 0.95])
C2_values = fuzz.trapmf(C2_domain, [0.4, 0.5, 0.6, 0.8])

# Lists to collect data for further analysis and visualization
data_list = []
x11_values = []
x12_values = []
x2_values = []

# Nested loops to iterate over all possible combinations of fuzzy variables
for i in range(len(C11_domain)):
    for j in range(len(C12_domain)):
        for k in range(len(C2_domain)):
            C11 = C11_domain[i]
            C12 = C12_domain[j]
            C2 = C2_domain[k]

            # Formulate the objective function coefficients based on fuzzy inputs
            c_coeff = [-(C11 * (1 - A11[0][0]) - C12 * A11[1][0] - C2 * A21[0][0]),
                       -(-C11 * A11[0][1] + C12 * (1 - A11[1][1]) - C2 * A21[0][1]),
                       -(-C11 * A12[0] - C12 * A12[1] - C2 * (A22 - 1))]

            # Constraints matrix
            A_ub = [[B11[0], B12[0], B2[0]],
                    [-1 + A11[0][0], A11[0][1], A12[0]],
                    [A11[1][0], -1 + A11[1][1], A12[1]],
                    [-A21[0][0], -A21[0][1], -A22 + 1]]
            b_vector = [b2[0], 0, 0, 0]
            x_bounds = [(0, None), (0, None), (0, None)]

            # Perform the linear programming optimization
            res = linprog(c_coeff, A_ub=A_ub, b_ub=b_vector, bounds=x_bounds, method='highs')

            # Check if the solution is valid under all constraints
            if res.success and all(res.x >= 0):
                data_list.append({"C11": C11, "C12": C12, "C2": C2,
                                  "Optimal Solution": -res.fun, 
                                  "Chance": "{:.2f}".format(min(C11_values[i], C12_values[j], C2_values[k])),
                                  "x_values": res.x})
                x11_values.append(res.x[0])
                x12_values.append(res.x[1])
                x2_values.append(res.x[2])

# Convert collected data into a pandas DataFrame
df = pd.DataFrame(data_list)

# Set up a Tkinter window for displaying the data table
root = tk.Tk()
frame = tk.Frame(root)
frame.pack(fill='both', expand=True)
pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
pt.show()

# Plot fuzzy membership functions using Matplotlib
fig, ax = plt.subplots(nrows=3, figsize=(8, 6))
ax[0].plot(C11_domain, C11_values, 'b', linewidth=1.5, label='C11')
ax[1].plot(C12_domain, C12_values, 'r', linewidth=1.5, label='C12')
ax[2].plot(C2_domain, C2_values, 'g', linewidth=1.5, label='C2')
for a in ax:
    a.legend()
    a.grid(True)
plt.tight_layout()
plt.show()

# 3D visualization of feasible solutions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x11_values, x12_values, x2_values, c='r', marker='o')
plt.show()

# Start the Tkinter event loop
root.mainloop()
