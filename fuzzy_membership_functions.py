import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the technology and constraint matrices and vectors for the model
A11 = np.array([[0.5, 0.3], [0.4, 0.3]])
A12 = np.array([[0.3], [0.2]])  
A21 = np.array([[0.3, 0.2]])
A22 = np.array([[0.2]])

b1 = np.array([50])
b2 = np.array([150])

B11 = np.array([0.6])
B12 = np.array([0.5])
B2 = np.array([0.4])
E1 = E2 = 1

C11 = 0.5
C12 = 0.6
C2 = 0.55

# Define the coefficients for the objective function to be minimized
c_coeff = [-(C11 * (1 - A11[0][0]) - C12 * A11[1][0] - C2*A21[0][0]),
           -(-C11*A11[0][1]+C12*(1-A11[1][1])-C2*A21[0][1]),
           -(-C11*A12[0]-C12*A12[1]-C2*(A22-1))]

# Define the inequality constraints matrix and vector
A_ub = [[B11[0], B12[0], B2[0]],
        [-1 + A11[0][0], A11[0][1],  A12[0]],
        [A11[1][0], -1 + A11[1][1], A12[1]],
        [-A21[0][0], -A21[0][1], -A22+1]]  
b_vector = [b2[0], 0, 0, 0]
x_bounds = [(0, None), (0, None), (0, None)]  

# Solve the linear programming problem
res = linprog(c_coeff, A_ub=A_ub, b_ub=b_vector, bounds=x_bounds, method='highs')

# Calculate additional variables to ensure non-negative production
y11 = (1 - A11[0][0]) * res.x[0] - A11[0][1] * res.x[1] - A12[0] * res.x[2]
y12 = -A11[1][0] * res.x[0] + (1 - A11[1][1]) * res.x[1] - A12[1] * res.x[2]
y2 = A21[0][0] * res.x[0] + A21[0][1] * res.x[1] + (A22 - 1) * res.x[2] 
b = B11[0] * res.x[0] + B12[0] * res.x[1] + B2[0] * res.x[2]

# Check if the solution meets all constraints and print results
if res.success and y11 >= 0 and y12 >= 0 and y2 >= 0 and b >= b2[0]:
    print("Optimal Solution=", -res.fun, "x_values=", res.x)

# Visualize the feasible region and solution in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the solution point
ax.scatter(res.x[0], res.x[1], res.x[2], c='r', marker='o')

# Generate a mesh grid to plot surfaces representing constraints
x11_grid, x12_grid = np.meshgrid(np.linspace(0, 300, 300), np.linspace(0, 300, 300))
x2_grid1 = ((1 - A11[0][0]) * x11_grid - A11[0][1] * x12_grid) / A12[0]
x2_grid2 = (-A11[1][0] * x11_grid + (1 - A11[1][1]) * x12_grid) / A12[1]
x2_grid3 = (A21[0][0] * x11_grid + A21[0][1] * x12_grid + (A22 - 1) * x2_grid2)
x2_grid4 = (b2[0] - B11[0] * x11_grid - B12[0] * x12_grid) / B2[0]

# Handle grid values outside of the feasible region
x2_grid1[x2_grid1 > 200] = np.nan
x2_grid2[x2_grid2 > 200] = np.nan
x2_grid3[x2_grid3 > 200] = np.nan
x2_grid4[x2_grid4 > 200] = np.nan
x2_grid1[0 > x2_grid1] = np.nan
x2_grid2[0 > x2_grid2] = np.nan
x2_grid3[0 > x2_grid3] = np.nan
x2_grid4[0 > x2_grid4] = np.nan

# Plot the constraint surfaces
ax.plot_surface(x11_grid, x12_grid, x2_grid1, color='b', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid2, color='y', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid3, color='g', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid4, color='r', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('X11')
ax.set_ylabel('X12')
ax.set_zlabel('X2')
ax.set_zlim([0, 200])
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.show()
