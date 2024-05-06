from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np

# Constants defining the linear optimization problem
A11 = 0.3
A12 = 0.4
A21 = 0.5
A22 = 0.6
C1 = 0.9
C2 = 0.8
B1 = 4
B2 = 4
b = 80
E1 = E2 = 1

# Coefficients for the objective function to be minimized
c = [-C1*(E1-A11)+C2*A21, C2*(E2-A22)-C1*A12]

# Coefficients for the inequality constraints
A_ub = [
    [B1, B2],
    [-E1 + A11, -A12],
    [-A21, -E2 + A22]
]
b_ub = [b, 0, 0]  # Right-hand side of inequality constraints

# Bounds for the variables x1 and x2
x0_bounds = (0, None)
x1_bounds = (0, None)

# Perform linear programming optimization
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[x0_bounds, x1_bounds], method='highs')

# Output the optimal value and corresponding values of x1 and x2
print('Optimal value:', -res.fun, '\nX1, X2:', res.x)

# Generate data for visualization
x1_values = np.linspace(0, 100, 400)
x2_values = np.linspace(0, 100, 400)

# Calculate values of x2 from x1 for each constraint line
constraint1 = (1 - A11) * x1_values / A12
constraint2 = A21 * x1_values / (1 - A22)
constraint3 = (b - B1 * x1_values) / B2

# Plotting the constraints and feasible region
plt.figure(figsize=(10,10))

plt.plot(x1_values, constraint1, label='(E1 - A11) * x1 - A12 * x2 >= 0')
plt.fill_between(x1_values, 0, constraint1, where=(x2_values<=constraint1), alpha=0.1, color='red')

plt.plot(x1_values, constraint2, label='A21 * x1 - (E2 - A22) * x2 >= 0')
plt.fill_between(x1_values, 0, constraint2, where=(x2_values<=constraint2), alpha=0.1, color='blue')

plt.plot(x1_values, constraint3, label='B1 * x1 + B2 * x2 <= b')
plt.fill_between(x1_values, constraint3, 0, alpha=0.1, color='green')

# Mark the optimal solution point on the plot
plt.plot(res.x[0], res.x[1], 'ro', label='Optimal solution')

# Set plot limits and labels
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.show()
