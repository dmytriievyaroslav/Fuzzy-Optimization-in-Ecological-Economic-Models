import numpy as np
from scipy.optimize import linprog
import skfuzzy as fuzz
import pandas as pd
import tkinter as tk
from pandastable import Table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

C11_domain = np.arange(0.1, 0.8, 0.05)
C12_domain = np.arange(0.1, 0.8, 0.05)
C2_domain = np.arange(0.1, 0.8, 0.05)

C11_values = fuzz.trapmf(C11_domain, [0.3, 0.4, 0.6, 0.85])
C12_values = fuzz.trapmf(C12_domain, [0.35, 0.45, 0.7, 0.95])
C2_values = fuzz.trapmf(C2_domain, [0.4, 0.5, 0.6, 0.8])

data_list = []
x11_values = []
x12_values = []
x2_values = []

for i in range(len(C11_domain)):
    for j in range(len(C12_domain)):
        for k in range(len(C2_domain)):

            C11 = C11_domain[i]
            C12 = C12_domain[j]
            C2 = C2_domain[k]

            c_coeff = [-(C11_domain[i] * (1 - A11[0][0]) - C12_domain[j] * A11[1][0] - C2_domain[k]*A21[0][0]),
                        -(-C11_domain[i]*A11[0][1]+C12_domain[j]*(1-A11[1][1])-C2_domain[k]*A21[0][1]),
                        -(-C11_domain[i]*A12[0]-C12_domain[j]*A12[1]-C2_domain[k]*(A22-1))]

            A_ub = [[B11[0], B12[0], B2[0]],
                    [-1 + A11[0][0], A11[0][1],  A12[0]],
                    [A11[1][0], -1 + A11[1][1], A12[1]],
                    [-A21[0][0], -A21[0][1], -A22+1]]  
                      
            b_vector = [b2[0], 0, 0, 0]

            x_bounds = [(0, None), (0, None), (0, None)] 

            res = linprog(c_coeff, A_ub=A_ub, b_ub=b_vector, bounds=x_bounds, method='highs')

            y11=(1-A11[0][0])*res.x[0]-A11[0][1]*res.x[1]-A12[0]*res.x[2]
            y12=-A11[1][0]*res.x[0]+(1-A11[1][1])*res.x[1]-A12[1]*res.x[2]
            y2=A21[0][0]*res.x[0]+A21[0][1]*res.x[1]+(A22-1)*res.x[2] 
            b=B11[0]*res.x[0]+B12[0]*res.x[1]+B2[0]*res.x[2]
            
            if res.success and y11>=0 and y12>=0 and y2>=0 and b>=b2[0]:
                y22=A21[0][0]*res.x[0]+A21[0][1]*res.x[1]+(A22-1)*res.x[2] 
                data_list.append({"C11": C11, 
                                    "C12": C12, 
                                    "C2": C2,
                                    "Optimal Solution": -res.fun, 
                                    "Chance": "{:.2f}".format(min(C11_values[i], C12_values[j], C2_values[k])),
                                    "x_values": res.x})
                
                x11_values.append(res.x[0])
                x12_values.append(res.x[1])
                x2_values.append(res.x[2])

df = pd.DataFrame(data_list)

root = tk.Tk()
frame = tk.Frame(root)
frame.pack(fill='both', expand=True)
pt = Table(frame, dataframe=df, showtoolbar=True, showstatusbar=True)
pt.show()

fig, ax = plt.subplots(nrows=3, figsize=(8, 6))

ax[0].plot(np.arange(0.1, 0.8, 0.05), C11_values, 'b', linewidth=1.5, label='C11')
ax[0].set_title('C11 Fuzzy Membership')
ax[0].legend()

ax[1].plot(np.arange(0.1, 0.8, 0.05), C12_values, 'r', linewidth=1.5, label='C12')
ax[1].set_title('C12 Fuzzy Membership')
ax[1].legend()

ax[2].plot(np.arange(0.1, 0.8, 0.05), C2_values, 'g', linewidth=1.5, label='C2')
ax[2].set_title('C2 Fuzzy Membership')
ax[2].legend()

for ax in ax.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x11_values, x12_values, x2_values, c='r', marker='o')

x11_grid, x12_grid = np.meshgrid(np.linspace(min(x11_values), max(x11_values), len(x11_values)),
                                 np.linspace(min(x12_values), max(x12_values), len(x12_values)))

x2_grid1 = ((1 - A11[0][0]) * x11_grid - A11[0][1] * x12_grid) / A12[0]
x2_grid2 = (-A11[1][0] * x11_grid + (1 - A11[1][1]) * x12_grid) / A12[1]
x2_grid3 = (A21[0][0] * x11_grid + A21[0][1] * x12_grid + (A22 - 1) * x2_grid2)
x2_grid4 = (b2[0] - B11[0]*x11_grid - B12[0]*x12_grid) / B2[0]

x2_grid1[x2_grid1 > 300] = np.nan
x2_grid2[x2_grid2 > 300] = np.nan
x2_grid3[x2_grid3 > 300] = np.nan
x2_grid4[x2_grid4 > 300] = np.nan

x2_grid1[0 > x2_grid1] = np.nan
x2_grid2[0 > x2_grid2] = np.nan
x2_grid3[0 > x2_grid3] = np.nan
x2_grid4[0 > x2_grid4] = np.nan

ax.plot_surface(x11_grid, x12_grid, x2_grid1, color='b', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid2, color='y', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid3, color='g', alpha=0.5, rstride=100, cstride=100)
ax.plot_surface(x11_grid, x12_grid, x2_grid4, color='r', alpha=0.5, rstride=100, cstride=100)

ax.set_xlabel('X11')
ax.set_ylabel('X12')
ax.set_zlabel('X2')
ax.set_zlim([0, 300])
plt.xlim(0, 300)
plt.ylim(0, 200)
plt.show()

root.mainloop()
