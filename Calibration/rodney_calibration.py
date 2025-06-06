import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read Excel file
T = pd.read_excel("05_01_Testing_combined.xlsx", sheet_name="Data")

# Filter by location
Ax = T['Location'] == "Ax"
Ay = T['Location'] == "Ay"
Bx = T['Location'] == "Bx"
By = T['Location'] == "By"

# Define arrays
x = np.array([10, 12, 14, 16, 18, 10, 12, 14, 16, 18, 10, 12, 14, 16, 18, 10, 12, 14, 16, 18, 10, 12, 14, 16, 18])  # cm
mass = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0])  # kg
F = mass * 9.81  # N

# Calibration conditions
cal = [T['Mass_kg'] == m & (T['Distance_cm'] == d) for m, d in zip(mass, x)]

# Half Bridges
V_Ax_cal = np.array([T[Ax & c]['SampleAverage'].mean() for c in cal])
V_Bx_cal = np.array([T[Bx & c]['SampleAverage'].mean() for c in cal])
V_Ay_cal = np.array([T[Ay & c]['SampleAverage'].mean() for c in cal])
V_By_cal = np.array([T[By & c]['SampleAverage'].mean() for c in cal])

# Calibrate via Regression: V = kF(x-d)+c = kFx - kdF + c
x = x / 100  # Convert to meters
A = np.vstack([F * x, -F]).T

# Bridge Ax
lm_Ax = LinearRegression().fit(A, V_Ax_cal)
c_Ax, k_Ax = lm_Ax.intercept_, lm_Ax.coef_[0]
d_Ax = lm_Ax.coef_[1] / k_Ax

# Bridge Bx
lm_Bx = LinearRegression().fit(A, V_Bx_cal)
c_Bx, k_Bx = lm_Bx.intercept_, lm_Bx.coef_[0]
d_Bx = lm_Bx.coef_[1] / k_Bx

# Bridge Ay
lm_Ay = LinearRegression().fit(A, V_Ay_cal)
c_Ay, k_Ay = lm_Ay.intercept_, lm_Ay.coef_[0]
d_Ay = lm_Ay.coef_[1] / k_Ay

# Bridge By
lm_By = LinearRegression().fit(A, V_By_cal)
c_By, k_By = lm_By.intercept_, lm_By.coef_[0]
d_By = lm_By.coef_[1] / k_By

# Save Calibration Data
np.savez('CalibrationData_Rodney.npz', 
         c_Ax=c_Ax, k_Ax=k_Ax, d_Ax=d_Ax,
         c_Bx=c_Bx, k_Bx=k_Bx, d_Bx=d_Bx,
         c_Ay=c_Ay, k_Ay=k_Ay, d_Ay=d_Ay,
         c_By=c_By, k_By=k_By, d_By=d_By)

# Compare Regression Values
r2_cal_1 = [lm_Ax.score(A, V_Ax_cal), lm_Ay.score(A, V_Ay_cal)]
r2_cal_2 = [lm_Bx.score(A, V_Bx_cal), lm_By.score(A, V_By_cal)]
r2_cal_array = np.array([r2_cal_1, r2_cal_2])

print("R^2 Values:")
print(f"lm_Ax: {r2_cal_1[0]}")
print(f"lm_Ay: {r2_cal_1[1]}")
print(f"lm_Bx: {r2_cal_2[0]}")
print(f"lm_By: {r2_cal_2[1]}")

# Create table
r2_cal_table = pd.DataFrame(r2_cal_array, index=['1', '2'], columns=['x', 'y'])

# Plot R^2 values
plt.imshow(r2_cal_array, cmap='summer')
plt.colorbar()
text_strings = [f"{val:.8f}" for val in r2_cal_array.flatten()]
x_grid, y_grid = np.meshgrid(range(2), range(2))
for i, text in enumerate(text_strings):
    plt.text(x_grid.flatten()[i], y_grid.flatten()[i], text, ha='center', va='center')
plt.title('R^2 Values for Calibration')
plt.xticks([0, 1], ['X', 'Y'])
plt.yticks([0, 1], ['A', 'B'])
plt.xlabel('Bridge Configuration')
plt.ylabel('Bridge Location')
plt.savefig('rodney_r2values.png')
plt.close()