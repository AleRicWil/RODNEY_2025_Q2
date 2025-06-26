import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load calibration coefficients
cal_csv_path = r'AllInOne\calibration_history.csv'
cal_data = pd.read_csv(cal_csv_path)
latest_cal = cal_data.iloc[-1]  # Get the most recent calibration
k_A1 = latest_cal['k_A1']
k_B1 = latest_cal['k_B1']
k_A2 = latest_cal['k_A2']
k_B2 = latest_cal['k_B2']
d_A1 = latest_cal['d_A1']
d_B1 = latest_cal['d_B1']
d_A2 = latest_cal['d_A2']
d_B2 = latest_cal['d_B2']
c_A1 = latest_cal['c_A1']
c_B1 = latest_cal['c_B1']
c_A2 = latest_cal['c_A2']
c_B2 = latest_cal['c_B2']

# Load strain data
csv_path = r'Raw Data\06_25_test_1.csv'
data = pd.read_csv(csv_path, skiprows=11)
time_sec = data['Time'].to_numpy()
strain_a1 = data['Strain A1'].to_numpy()
strain_b1 = data['Strain B1'].to_numpy()
strain_a2 = data['Strain A2'].to_numpy()
strain_b2 = data['Strain B2'].to_numpy()
# strain_a1 = strain_a1 - np.mean(strain_a1)
# strain_b1 = strain_b1 - np.mean(strain_b1)
# strain_a2 = strain_a2 - np.mean(strain_a2)
# strain_b2 = strain_b2 - np.mean(strain_b2)

# Smooth strain data
strain_a1_smooth = savgol_filter(strain_a1, 25, 2)
strain_b1_smooth = savgol_filter(strain_b1, 25, 2)
strain_a2_smooth = savgol_filter(strain_a2, 25, 2)
strain_b2_smooth = savgol_filter(strain_b2, 25, 2)

# Calculate force
force = (k_A2 * (strain_a1_smooth - c_A1) - k_A1 * (strain_a2_smooth - c_A2)) / (k_A1 * k_A2 * (d_A2 - d_A1))
position = (k_A2*d_A2*(strain_a1_smooth - c_A1) - k_A1*d_A1*(strain_a2_smooth - c_A2)) / (k_A2*(strain_a1_smooth - c_A1) - k_A1*(strain_a2_smooth - c_A2))

# Plot strains (Figure 1)
fig1, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
axs[0, 0].plot(time_sec, strain_a1, linewidth=0.3)
axs[0, 0].plot(time_sec, strain_a1_smooth, label='A1')
axs[0, 0].legend()
axs[0, 1].plot(time_sec, strain_a2, linewidth=0.3)
axs[0, 1].plot(time_sec, strain_a2_smooth, label='A2')
axs[0, 1].legend()
axs[1, 0].plot(time_sec, strain_b1, linewidth=0.3)
axs[1, 0].plot(time_sec, strain_b1_smooth, label='B1')
axs[1, 0].legend()
axs[1, 1].plot(time_sec, strain_b2, linewidth=0.3)
axs[1, 1].plot(time_sec, strain_b2_smooth, label='B2')
axs[1, 1].legend()
plt.tight_layout()

# Plot force and position (Figure 2)
fig2, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
ax[0].plot(time_sec, force, label='Force')
ax[0].set_ylabel('Force (N)')
ax[1].plot(time_sec, position*100, label='Position')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Position (cm)')
plt.tight_layout()

plt.show()