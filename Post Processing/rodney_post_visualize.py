import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

kAx = 0.0473
kAy = 0.0423
kBx = 0.0483
kBy = 0.0483

dAx = 0.0181
dAy = 0.0181
dBx = 0.0706
dBy = 0.0611

cAx = -0.0296
cAy = -0.0785
cBx = 0.0174
cBy = -0.0863

csv_path = r'06_11_test_3.csv'
data = pd.read_csv(csv_path, skiprows=11)
time_sec = data['Time'].to_numpy()
strain_ax = data['Strain Ax'].to_numpy()
strain_bx = data['Strain Bx'].to_numpy()
strain_ay = data['Strain Ay'].to_numpy()
strain_by = data['Strain By'].to_numpy()
strain_ax = strain_ax - np.mean(strain_ax)
strain_ay = strain_ay - np.mean(strain_ay)
strain_bx = strain_bx - np.mean(strain_bx)
strain_by = strain_by - np.mean(strain_by)


strain_ax_smooth = savgol_filter(strain_ax, 25, 2)
strain_bx_smooth = savgol_filter(strain_bx, 25, 2)
strain_ay_smooth = savgol_filter(strain_ay, 25, 2)
strain_by_smooth = savgol_filter(strain_by, 25, 2)

# force_x = (kBx*(strain_ax_smooth - cAx) - kAx*(strain_bx_smooth - cBx))/(kAx*kBx*(dBx - dAx))
# force_y = (kBy*(strain_ay_smooth - cAy) - kAy*(strain_by_smooth - cBy))/(kAy*kBy*(dBy - dAy))
# force = np.sqrt(force_x**2 + force_y**2)



# base_strain_ax = strain_ax_smooth[(time_sec > 10.5) & (time_sec < 17)]
# base_val = np.mean(base_strain_ax)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))

axs[0, 0].plot(time_sec, strain_ax, linewidth=0.3)
axs[0, 0].plot(time_sec, strain_ax_smooth, label='Ax')
axs[0, 0].legend()

axs[0, 1].plot(time_sec, strain_ay, linewidth=0.3)
axs[0, 1].plot(time_sec, strain_ay_smooth, label='Ay')
axs[0, 1].legend()

axs[1, 0].plot(time_sec, strain_bx, linewidth=0.3)
axs[1, 0].plot(time_sec, strain_bx_smooth, label='Bx')
axs[1, 0].legend()

axs[1, 1].plot(time_sec, strain_by, linewidth=0.3)
axs[1, 1].plot(time_sec, strain_by_smooth, label='By')
axs[1, 1].legend()

plt.tight_layout()
plt.show()