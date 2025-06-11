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

csv_path = r'06_11_test_1.csv'
data = pd.read_csv(csv_path, skiprows=11)
time_sec = data['Time (Microseconds)'].to_numpy()
strain_ax = data['Strain Ax'].to_numpy()
strain_bx = data['Strain Bx'].to_numpy()
strain_ay = data['Strain Ay'].to_numpy()
strain_by = data['Strain By'].to_numpy()
strain_ax_smooth = savgol_filter(strain_ax, 25, 2)
strain_bx_smooth = savgol_filter(strain_bx, 25, 2)
strain_ay_smooth = savgol_filter(strain_ay, 25, 2)
strain_by_smooth = savgol_filter(strain_by, 25, 2)

force_x = (kBx*(strain_ax_smooth - cAx) - kAx*(strain_bx_smooth - cBx))/(kAx*kBx*(dBx - dAx))
force_y = (kBy*(strain_ay_smooth - cAy) - kAy*(strain_by_smooth - cBy))/(kAy*kBy*(dBy - dAy))
force = np.sqrt(force_x**2 + force_y**2)



base_strain_ax = strain_ax_smooth[(time_sec > 10.5) & (time_sec < 17)]
base_val = np.mean(base_strain_ax)

plt.plot(time_sec, strain_ax, linewidth=0.5)
plt.plot(time_sec, strain_ax_smooth)
# plt.plot(time_sec, force)

# plt.scatter(time_sec, strain_ax, s=0.5)
# plt.plot(csv_time_sec, csv_strain_ax, linewidth=0.5)
# plt.plot(time_sec, smooth_strain_ax, c='orange', linewidth=1)
# plt.axhline(base_val, c='red')

# plt.xlim(0,12)
# plt.ylim(-0.04, 0.04)
plt.show()