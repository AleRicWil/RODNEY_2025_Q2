import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

csv_path = r'Impact_test.csv'
data = pd.read_csv(csv_path, skiprows=11)
csv_time_sec = data['Time (Microseconds)'].to_numpy()
csv_strain_ax = data['Strain Ax'].to_numpy()
csv_strain_bx = data['Strain Bx'].to_numpy()
csv_strain_ay = data['Strain Ay'].to_numpy()
csv_strain_by = data['Strain By'].to_numpy()

smooth_strain_ax = savgol_filter(csv_strain_ax, 10, 4)

# plt.scatter(csv_time_sec, csv_strain_ax, s=0.5)
plt.plot(csv_time_sec, csv_strain_ax, linewidth=0.5)
plt.plot(csv_time_sec, smooth_strain_ax, linewidth=1)

plt.show()