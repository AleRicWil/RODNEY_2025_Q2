import numpy as np
import matplotlib.pyplot as plt

offset = np.array([3.4, 4.6, 5.9, 7.2, 8.5, 9.7, 11.0, 8.9, 11.4, 6.4, 6.4, 6.4])
height = np.array([88.6, 84.8, 80.3])
offset = offset - offset[1]
avg_stiff = np.array([27.15, 27.84, 27.76, 28.59, 29.0, 28.98, 29.69, 27.1, 28.75, 27.1, 27.42, 26.62])
avg_stiff2 = np.array([27.1, 27.42, 26.62])
err = np.array([4.73, 4.18, 2.2, 1.83, 1.88, 1.88, 2.4, 3.47, 3.84, 2.62, 1.81, 1.84])
err2 = np.array([2.62, 1.81, 1.84])
colors = ['blue']*7 + ['red']*5
colors2 = ['blue']*3

for x, y, e, c in zip(offset, avg_stiff, err, colors):
    plt.errorbar(x, y, yerr=e, fmt='o', capsize=5, ecolor='black', c=c)
plt.xlabel('Predeflection (cm)')
plt.ylabel(r'Average Stiffness of Row (N/$m^2$)')
plt.ylim(0, np.max(avg_stiff)*1.2)
# plt.show()

plt.figure()
for x, y, e, c in zip(height, avg_stiff2, err2, colors2):
    plt.errorbar(x, y, yerr=e, fmt='o', capsize=5, ecolor='black', c=c)
plt.xlabel('Sensor Beam Height (cm)')
plt.ylabel(r'Average Stiffness of Row (N/$m^2$)')
plt.ylim(0, np.max(avg_stiff2)*1.2)
plt.xlim(75, 95)
plt.show()