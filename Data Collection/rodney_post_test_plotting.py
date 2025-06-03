### Make sure to adjust the date, test_num, height, and csv_path for each file ####

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

date = "10_28"
test_num = 4
 
height = .813 #meters ...the "a" value in compliant mechanisms equations

# calibration coefficients, from 06/17/24
kAx =  0.1265
kAy = 0.109
kBx = 0.1227
kBy = 0.0997

dAx = 0.0634
dAy = 0.0157
dBx = 0.015
dBy = 0.0647

# the file containing the dynamic test data
csv_path = r"C:\Users\chris\Documents\Crop Biomechanics\Stiffness Sensor Files\10_28_rodney_field_data\10_28_test_04.csv" 

data = pd.read_csv(csv_path, skiprows=1)

csv_time_sec = data['Time (Microseconds)'].to_numpy()
csv_strain_ax = data['Strain Ax'].to_numpy()
csv_strain_bx = data['Strain Bx'].to_numpy()
csv_strain_ay = data['Strain Ay'].to_numpy()
csv_strain_by = data['Strain By'].to_numpy()

count = -1
cAx_array = np.array([])
cBx_array = np.array([])
cAy_array = np.array([])
cBy_array = np.array([])

while True:
    count = count + 1
    if csv_time_sec[count] > 5:
        break

    cAx_array = np.append(cAx_array, csv_strain_ax[count])
    cBx_array = np.append(cBx_array, csv_strain_bx[count])
    cAy_array = np.append(cAy_array, csv_strain_ay[count])
    cBy_array = np.append(cBy_array, csv_strain_by[count])

cAx = np.average(cAx_array)
cBx = np.average(cBx_array)
cAy = np.average(cAy_array)
cBy = np.average(cBy_array)

time_filtered = np.array([])
strain_ax_filtered = np.array([])
strain_bx_filtered = np.array([])
strain_ay_filtered = np.array([])
strain_by_filtered = np.array([])

for i in range(len(csv_time_sec)):
    if len(csv_time_sec) > 2:
        
        if abs((csv_strain_ax[i] - cAx)/cAx) > .02 and \
            abs((csv_strain_bx[i] - cBx)/cBx) > .02 and \
            abs((csv_strain_ay[i] - cAy)/cAy) > .02 and \
            abs((csv_strain_by[i] - cBy)/cBy) > .02:
        
            time_filtered = np.append(time_filtered, csv_time_sec[i])
            strain_ax_filtered = np.append(strain_ax_filtered, csv_strain_ax[i])
            strain_bx_filtered = np.append(strain_bx_filtered, csv_strain_bx[i])
            strain_ay_filtered = np.append(strain_ay_filtered, csv_strain_ay[i])
            strain_by_filtered = np.append(strain_by_filtered, csv_strain_by[i])

#from scipy.ndimage import gaussian_filter1d

# Smooth the data with a Gaussian filter, adjusting sigma for the degree of smoothing
#strain_ax_filtered = gaussian_filter1d(strain_ax_filtered, sigma=20)
#strain_bx_filtered = gaussian_filter1d(strain_bx_filtered, sigma=20)
#strain_ay_filtered = gaussian_filter1d(strain_ay_filtered, sigma=20)
#strain_by_filtered = gaussian_filter1d(strain_by_filtered, sigma=20)


# Plot the force, position, and stiffness plots
force_x = (kBx*(strain_ax_filtered - cAx) - kAx*(strain_bx_filtered - cBx))/(kAx*kBx*(dBx - dAx))
force_y = (kBy*(strain_ay_filtered - cAy) - kAy*(strain_by_filtered - cBy))/(kAy*kBy*(dBy - dAy))
force = (force_x**2 + force_y**2)**(1/2)

stalk_starts = np.array([])
stalk_ends = np.array([])
switch = False

for i in range(10, len(force) - 10):
    if switch == False and (time_filtered[i] - time_filtered[i-1] > .1 or len(stalk_starts) == 0) \
        and time_filtered[i+1] - time_filtered[i] < .1:
        switch = True
        #stalk_starts = np.append(stalk_starts, i)
    if switch == True and (time_filtered[i+1] - time_filtered[i] > .1 or i == len(force) - 11):
        switch = False
        #stalk_ends = np.append(stalk_ends, i)

theta_0_deg = 45 - (np.arctan(force_y/force_x) * 180/np.pi)

theta_0 = theta_0_deg * np.pi/180

position_x = abs((kBx*dBx*(strain_ax_filtered-cAx) - kAx*dAx*(strain_bx_filtered-cBx))/(kBx*(strain_ax_filtered-cAx) - kAx*(strain_bx_filtered-cBx))) * 100
position_y = abs((kBy*dBy*(strain_ay_filtered-cAy) - kAy*dAy*(strain_by_filtered-cBy))/(kBy*(strain_ay_filtered-cAy) - kAy*(strain_by_filtered-cBy))) * 100

position = np.array([])
position_fit = np.array([])
for i in range(len(force_x)):
    if abs(force_x[i]) > abs(force_y[i]):
        position = np.append(position, position_x[i])
    
    if abs(force_y[i]) > abs(force_x[i]):
        position = np.append(position, position_y[i])

#phi = theta_0 + np.pi/2
#n = -1/np.tan(phi)
n_table = np.array([-5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
c_theta_table = np.array([1.1788, 1.1971, 1.2119, 1.2293, 1.2322, 1.2323, 1.2348, 1.2385, 1.243, 1.2467, 1.2492, 1.2511, 1.2534, 1.2548, 1.2557, 1.257, 1.2578])
c_theta = np.array([])
gamma = np.array([])
k_theta = np.array([])
    
c_theta = np.ones(len(theta_0))* 1.24
gamma = np.ones(len(theta_0)) * 0.85
k_theta = np.ones(len(theta_0)) * 2.65


#print(n)
capital_theta = theta_0 / c_theta

L = height/(1-gamma*(1-np.cos(capital_theta)))
b = gamma*L*np.sin(capital_theta)

force_x_prime = force*np.sin(np.pi/2-theta_0)
force_y_prime = force*np.cos(np.pi/2-theta_0)
stiffness = np.abs((force_x_prime*(height-L*(1-gamma)) + force_y_prime*b)*L / (capital_theta*gamma*k_theta)) # units of N*m^2

plot_time = np.array([])
plot_force_x = np.array([])
plot_force_y = np.array([])
plot_force = np.array([])
plot_position_x = np.array([])
plot_position_y = np.array([])
plot_position = np.array([])
plot_theta_0_deg = np.array([])
plot_stiffness = np.array([])
plot_capital_theta = np.array([])

for i in range(len(position_x)):
    if position_x[i] < 25 and position_x[i] > 10:
        plot_time = np.append(plot_time, time_filtered[i]) 
        plot_force_x = np.append(plot_force_x, force_x[i])
        plot_force_y = np.append(plot_force_y, force_y[i])
        plot_force = np.append(plot_force, force[i])
        plot_position_x = np.append(plot_position_x, position_x[i])
        plot_position_y = np.append(plot_position_y, position_y[i])
        plot_position = np.append(plot_position, position[i])
        plot_theta_0_deg = np.append(plot_theta_0_deg, theta_0_deg[i])
        plot_stiffness = np.append(plot_stiffness, stiffness[i])
        plot_capital_theta = np.append(plot_capital_theta, capital_theta[i])
        
stiffness_sum_1 = 0
stiffness_num_1 = 0
stiffness_sum_2 = 0
stiffness_num_2 = 0
stiffness_sum_3 = 0
stiffness_num_3 = 0
stiffness_sum_4 = 0
stiffness_num_4 = 0
stiffness_sum_5 = 0
stiffness_num_5 = 0
stiffness_sum_6 = 0
stiffness_num_6 = 0

# adjust the values in the if-statement conditions below to change which times correspond with which stalks.

for i in range(len(plot_capital_theta)):
    if plot_time[i] > 13.685 and plot_time[i] < 14.7364 and plot_stiffness[i] < 25:
        #print(f'filtered capital_theta[i]: {plot_capital_theta[i]}')
        #print(f'stiffness: {plot_stiffness[i]}')
        stiffness_sum_1 = stiffness_sum_1 + plot_stiffness[i]
        stiffness_num_1 = stiffness_num_1 + 1
    elif plot_time[i] > 16.447 and plot_time[i] < 16.8707 and plot_stiffness[i] < 35:
        stiffness_sum_2 = stiffness_sum_2 + plot_stiffness[i]
        stiffness_num_2 = stiffness_num_2 + 1
    elif plot_time[i] > 18.3075 and plot_time[i] < 19.4355 and plot_stiffness[i] < 25:
        stiffness_sum_3 = stiffness_sum_3 + plot_stiffness[i]
        stiffness_num_3 = stiffness_num_3 + 1
    elif plot_time[i] > 20.5123 and plot_time[i] < 21.1202 and plot_stiffness[i] < 25:
        stiffness_sum_4 = stiffness_sum_4 + plot_stiffness[i]
        stiffness_num_4 = stiffness_num_4 + 1
    elif plot_time[i] > 22.8091 and plot_time[i] < 23.9099 and plot_stiffness[i] < 25:
        stiffness_sum_5 = stiffness_sum_5 + plot_stiffness[i]
        stiffness_num_5 = stiffness_num_5 + 1
    elif plot_time[i] > 24.7386 and plot_time[i] < 25.4192 and plot_stiffness[i] < 25:
        stiffness_sum_6 = stiffness_sum_6 + plot_stiffness[i]
        stiffness_num_6 = stiffness_num_6 + 1

print(stiffness_num_1)

plt.plot(csv_time_sec, csv_strain_ax)
plt.show()

stiffness_1 = stiffness_sum_1/stiffness_num_1
print(f'stiffness average 1: {stiffness_1}')

stiffness_2 = stiffness_sum_2/stiffness_num_2
print(f'stiffness average 2: {stiffness_2}')

stiffness_3 = stiffness_sum_3/stiffness_num_3
print(f'stiffness average 3: {stiffness_3}')

stiffness_4 = stiffness_sum_4/stiffness_num_4
print(f'stiffness average 4: {stiffness_4}')

stiffness_5 = stiffness_sum_5/stiffness_num_5
print(f'stiffness average 5: {stiffness_5}')

stiffness_6 = stiffness_sum_6/stiffness_num_6
print(f'stiffness average 6: {stiffness_6}')

plt.figure(1)
plt.plot(plot_time, plot_position_x, 'r', label='X-Position')
plt.plot(plot_time, plot_position_y, 'b', label='Y-Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.legend()
plt.savefig(f'post_processing_positions_vs_time_{date}_test_{test_num}')

plt.figure(2)
plt.plot(plot_time, plot_force_x, 'r', label='X-Axis Force')
plt.plot(plot_time, plot_force_y, 'b', label='Y-Axis Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.savefig(f'post_processing_forces_vs_time_{date}_test_{test_num}')

plt.figure(3)
plt.plot(plot_time, plot_position)
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.savefig(f'post_processing_position_vs_time_{date}_test_{test_num}')

plt.figure(4)
plt.plot(plot_time, plot_force)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.savefig(f'post_processing_force_vs_time_{date}_test_{test_num}')

plt.figure(5)
plt.plot(plot_time, plot_stiffness, '.')
plt.xlabel('Time (s)')
plt.ylabel('Stiffness (N*m^2)')
plt.ylim(0, 50)
plt.savefig(f'post_processing_stiffness_vs_time_{date}_test_{test_num}')

plt.figure(6)
plt.plot(plot_time, plot_theta_0_deg)
plt.xlabel('Time (s)')
plt.ylabel('Theta_0 (deg)')
plt.savefig(f'post_processing_theta_0_vs_time_{date}_test_{test_num}')
plt.show()

