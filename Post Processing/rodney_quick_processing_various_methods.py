### Make sure to adjust the date, test_num, height, and csv_path for each file ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

date = "01_31"
test_num = 2
 
height = .80645 #meters ...the "a" value in compliant mechanisms equations
yaw = 5

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
csv_path = r"C:\Users\chris\Documents\Crop Biomechanics\Stiffness Sensor Files\synthetic_stalk_data\01_31_test_2.csv" 

data = pd.read_csv(csv_path, skiprows=11)

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

########## Instantaneous stiffness method with compliant mechanisms equations ############

force_x = (kBx*(strain_ax_filtered - cAx) - kAx*(strain_bx_filtered - cBx))/(kAx*kBx*(dBx - dAx))
force_y = (kBy*(strain_ay_filtered - cAy) - kAy*(strain_by_filtered - cBy))/(kAy*kBy*(dBy - dAy))
force = (force_x**2 + force_y**2)**(1/2)

theta_0_deg = 45 - (np.arctan(force_y/force_x) * 180/np.pi)

theta_0 = theta_0_deg * np.pi/180

position_x = abs((kBx*dBx*(strain_ax_filtered-cAx) - kAx*dAx*(strain_bx_filtered-cBx))/(kBx*(strain_ax_filtered-cAx) - kAx*(strain_bx_filtered-cBx))) * 100
position_y = abs((kBy*dBy*(strain_ay_filtered-cAy) - kAy*dAy*(strain_by_filtered-cBy))/(kBy*(strain_ay_filtered-cAy) - kAy*(strain_by_filtered-cBy))) * 100

position = np.array([])

for i in range(len(force_x)):
    if abs(force_x[i]) > abs(force_y[i]):
        position = np.append(position, position_x[i])
    
    if abs(force_y[i]) > abs(force_x[i]):
        position = np.append(position, position_y[i])
    
c_theta = 1.24
gamma = 0.85
k_theta = 2.65

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
stiffness_sum_7 = 0
stiffness_num_7 = 0
stiffness_sum_8 = 0
stiffness_num_8 = 0
stiffness_sum_9 = 0
stiffness_num_9 = 0

# adjust the values in the if-statement conditions below to change which times correspond with which stalks.

for i in range(len(plot_capital_theta)):
    if plot_time[i] > 13.7 and plot_time[i] < 14.4 and plot_stiffness[i] < 25:
        stiffness_sum_1 = stiffness_sum_1 + plot_stiffness[i]
        stiffness_num_1 = stiffness_num_1 + 1
    elif plot_time[i] > 15.94 and plot_time[i] < 16.72 and plot_stiffness[i] < 35:
        stiffness_sum_2 = stiffness_sum_2 + plot_stiffness[i]
        stiffness_num_2 = stiffness_num_2 + 1
    elif plot_time[i] > 18.42 and plot_time[i] < 19.22 and plot_stiffness[i] < 25:
        stiffness_sum_3 = stiffness_sum_3 + plot_stiffness[i]
        stiffness_num_3 = stiffness_num_3 + 1
    elif plot_time[i] > 20.68 and plot_time[i] < 21.46 and plot_stiffness[i] < 25:
        stiffness_sum_4 = stiffness_sum_4 + plot_stiffness[i]
        stiffness_num_4 = stiffness_num_4 + 1
    elif plot_time[i] > 22.93 and plot_time[i] < 23.68 and plot_stiffness[i] < 25:
        stiffness_sum_5 = stiffness_sum_5 + plot_stiffness[i]
        stiffness_num_5 = stiffness_num_5 + 1
    elif plot_time[i] > 25.2 and plot_time[i] < 25.96 and plot_stiffness[i] < 25:
        stiffness_sum_6 = stiffness_sum_6 + plot_stiffness[i]
        stiffness_num_6 = stiffness_num_6 + 1
    elif plot_time[i] > 27.63 and plot_time[i] < 28.44 and plot_stiffness[i] < 35:
        stiffness_sum_7 = stiffness_sum_7 + plot_stiffness[i]
        stiffness_num_7 = stiffness_num_7 + 1
    elif plot_time[i] > 30.02 and plot_time[i] < 30.77 and plot_stiffness[i] < 25:
        stiffness_sum_8 = stiffness_sum_8 + plot_stiffness[i]
        stiffness_num_8 = stiffness_num_8 + 1
    elif plot_time[i] > 32.35 and plot_time[i] < 33.04 and plot_stiffness[i] < 25:
        stiffness_sum_9 = stiffness_sum_9 + plot_stiffness[i]
        stiffness_num_9 = stiffness_num_9 + 1

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

stiffness_5 = stiffness_sum_5/stiffness_num_5
print(f'stiffness average 5: {stiffness_5}')

stiffness_6 = stiffness_sum_6/stiffness_num_6
print(f'stiffness average 6: {stiffness_6}')

stiffness_7 = stiffness_sum_7/stiffness_num_7
print(f'stiffness average 7: {stiffness_7}')

stiffness_8 = stiffness_sum_8/stiffness_num_8
print(f'stiffness average 8: {stiffness_8}')

stiffness_9 = stiffness_sum_9/stiffness_num_9
print(f'stiffness average 9: {stiffness_9}')

################### Differential method for small angle approximations for stiffness with large angle correction factor #####################

window_size = 5

time_filtered_2 = time_filtered[2*window_size-1:]

force_x_time_1 = (kBx*(strain_ax_filtered[:-1] - cAx) - kAx*(strain_bx_filtered[:-1] - cBx))/(kAx*kBx*(dBx - dAx))
force_y_time_1 = (kBy*(strain_ay_filtered[:-1] - cAy) - kAy*(strain_by_filtered[:-1] - cBy))/(kAy*kBy*(dBy - dAy))
force_x_time_2 = (kBx*(strain_ax_filtered[1:] - cAx) - kAx*(strain_bx_filtered[1:] - cBx))/(kAx*kBx*(dBx - dAx))
force_y_time_2 = (kBy*(strain_ay_filtered[1:] - cAy) - kAy*(strain_by_filtered[1:] - cBy))/(kAy*kBy*(dBy - dAy))

force_x_time_1_smoothed = np.convolve(force_x_time_1, np.ones(window_size)/window_size, mode='valid')
force_y_time_1_smoothed = np.convolve(force_y_time_1, np.ones(window_size)/window_size, mode='valid')
force_x_time_2_smoothed = np.convolve(force_x_time_2, np.ones(window_size)/window_size, mode='valid')
force_y_time_2_smoothed = np.convolve(force_y_time_2, np.ones(window_size)/window_size, mode='valid')

force_x_time_1_extra_smoothed = np.convolve(force_x_time_1_smoothed, np.ones(window_size)/window_size, mode='valid')
force_y_time_1_extra_smoothed = np.convolve(force_y_time_1_smoothed, np.ones(window_size)/window_size, mode='valid')
force_x_time_2_extra_smoothed = np.convolve(force_x_time_2_smoothed, np.ones(window_size)/window_size, mode='valid')
force_y_time_2_extra_smoothed = np.convolve(force_y_time_2_smoothed, np.ones(window_size)/window_size, mode='valid')

force_x_diff = force_x_time_2_extra_smoothed - force_x_time_1_extra_smoothed
force_y_diff = force_y_time_2_extra_smoothed - force_y_time_1_extra_smoothed
force_diff = (force_x_diff**2 + force_y_diff**2)**(1/2)

position_x_time_1 = abs((kBx*dBx*(strain_ax_filtered[:-1] - cAx) - kAx*dAx*(strain_bx_filtered[:-1] - cBx))/(kBx*(strain_ax_filtered[:-1] - cAx) - kAx*(strain_bx_filtered[:-1] - cBx)))
position_y_time_1 = abs((kBy*dBy*(strain_ay_filtered[:-1] - cAy) - kAy*dAy*(strain_by_filtered[:-1] - cBy))/(kBy*(strain_ay_filtered[:-1] - cAy) - kAy*(strain_by_filtered[:-1] - cBy)))
position_x_time_2 = abs((kBx*dBx*(strain_ax_filtered[1:] - cAx) - kAx*dAx*(strain_bx_filtered[1:] - cBx))/(kBx*(strain_ax_filtered[1:] - cAx) - kAx*(strain_bx_filtered[1:] - cBx)))
position_y_time_2 = abs((kBy*dBy*(strain_ay_filtered[1:] - cAy) - kAy*dAy*(strain_by_filtered[1:] - cBy))/(kBy*(strain_ay_filtered[1:] - cAy) - kAy*(strain_by_filtered[1:] - cBy)))

position_x_time_1_smoothed = np.convolve(position_x_time_1, np.ones(window_size)/window_size, mode='valid') * 100
position_y_time_1_smoothed = np.convolve(position_y_time_1, np.ones(window_size)/window_size, mode='valid') * 100
position_x_time_2_smoothed = np.convolve(position_x_time_2, np.ones(window_size)/window_size, mode='valid') * 100
position_y_time_2_smoothed = np.convolve(position_y_time_2, np.ones(window_size)/window_size, mode='valid') * 100

position_x_time_1_extra_smoothed = np.convolve(position_x_time_1_smoothed, np.ones(window_size)/window_size, mode='valid')
position_y_time_1_extra_smoothed = np.convolve(position_y_time_1_smoothed, np.ones(window_size)/window_size, mode='valid') 
position_x_time_2_extra_smoothed = np.convolve(position_x_time_2_smoothed, np.ones(window_size)/window_size, mode='valid') 
position_y_time_2_extra_smoothed = np.convolve(position_y_time_2_smoothed, np.ones(window_size)/window_size, mode='valid') 

position_x_diff = (position_x_time_2_extra_smoothed - position_x_time_1_extra_smoothed)
position_y_diff = (position_y_time_2_extra_smoothed - position_y_time_1_extra_smoothed)
position_diff = (position_x_diff + position_y_diff)/2

deflection_diff = position_diff*np.sin(yaw*np.pi/180) / 100

stiffness_diff = np.abs(force_diff*height**3/(18*deflection_diff))

plot_time_2 = np.array([])
plot_stiffness_diff = np.array([])

plt.plot(time_filtered_2, position_x_time_1_extra_smoothed)
plt.plot(time_filtered_2, position_x_time_2_extra_smoothed)

plt.show()

for i in range(len(time_filtered_2)):
    if position_x_time_2_extra_smoothed[i] < 25 and position_x_time_2_extra_smoothed[i] > 10:
        plot_time_2 = np.append(plot_time_2, time_filtered_2[i]) 
        plot_stiffness_diff = np.append(plot_stiffness_diff, stiffness_diff[i])

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
stiffness_sum_7 = 0
stiffness_num_7 = 0
stiffness_sum_8 = 0
stiffness_num_8 = 0
stiffness_sum_9 = 0
stiffness_num_9 = 0

# adjust the values in the if-statement conditions below to change which times correspond with which stalks.

for i in range(len(plot_time_2)):
    if plot_time_2[i] > 13.7 and plot_time_2[i] < 14.4 and plot_stiffness_diff[i] < 60:
        stiffness_sum_1 = stiffness_sum_1 + plot_stiffness_diff[i]
        stiffness_num_1 = stiffness_num_1 + 1
        print(plot_stiffness_diff[i])
    elif plot_time_2[i] > 15.94 and plot_time_2[i] < 16.72 and plot_stiffness_diff[i] < 60:
        stiffness_sum_2 = stiffness_sum_2 + plot_stiffness_diff[i]
        stiffness_num_2 = stiffness_num_2 + 1
    elif plot_time_2[i] > 18.42 and plot_time_2[i] < 19.22 and plot_stiffness_diff[i] < 60:
        stiffness_sum_3 = stiffness_sum_3 + plot_stiffness_diff[i]
        stiffness_num_3 = stiffness_num_3 + 1
    elif plot_time_2[i] > 20.68 and plot_time_2[i] < 21.46 and plot_stiffness_diff[i] < 60:
        stiffness_sum_4 = stiffness_sum_4 + plot_stiffness_diff[i]
        stiffness_num_4 = stiffness_num_4 + 1
    elif plot_time_2[i] > 22.93 and plot_time_2[i] < 23.68 and plot_stiffness_diff[i] < 60:
        stiffness_sum_5 = stiffness_sum_5 + plot_stiffness_diff[i]
        stiffness_num_5 = stiffness_num_5 + 1
    elif plot_time_2[i] > 25.2 and plot_time_2[i] < 25.96 and plot_stiffness_diff[i] < 60:
        stiffness_sum_6 = stiffness_sum_6 + plot_stiffness_diff[i]
        stiffness_num_6 = stiffness_num_6 + 1
    elif plot_time_2[i] > 27.63 and plot_time_2[i] < 28.44 and plot_stiffness_diff[i] < 60:
        stiffness_sum_7 = stiffness_sum_7 + plot_stiffness_diff[i]
        stiffness_num_7 = stiffness_num_7 + 1
    elif plot_time_2[i] > 30.02 and plot_time_2[i] < 30.77 and plot_stiffness_diff[i] < 60:
        stiffness_sum_8 = stiffness_sum_8 + plot_stiffness_diff[i]
        stiffness_num_8 = stiffness_num_8 + 1
    elif plot_time_2[i] > 32.35 and plot_time_2[i] < 33.04 and plot_stiffness_diff[i] < 60:
        stiffness_sum_9 = stiffness_sum_9 + plot_stiffness_diff[i]
        stiffness_num_9 = stiffness_num_9 + 1

stiffness_1 = stiffness_sum_1/stiffness_num_1
print(f'differential small angle deflection stiffness average 1: {stiffness_1}')

stiffness_2 = stiffness_sum_2/stiffness_num_2
print(f'differential small angle deflection stiffness average 2: {stiffness_2}')

stiffness_3 = stiffness_sum_3/stiffness_num_3
print(f'differential small angle deflection stiffness average 3: {stiffness_3}')

stiffness_4 = stiffness_sum_4/stiffness_num_4
print(f'differential small angle deflection stiffness average 4: {stiffness_4}')

stiffness_5 = stiffness_sum_5/stiffness_num_5
print(f'differential small angle deflection stiffness average 5: {stiffness_5}')

stiffness_6 = stiffness_sum_6/stiffness_num_6
print(f'differential small angle deflection stiffness average 6: {stiffness_6}')

stiffness_5 = stiffness_sum_5/stiffness_num_5
print(f'differential small angle deflection stiffness average 5: {stiffness_5}')

stiffness_6 = stiffness_sum_6/stiffness_num_6
print(f'differential small angle deflection stiffness average 6: {stiffness_6}')

stiffness_7 = stiffness_sum_7/stiffness_num_7
print(f'differential small angle deflection stiffness average 7: {stiffness_7}')

stiffness_8 = stiffness_sum_8/stiffness_num_8
print(f'differential small angle deflection stiffness average 8: {stiffness_8}')

stiffness_9 = stiffness_sum_9/stiffness_num_9
print(f'differential small angle deflection stiffness average 9: {stiffness_9}')


plt.figure(1)
plt.plot(plot_time, plot_position_x, 'r', label='Instantaenous Compliant Mechanisms X-Position')
plt.plot(plot_time, plot_position_y, 'b', label='Instantaenous Compliant Mechanisms Y-Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.legend()
plt.savefig(f'post_processing_positions_vs_time_{date}_test_{test_num}')

plt.figure(2)
plt.plot(plot_time, plot_force_x, 'r', label='Instnataneous Compliant Mechanisms X-Axis Force')
plt.plot(plot_time, plot_force_y, 'b', label='Instantaneous Compliant Mechanisms Y-Axis Force')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.savefig(f'post_processing_forces_vs_time_{date}_test_{test_num}')

plt.figure(5)
plt.plot(plot_time, plot_stiffness, '.', label='Instantaneous Compliant Mechanisms Stiffness')
plt.plot(plot_time_2, plot_stiffness_diff, '.', label='Differential Small Angle Deflection Stiffness')
plt.xlabel('Time (s)')
plt.ylabel('Stiffness (N*m^2)')
plt.ylim(0, 50)
plt.savefig(f'post_processing_stiffness_vs_time_{date}_test_{test_num}')

plt.show()