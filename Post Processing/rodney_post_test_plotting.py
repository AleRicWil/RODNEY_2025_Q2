### Make sure to adjust the date, test_num, height, and csv_path for each file ####

import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

date = "09_20"
test_num = 11

height = .578 #meters ...the "a" value in compliant mechanisms equations

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
csv_path = r"Impact_test.csv" 

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
    
print(stalk_starts)
print(stalk_ends)
#print(time_filtered[int(stalk_ends[0])])
time_fit = np.array([])
force_x_fit = np.array([])
force_y_fit = np.array([])
strain_ax_fit = np.array([])
strain_bx_fit = np.array([])
strain_ay_fit = np.array([])
strain_by_fit = np.array([])

poly = PolynomialFeatures(degree=12)
poly_count = 0
for i in range(len(stalk_starts)):
    
    time_poly = poly.fit_transform(time_filtered[int(stalk_starts[i]):int(stalk_ends[i])].reshape(-1, 1))
    strain_ax_model = LinearRegression()
    strain_ax_model.fit(time_poly, strain_ax_filtered[int(stalk_starts[i]):int(stalk_ends[i])])
    strain_bx_model = LinearRegression()
    strain_bx_model.fit(time_poly, strain_bx_filtered[int(stalk_starts[i]):int(stalk_ends[i])])
    strain_ay_model = LinearRegression()
    strain_ay_model.fit(time_poly, strain_ay_filtered[int(stalk_starts[i]):int(stalk_ends[i])])
    strain_by_model = LinearRegression()
    strain_by_model.fit(time_poly, strain_by_filtered[int(stalk_starts[i]):int(stalk_ends[i])])
    poly_count = poly_count + 1
    #print(poly_count)

    strain_ax_fit = np.append(strain_ax_fit, strain_ax_model.predict(time_poly))
    strain_bx_fit = np.append(strain_bx_fit, strain_bx_model.predict(time_poly))
    strain_ay_fit = np.append(strain_ay_fit, strain_ay_model.predict(time_poly))
    strain_by_fit = np.append(strain_by_fit, strain_by_model.predict(time_poly))

    time_fit = np.append(time_fit, time_filtered[int(stalk_starts[i]):int(stalk_ends[i])])


force_x_fit = (kBx*(strain_ax_fit - cAx) - kAx*(strain_bx_fit - cBx))/(kAx*kBx*(dBx - dAx))
force_y_fit = (kBy*(strain_ay_fit - cAy) - kAy*(strain_by_fit - cBy))/(kAy*kBy*(dBy - dAy))
force_fit = (force_x_fit**2 + force_y_fit**2)**(1/2)

theta_0_deg = 43.15 - (np.arctan(force_y/force_x) * 180/np.pi)

theta_0_deg_fit = 43.15 - (np.arctan(force_y_fit/force_x_fit) * 180/np.pi)

theta_0 = theta_0_deg * np.pi/180

theta_0_fit = theta_0_deg_fit * np.pi/180

position_x = abs((kBx*dBx*(strain_ax_filtered-cAx) - kAx*dAx*(strain_bx_filtered-cBx))/(kBx*(strain_ax_filtered-cAx) - kAx*(strain_bx_filtered-cBx))) * 100
position_y = abs((kBy*dBy*(strain_ay_filtered-cAy) - kAy*dAy*(strain_by_filtered-cBy))/(kBy*(strain_ay_filtered-cAy) - kAy*(strain_by_filtered-cBy))) * 100

position_x_fit = abs((kBx*dBx*(strain_ax_fit-cAx) - kAx*dAx*(strain_bx_fit-cBx))/(kBx*(strain_ax_fit-cAx) - kAx*(strain_bx_fit-cBx))) * 100
position_y_fit = abs((kBy*dBy*(strain_ay_fit-cAy) - kAy*dAy*(strain_by_fit-cBy))/(kBy*(strain_ay_fit-cAy) - kAy*(strain_by_fit-cBy))) * 100

position = np.array([])
position_fit = np.array([])
for i in range(len(force_x)):
    if abs(force_x[i]) > abs(force_y[i]):
        position = np.append(position, position_x[i])
    
    if abs(force_y[i]) > abs(force_x[i]):
        position = np.append(position, position_y[i])

for i in range(len(force_x_fit)):
    if abs(force_x_fit[i]) > abs(force_y_fit[i]):
        position_fit = np.append(position_fit, position_x_fit[i])
    
    if abs(force_y_fit[i]) > abs(force_x_fit[i]):
        position_fit = np.append(position_fit, position_y_fit[i])

phi = theta_0 + np.pi/2
phi_fit = theta_0_fit + np.pi/2
n = -1/np.tan(phi)
n_fit = -1/np.tan(phi_fit)
n_table = np.array([-5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
c_theta_table = np.array([1.1788, 1.1971, 1.2119, 1.2293, 1.2322, 1.2323, 1.2348, 1.2385, 1.243, 1.2467, 1.2492, 1.2511, 1.2534, 1.2548, 1.2557, 1.257, 1.2578])
c_theta = np.array([])
c_theta_fit = np.array([])
gamma = np.array([])
gamma_fit = np.array([])
k_theta = np.array([])
k_theta_fit = np.array([])

for i in range(len(n)):
    if n[i] <= n[1]:
        c_theta_calc = c_theta_table[1] - (c_theta_table[1]-c_theta_table[0])/(n_table[1]-n_table[0])
    elif n[i] <= n[2]:
        c_theta_calc = c_theta_table[2] - (c_theta_table[2]-c_theta_table[1])/(n_table[2]-n_table[1])
    elif n[i] <= n[3]:
        c_theta_calc = c_theta_table[3] - (c_theta_table[3]-c_theta_table[2])/(n_table[3]-n_table[2])
    elif n[i] <= n[4]:
        c_theta_calc = c_theta_table[4] - (c_theta_table[4]-c_theta_table[3])/(n_table[4]-n_table[3])
    elif n[i] <= n[5]:
        c_theta_calc = c_theta_table[5] - (c_theta_table[5]-c_theta_table[4])/(n_table[5]-n_table[4])
    elif n[i] <= n[6]:
        c_theta_calc = c_theta_table[6] - (c_theta_table[6]-c_theta_table[5])/(n_table[6]-n_table[5])
    elif n[i] <= n[7]:
        c_theta_calc = c_theta_table[7] - (c_theta_table[7]-c_theta_table[6])/(n_table[7]-n_table[6])
    elif n[i] <= n[8]:
        c_theta_calc = c_theta_table[8] - (c_theta_table[8]-c_theta_table[7])/(n_table[8]-n_table[7])
    elif n[i] <= n[9]:
        c_theta_calc = c_theta_table[9] - (c_theta_table[9]-c_theta_table[8])/(n_table[9]-n_table[8])
    elif n[i] <= n[10]:
        c_theta_calc = c_theta_table[10] - (c_theta_table[10]-c_theta_table[9])/(n_table[10]-n_table[9])
    elif n[i] <= n[11]:
        c_theta_calc = c_theta_table[11] - (c_theta_table[11]-c_theta_table[10])/(n_table[11]-n_table[10])
    elif n[i] <= n[12]:
        c_theta_calc = c_theta_table[12] - (c_theta_table[12]-c_theta_table[11])/(n_table[12]-n_table[11])
    elif n[i] <= n[13]:
        c_theta_calc = c_theta_table[13] - (c_theta_table[13]-c_theta_table[12])/(n_table[13]-n_table[12])
    elif n[i] <= n[14]:
        c_theta_calc = c_theta_table[14] - (c_theta_table[14]-c_theta_table[13])/(n_table[14]-n_table[13])
    elif n[i] <= n[15]:
        c_theta_calc = c_theta_table[15] - (c_theta_table[15]-c_theta_table[14])/(n_table[15]-n_table[14])
    elif n[i] <= n[16]:
        c_theta_calc = c_theta_table[16] - (c_theta_table[16]-c_theta_table[15])/(n_table[16]-n_table[15])
    else:
        #print("hello")
        c_theta_calc = c_theta_table[16]

    c_theta = np.append(c_theta, 1.24)

    if n[i] > 0.5 and n[i] <= 10:
        gamma_calc = 0.841655 - 0.0067807*n[i] + 0.000438*n[i]**2
    elif n[i] > -1.8316 and n[i] <= 0.5:
        gamma_calc = 0.852144 - 0.0182867*n[i]
    elif n[i] > -5 and n[i] <= -1.8316:
        gamma_calc = 0.912364 + 0.0145928*n[i]
    else:
        gamma_calc = .85
    
    gamma = np.append(gamma, .85)

    if n[i] > -5 and n[i] <= -2.5:
        k_theta_calc = 3.024112 + 0.12129*n[i] + 0.003169*n[i]**2
    elif n[i] > -2.5 and n[i] <= -1:
        k_theta_calc = 1.967657 - 2.616021*n[i] - 3.738166*n[i]**2 - 2.649437*n[i]**3 \
                        - 0.891906*n[i]**4 - 0.113063*n[i]**5
    elif n[i] > -1 and n[i] <= 10:
        k_theta_calc = 2.654855 - 0.509896*10**-1*n[i] + 0.126749*10**-1*n[i]**2 \
                        - 0.142039*10**-2*n[i]**3 + 0.584525*10**-4*n[i]**4
    else:
        k_theta_calc = 2.65

    k_theta = np.append(k_theta, 2.65)

#curve fit calculations
for i in range(len(n_fit)):
    if n_fit[i] <= n_fit[1]:
        c_theta_calc_fit = c_theta_table[1] - (c_theta_table[1]-c_theta_table[0])/(n_table[1]-n_table[0])
    elif n_fit[i] <= n_fit[2]:
        c_theta_calc_fit = c_theta_table[2] - (c_theta_table[2]-c_theta_table[1])/(n_table[2]-n_table[1])
    elif n_fit[i] <= n[3]:
        c_theta_calc_fit = c_theta_table[3] - (c_theta_table[3]-c_theta_table[2])/(n_table[3]-n_table[2])
    elif n_fit[i] <= n[4]:
        c_theta_calc_fit = c_theta_table[4] - (c_theta_table[4]-c_theta_table[3])/(n_table[4]-n_table[3])
    elif n_fit[i] <= n[5]:
        c_theta_calc_fit = c_theta_table[5] - (c_theta_table[5]-c_theta_table[4])/(n_table[5]-n_table[4])
    elif n_fit[i] <= n[6]:
        c_theta_calc_fit = c_theta_table[6] - (c_theta_table[6]-c_theta_table[5])/(n_table[6]-n_table[5])
    elif n_fit[i] <= n[7]:
        c_theta_calc_fit = c_theta_table[7] - (c_theta_table[7]-c_theta_table[6])/(n_table[7]-n_table[6])
    elif n_fit[i] <= n[8]:
        c_theta_calc_fit = c_theta_table[8] - (c_theta_table[8]-c_theta_table[7])/(n_table[8]-n_table[7])
    elif n_fit[i] <= n[9]:
        c_theta_calc_fit = c_theta_table[9] - (c_theta_table[9]-c_theta_table[8])/(n_table[9]-n_table[8])
    elif n_fit[i] <= n[10]:
        c_theta_calc_fit = c_theta_table[10] - (c_theta_table[10]-c_theta_table[9])/(n_table[10]-n_table[9])
    elif n_fit[i] <= n[11]:
        c_theta_calc_fit = c_theta_table[11] - (c_theta_table[11]-c_theta_table[10])/(n_table[11]-n_table[10])
    elif n_fit[i] <= n[12]:
        c_theta_calc_fit = c_theta_table[12] - (c_theta_table[12]-c_theta_table[11])/(n_table[12]-n_table[11])
    elif n_fit[i] <= n[13]:
        c_theta_calc_fit = c_theta_table[13] - (c_theta_table[13]-c_theta_table[12])/(n_table[13]-n_table[12])
    elif n_fit[i] <= n[14]:
        c_theta_calc_fit = c_theta_table[14] - (c_theta_table[14]-c_theta_table[13])/(n_table[14]-n_table[13])
    elif n_fit[i] <= n[15]:
        c_theta_calc_fit = c_theta_table[15] - (c_theta_table[15]-c_theta_table[14])/(n_table[15]-n_table[14])
    elif n_fit[i] <= n[16]:
        c_theta_calc_fit = c_theta_table[16] - (c_theta_table[16]-c_theta_table[15])/(n_table[16]-n_table[15])
    else:
        c_theta_calc_fit = c_theta_table[16]

    c_theta_fit = np.append(c_theta_fit, c_theta_calc_fit)

    if n_fit[i] > 0.5 and n_fit[i] <= 10:
        gamma_calc_fit = 0.841655 - 0.0067807*n_fit[i] + 0.000438*n_fit[i]**2
    elif n_fit[i] > -1.8316 and n_fit[i] <= 0.5:
        gamma_calc_fit = 0.852144 - 0.0182867*n_fit[i]
    elif n_fit[i] > -5 and n_fit[i] <= -1.8316:
        gamma_calc_fit = 0.912364 + 0.0145928*n_fit[i]
    else:
        gamma_calc_fit = .85
    
    gamma_fit = np.append(gamma_fit, gamma_calc_fit)

    if n_fit[i] > -5 and n_fit[i] <= -2.5:
        k_theta_calc_fit = 3.024112 + 0.12129*n_fit[i] + 0.003169*n_fit[i]**2
    elif n_fit[i] > -2.5 and n_fit[i] <= -1:
        k_theta_calc_fit = 1.967657 - 2.616021*n_fit[i] - 3.738166*n_fit[i]**2 - 2.649437*n_fit[i]**3 \
                        - 0.891906*n_fit[i]**4 - 0.113063*n_fit[i]**5
    elif n_fit[i] > -1 and n_fit[i] <= 10:
        k_theta_calc_fit = 2.654855 - 0.509896*10**-1*n_fit[i] + 0.126749*10**-1*n_fit[i]**2 \
                        - 0.142039*10**-2*n_fit[i]**3 + 0.584525*10**-4*n_fit[i]**4
    else:
        k_theta_calc_fit = 2.65

    k_theta_fit = np.append(k_theta_fit, k_theta_calc_fit)


print(n)
capital_theta = theta_0 / c_theta
capital_theta_fit = theta_0_fit / c_theta_fit

L = height/(1-gamma*(1-np.cos(capital_theta)))
b = gamma*L*np.sin(capital_theta)

force_x_prime = force*np.sin(np.pi/2-theta_0)
force_y_prime = force*np.cos(np.pi/2-theta_0)
stiffness = np.abs((force_x_prime*(height-L*(1-gamma)) + force_y_prime*b)*L / (capital_theta*gamma*k_theta)) # units of N*m^2

L_fit = height/(1-gamma_fit*(1-np.cos(capital_theta_fit)))
b_fit = gamma_fit*L_fit*np.sin(capital_theta_fit)
force_x_prime_fit = force_fit*np.sin(np.pi/2-theta_0_fit)
force_y_prime_fit = force_fit*np.cos(np.pi/2-theta_0_fit)
stiffness_fit = np.abs((force_x_prime_fit*height + force_y_prime_fit*b_fit)*L_fit / (capital_theta_fit*gamma_fit*k_theta_fit)) # units of N*m^2

plot_time = np.array([])
plot_force_x = np.array([])
plot_force_y = np.array([])
plot_force = np.array([])
plot_position_x = np.array([])
plot_position_y = np.array([])
plot_position = np.array([])
plot_theta_0_deg = np.array([])
plot_stiffness = np.array([])

plot_time_fit = np.array([])
plot_force_x_fit = np.array([])
plot_force_y_fit = np.array([])
plot_force_fit = np.array([])
plot_position_x_fit = np.array([])
plot_position_y_fit = np.array([])
plot_position_fit = np.array([])
plot_theta_0_deg_fit = np.array([])
plot_stiffness_fit = np.array([])

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

for i in range(len(position_x_fit)):
    if position_x_fit[i] < 17 and position_x_fit[i] > 14 \
        and position_y_fit[i] < 20.5 and position_y_fit[i] > 10:
        plot_time_fit = np.append(plot_time_fit, time_fit[i]) 
        plot_force_x_fit = np.append(plot_force_x_fit, force_x_fit[i])
        plot_force_y_fit = np.append(plot_force_y_fit, force_y_fit[i])
        plot_force_fit = np.append(plot_force_fit, force_fit[i])
        plot_position_x_fit = np.append(plot_position_x_fit, position_x_fit[i])
        plot_position_y_fit = np.append(plot_position_y_fit, position_y_fit[i])
        plot_position_fit = np.append(plot_position_fit, position_fit[i])
        plot_theta_0_deg_fit = np.append(plot_theta_0_deg_fit, theta_0_deg_fit[i])
        plot_stiffness_fit = np.append(plot_stiffness_fit, stiffness_fit[i])
        
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

plt.figure(7)
plt.plot(plot_time_fit, plot_position_x_fit, 'r.', label='X-Position Curve Fit')
plt.plot(plot_time_fit, plot_position_y_fit, 'b.', label='Y-Position Curve Fit')
plt.xlabel('Time (s)')
plt.ylabel('Position (cm)')
plt.legend()
plt.savefig(f'post_processing_curve_fit_positions_vs_time_{date}_test_{test_num}')

plt.figure(8)
plt.plot(plot_time_fit, plot_force_x_fit, 'r.', label='X-Axis Force Curve Fit')
plt.plot(plot_time_fit, plot_force_y_fit, 'b.', label='Y-Axis Force Curve Fit')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.savefig(f'post_processing_curve_fit_forces_vs_time_{date}_test_{test_num}')

plt.figure(9)
plt.plot(plot_time_fit, plot_stiffness_fit, '.', label='Curve Fit Stiffness')
plt.xlabel('Time (s)')
plt.ylabel('Stiffness (N*m^2)')
plt.ylim(0, 50)
plt.legend()
plt.savefig(f'post_processing_curve_fit_stiffness_vs_time_{date}_test_{test_num}')

plt.figure(10)
plt.plot(plot_time_fit, plot_theta_0_deg_fit, '.', label='Curve Fit Theta_0')
plt.xlabel('Time (s)')
plt.ylabel('Theta_0 (deg)')
plt.savefig(f'post_processing_curve_fit_theta_0_vs_time_{date}_test_{test_num}')
plt.show()
