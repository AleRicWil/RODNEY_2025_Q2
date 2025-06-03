#Name: Christian Shamo
#Date: 02/07/2025
#Description: An arduino Uno prints strain data for 4 wheatstone half-bridges, which data is separated by commas,
#             to a serial port. Time is also included. This program waits for a push-button, which is connected 
#             to the Arduino to be pressed. When it's pressed, this program starts to store the incoming data and
#             plot it in real time for the user to see. When the push-button is pressed again, the .csv file is 
#             closed and data storage stops. Each time the button is pressed, a message is shown on the terminal
#             window below to indicate that the button-press was recognized. Only this program, not the Arduino
#             program, needs to be open for everything to work - as long as the Arduino code has already been 
#             flashed to the Arduino. After this push-button has been pressed twice, to initialize and end the
#             data storage process, this program must be run again before attempting another data storage
#             sequence. The csv file name should be changed as well to be the name of the new test. 
# 
#Dynamic Inputs (Change Every Test or Check Every Test):
#        -date       
#        -test_num 
#        -height     
#Other Inputs: 
#        -Incoming data from the serial port, which is printed to the serial port from an Arduino Uno.  
#        -Com # (must be same as in the corresponding Arduino program)
#        -Baud rate (must be the same as in the corresponding Arduino program)
#        
#
#Outputs: 
#         -A .csv file containing the data from the serial port is created and written to in real time. 
#          The data's time frame is from between times either space bar or push button is pressed.
#         -Plots using filtered data from the .csv file are displayed at the end of the file's execution


import serial   # pip install pyserial
import sys
import csv
import keyboard
from sklearn.linear_model import LinearRegression   # pip install scikit-learn
from sklearn.preprocessing import PolynomialFeatures
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
np.set_printoptions(threshold=sys.maxsize)

supply_voltage = 5
resolution = 2**9
gain = 16

# calibration coefficients, from 06/17/24
kAx =  0.1265
kAy = 0.109
kBx = 0.1227
kBy = 0.0997

dAx = 0.0634
dAy = 0.0157
dBx = 0.015
dBy = 0.0647

### Update Each Test Run ###
date = "02_21"
test_num = 27

### Check Each Test Run ###
configuration = "Config 1"
pvc_stiffness = "Medium"
height = 80.645         # cm
yaw = 5                 # degrees
pitch = 0               # degrees
roll = 0                # degrees
rate_of_travel = 25     # ft/min
angle_of_travel = 0     # degrees
offset_distance = 25    # cm
 
class RealTimePlotWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Configure the serial connection
        self.ser = serial.Serial('COM6', 115200)  # Adjust the 'COM' # to your serial port, and the baud rate

        while True:
            incoming_data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if incoming_data == "#" or keyboard.is_pressed('space'):
            #if incoming_data == "#":
                print("Starting data collection.")
                # Clear the buffer after sleeping
                #while self.ser.in_waiting > 0:
                #    self.ser.readline()
                break

        # Open a CSV file to write data
        self.csvfile = open(f'{date}_test_{test_num}.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        pre_test_notes = [
            ["pre_test_note_1"],
            ["rodney configuration", configuration],
            ["medium stiffness (Nm^2)", pvc_stiffness],
            ["height (cm)", height],
            ["yaw angle (degrees)", yaw],
            ["pitch angle (degrees)", pitch],
            ["roll angle (degrees)", roll],
            ["rate of travel (ft/min)", rate_of_travel],
            ["angle of travel (degrees)", angle_of_travel],
            ["offset distance (cm)", offset_distance],
            ["=================================================================="]
        ]

        for note in pre_test_notes:
            self.csvwriter.writerow(note)

        # Write the headers in the next row
        headers = ['Time (Microseconds)', 'Strain Ax', 'Strain Bx', 'Strain Ay', 'Strain By', 'Current Time']
        self.csvwriter.writerow(headers)

        # Initialize PyQtGraph application
        pg.setConfigOptions(antialias=True)

        # Create 2 plot windows, one for x strains and one for y strains
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.resize(1500, 1000)
        self.win.move(0, 0)
        self.win.setWindowTitle('X Strain Vs Time')
        self.win.setWindowTitle('Y Strain Vs Time')
        self.plot_x = self.win.addPlot(title='X Strain Vs Time')
        self.curve_ax = self.plot_x.plot(pen='r', name='Ax Strain')
        self.curve_bx = self.plot_x.plot(pen='b', name='Bx Strain')
        self.plot_y = self.win.addPlot(title='Y Strain Vs Time')
        self.curve_ay = self.plot_y.plot(pen='r', name='Ay Strain')
        self.curve_by = self.plot_y.plot(pen='b', name='By Strain')
            
        self.time_sec = np.array([])
        self.strain_ax = np.array([])
        self.strain_bx = np.array([])
        self.strain_ay = np.array([])
        self.strain_by = np.array([])

        # QTimer for updating plot
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)  # Update plot every 1 ms
        self.plot_time = 0

        self.time_offset_check = True
        self.time_offset = 0

    def update_plot(self):
        
        try:
            line = self.ser.readline().decode('utf-8').strip()
            # separate the incoming data into columns after every comma
            data = line.split(',')
            
            if data[0] == "$":
                print(f'reset at {float(data[1])*10**-6}')
                return

            now = datetime.now()
            current_time = now.time()

            # When the push-button is pushed a 2nd time, "test ended" is printed to the serial port in Arduino
            if data[0] == "test ended" or keyboard.is_pressed('space'):
                print("The data collection has ended.")
                csv_path = self.csvfile.name
                self.ser.close()
                self.csvfile.close()
                self.timer.stop()

                csv_data = pd.read_csv(csv_path, skiprows=11)

                csv_time_sec = csv_data['Time (Microseconds)'].to_numpy()
                csv_strain_ax = csv_data['Strain Ax'].to_numpy()
                csv_strain_bx = csv_data['Strain Bx'].to_numpy()
                csv_strain_ay = csv_data['Strain Ay'].to_numpy()
                csv_strain_by = csv_data['Strain By'].to_numpy()

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
                gamma = .85
                k_theta = 2.65

                capital_theta = theta_0 / c_theta

                L = height/100/(1-gamma*(1-np.cos(capital_theta)))
                b = gamma*L*np.sin(capital_theta)

                force_x_prime = force*np.sin(np.pi/2-theta_0)
                force_y_prime = force*np.cos(np.pi/2-theta_0)
                stiffness = np.abs((force_x_prime*(height/100-L*(1-gamma)) + force_y_prime*b)*L / (capital_theta*gamma*k_theta)) # units of N*m^2

                plot_time = np.array([])
                plot_force_x = np.array([])
                plot_force_y = np.array([])
                plot_force = np.array([])
                plot_position_x = np.array([])
                plot_position_y = np.array([])
                plot_position = np.array([])
                plot_theta_0_deg = np.array([])
                plot_stiffness = np.array([])

                for i in range(len(position_x)):
                    if position_x[i] < 20.5 and position_x[i] > 12:
                        plot_time = np.append(plot_time, time_filtered[i]) 
                        plot_force_x = np.append(plot_force_x, force_x[i])
                        plot_force_y = np.append(plot_force_y, force_y[i])
                        plot_force = np.append(plot_force, force[i])
                        plot_position_x = np.append(plot_position_x, position_x[i])
                        plot_position_y = np.append(plot_position_y, position_y[i])
                        plot_position = np.append(plot_position, position[i])
                        plot_theta_0_deg = np.append(plot_theta_0_deg, theta_0_deg[i])
                        plot_stiffness = np.append(plot_stiffness, stiffness[i])

                ##### smoothed data instead of filtered data ########

                window_size = 5

                time_doubly_smoothed = time_filtered[2*window_size-2:]

                print(len(time_doubly_smoothed))

                strain_ax_smoothed = np.convolve(strain_ax_filtered, np.ones(window_size)/window_size, mode='valid')
                strain_bx_smoothed = np.convolve(strain_bx_filtered, np.ones(window_size)/window_size, mode='valid')
                strain_ay_smoothed = np.convolve(strain_ay_filtered, np.ones(window_size)/window_size, mode='valid')
                strain_by_smoothed = np.convolve(strain_by_filtered, np.ones(window_size)/window_size, mode='valid')

                strain_ax_doubly_smoothed = np.convolve(strain_ax_smoothed, np.ones(window_size)/window_size, mode='valid')
                strain_bx_doubly_smoothed = np.convolve(strain_bx_smoothed, np.ones(window_size)/window_size, mode='valid')
                strain_ay_doubly_smoothed = np.convolve(strain_ay_smoothed, np.ones(window_size)/window_size, mode='valid')
                strain_by_doubly_smoothed = np.convolve(strain_by_smoothed, np.ones(window_size)/window_size, mode='valid')

                force_x_2 = (kBx*(strain_ax_doubly_smoothed - cAx) - kAx*(strain_bx_doubly_smoothed - cBx))/(kAx*kBx*(dBx - dAx))
                force_y_2 = (kBy*(strain_ay_doubly_smoothed - cAy) - kAy*(strain_by_doubly_smoothed - cBy))/(kAy*kBy*(dBy - dAy))
                force_2 = (force_x_2**2 + force_y_2**2)**(1/2)

                position_x_2 = abs((kBx*dBx*(strain_ax_doubly_smoothed - cAx) - kAx*dAx*(strain_bx_doubly_smoothed - cBx))/(kBx*(strain_ax_doubly_smoothed - cAx) - kAx*(strain_bx_doubly_smoothed - cBx))) * 100
                position_y_2 = abs((kBy*dBy*(strain_ay_doubly_smoothed - cAy) - kAy*dAy*(strain_by_doubly_smoothed - cBy))/(kBy*(strain_ay_doubly_smoothed - cAy) - kAy*(strain_by_doubly_smoothed - cBy))) * 100

                theta_0_deg_2 = 45 - (np.arctan(force_y_2/force_x_2) * 180/np.pi)

                theta_0_2 = theta_0_deg_2 * np.pi/180

                position_2 = np.array([])
                
                for i in range(len(force_x_2)):
                    if abs(force_x_2[i]) > abs(force_y_2[i]):
                        position_2 = np.append(position_2, position_x_2[i])
                        print(i)
                    
                    if abs(force_y_2[i]) > abs(force_x_2[i]):
                        position_2 = np.append(position_2, position_y_2[i])

                c_theta_2 = 1.24
                gamma_2 = .85
                k_theta_2 = 2.65

                capital_theta_2 = theta_0_2 / c_theta_2

                L_2 = height/100/(1-gamma_2*(1-np.cos(capital_theta_2)))
                b_2 = gamma_2*L_2*np.sin(capital_theta_2)

                force_x_prime_2 = force_2*np.sin(np.pi/2-theta_0_2)
                force_y_prime_2 = force_2*np.cos(np.pi/2-theta_0_2)
                stiffness_2 = np.abs((force_x_prime_2*(height/100-L_2*(1-gamma_2)) + force_y_prime_2*b_2)*L_2 / (capital_theta_2*gamma_2*k_theta_2)) # units of N*m^2

                print(len(stiffness_2))
                ### comment out below this line when doing normal test runs ###

                """
                plt.figure(1)
                plt.plot(plot_time, plot_position_x, 'r', label='X-Position')
                plt.plot(plot_time, plot_position_y, 'b', label='Y-Position')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (cm)')
                plt.legend()
                plt.savefig(f'positions_vs_time_{date}_test_{test_num}')

                plt.figure(2)
                plt.plot(plot_time, plot_force_x, 'r', label='X-Axis Force')
                plt.plot(plot_time, plot_force_y, 'b', label='Y-Axis Force')
                plt.xlabel('Time (s)')
                plt.ylabel('Force (N)')
                plt.legend()
                plt.savefig(f'forces_vs_time_{date}_test_{test_num}')

                plt.figure(3)
                plt.plot(plot_time, plot_position)
                plt.xlabel('Time (s)')
                plt.ylabel('Position (cm)')
                #plt.savefig(f'position_vs_time_{date}_test_{test_num}')

                plt.figure(4)
                plt.plot(plot_time, plot_force)
                plt.xlabel('Time (s)')
                plt.ylabel('Force (N)')
                #plt.savefig(f'force_vs_time_{date}_test_{test_num}')

                ### comment out below this line when doing normal test runs ###

                """
                plt.figure(5)
                plt.plot(time_doubly_smoothed, strain_ax_doubly_smoothed-cAx, label='Doubly Smoothed Ax Voltage')
                plt.plot(time_doubly_smoothed, strain_bx_doubly_smoothed-cBx, label='Doubly Smoothed Bx Voltage')
                plt.plot(time_doubly_smoothed, strain_ay_doubly_smoothed-cAy, label='Doubly Smoothed Ay Voltage')
                plt.plot(time_doubly_smoothed, strain_by_doubly_smoothed-cBy, label='Doubly Smoothed By Voltage')
                plt.xlabel('Time (s)')
                plt.ylabel('Voltage (V)')
                plt.legend()
                plt.savefig(f'voltage_vs_time_{date}_test_{test_num}')

                plt.figure(6)
                plt.plot(plot_time, plot_stiffness, '.', label='Stiffness from Filtered Voltage Data')
                plt.plot(time_doubly_smoothed, stiffness_2, '.', label='Stiffness from Doubly Smoothed Voltage Data')
                plt.xlabel('Time (s)')
                plt.ylabel('Stiffness (N*m^2)')
                #plt.ylim(0, np.mean(stiffness) + 3*np.std(stiffness))
                plt.legend()
                plt.savefig(f'stiffness_vs_time_{date}_test_{test_num}')

                plt.show()
            
            if data[0] == " " or data[1] == " ":
                return
                        
            # Write the data into the next row in the csv file.
            time_sec = float(data[0])*10**-6 #convert time to seconds
            strain_ax = float(data[1])*supply_voltage/(resolution*gain)
            strain_bx = float(data[2])*supply_voltage/(resolution*gain)
            strain_ay = float(data[3])*supply_voltage/(resolution*gain)
            strain_by = float(data[4])*supply_voltage/(resolution*gain)

            if self.time_offset_check == True:
                self.time_offset = time_sec
                self.time_offset_check = False

            time_sec = time_sec - self.time_offset
            self.csvwriter.writerow([time_sec, strain_ax, strain_bx, strain_ay, strain_by, current_time])

            # Ensure data is written to the file immediately
            self.csvfile.flush()

            # only use data for the plot every now and then, as specifed in seconds by the variable increment
            increment = .05
            if time_sec - self.plot_time > increment:

                self.time_sec = np.append(self.time_sec, time_sec)
                self.strain_ax = np.append(self.strain_ax, strain_ax)
                self.strain_bx = np.append(self.strain_bx, strain_bx)
                self.strain_ay = np.append(self.strain_ay, strain_ay)
                self.strain_by = np.append(self.strain_by, strain_by)

                # Update plot with new data
                self.curve_ax.setData(self.time_sec, self.strain_ax)
                self.curve_bx.setData(self.time_sec, self.strain_bx)
                self.curve_ay.setData(self.time_sec, self.strain_ay) 
                self.curve_by.setData(self.time_sec, self.strain_by)

                # Set x-axis range to show the last 4 seconds of data
                if len(self.time_sec) > 0:
                    self.plot_x.setXRange(max(0, self.time_sec[-1] - 4), self.time_sec[-1])
                    self.plot_y.setXRange(max(0, self.time_sec[-1] - 4), self.time_sec[-1])

                self.plot_time = time_sec

            
        # if control+c is pressed, the program ends. 
        except KeyboardInterrupt:  
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            return  # Exit the loop on Ctrl+C

        except Exception as e:
            print(f"Error: {e}")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            return


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimePlotWindow()
    sys.exit(app.exec_())
    