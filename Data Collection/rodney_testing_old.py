#Name: Christian Shamo
#Date: 07/01/2024
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
#Inputs: -Incoming data from the serial port, which is printed to the serial port from an Arduino Uno.  
#        -Com # (must be same as in the corresponding Arduino program)
#        -Baud rate (must be the same as in the corresponding Arduino program)
#
#Outputs: A .csv file containing the data from the serial port. The data is from between the 1st and 2nd times 
#         the push-button is pressed.


import serial
import sys
import csv
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# calibration coefficients, from 05/15/24
kAx =  -0.0406
kAy = 0.0408
kBx = -0.0407
kBy = 0.0405

dAx = 0.0182
dAy = 0.0653
dBx = 0.0645
dBy = 0.0187

#cAx = 0.0150
#cAy = -0.1353
#cBx = -0.0752
#cBy = -0.0086

cAx = 0.020752
cAy = -0.13306
cBx = -0.07263
cBy = -0.00061

height = 0.7 # measurement height (meters)
supply_voltage = 5
resolution = 2**9
gain = 16
date = "07_08"
test_num = 32


class RealTimePlotWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Configure the serial connection
        self.ser = serial.Serial('COM3', 115200)  # Adjust the 'COM' # to your serial port, and the baud rate

        while True:
            incoming_data = self.ser.readline().decode('utf-8').strip()
            if incoming_data == "#":
                print("Starting data collection.")
                break

        # Open a CSV file to write data
        self.csvfile = open(f'{date}_test_{test_num}.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        # Write a note in the first row
        note_prefix = "PRE_TEST_NOTE_1"
        note_suffix = "medium stiffness pvc, laptop charging, and filtering applied based on values' proximities to relevant 'c' offset values"
        self.csvwriter.writerow([note_prefix, note_suffix])

        # Write the headers in the next row
        headers = ['TIME (Microseconds)', 'Strain Ax', 'Strain Bx', 'Strain Ay', 'Strain By']
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
        self.curve_ay = self.plot_y.plot(pen='r', name='Ax Strain')
        self.curve_by = self.plot_y.plot(pen='b', name='Bx Strain')
        #self.plot.setLabel("left", "Strain")
        #self.plot.setLabel("bottom", "Time (s)")
        #self.plot.setXRange(1, 10)
        #self.plot.setYRange(-650, 650)

            
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

    def update_plot(self):
        
        try:
            line = self.ser.readline().decode('utf-8').strip()
            # separate the incoming data into columns after every comma
            data = line.split(',')

            # When the push-button is pushed a 2nd time, "test ended" is printed to the serial port in Arduino
            if data[0] == "test ended":
                print("The data collection has ended.")
                csv_path = self.csvfile.name
                self.ser.close()
                self.csvfile.close()
                self.timer.stop()

                csv_data = pd.read_csv(csv_path, skiprows=1)

                csv_time_sec = csv_data['TIME (Microseconds)'].to_numpy()
                csv_strain_ax = csv_data['Strain Ax'].to_numpy()
                csv_strain_bx = csv_data['Strain Bx'].to_numpy()
                csv_strain_ay = csv_data['Strain Ay'].to_numpy()
                csv_strain_by = csv_data['Strain By'].to_numpy()
    
                time_filtered = np.array([])
                strain_ax_filtered = np.array([])
                strain_bx_filtered = np.array([])
                strain_ay_filtered = np.array([])
                strain_by_filtered = np.array([])

                for i in range(len(csv_time_sec)):
                    if len(csv_time_sec) > 2:
                        if abs((csv_strain_ax[i] - cAx)/cAx) > .05 and \
                            abs((csv_strain_bx[i] - cBx)/cBx) > .05 and \
                            abs((csv_strain_ay[i] - cAy)/cAy) > .05 and \
                            abs((csv_strain_by[i] - cBy)/cBy) > .05:

                            time_filtered = np.append(time_filtered, csv_time_sec[i])
                            strain_ax_filtered = np.append(strain_ax_filtered, csv_strain_ax[i])
                            strain_bx_filtered = np.append(strain_bx_filtered, csv_strain_bx[i])
                            strain_ay_filtered = np.append(strain_ay_filtered, csv_strain_ay[i])
                            strain_by_filtered = np.append(strain_by_filtered, csv_strain_by[i])


                # Plot the force, position, and stiffness plots
                force_x = (kBx*(strain_ax_filtered - cAx) - kAx*(strain_bx_filtered - cBx))/(kAx*kBx*(dBx - dAx))
                force_y = -(kBy*(strain_ay_filtered - cAy) - kAy*(strain_by_filtered - cBy))/(kAy*kBy*(dBy - dAy))
                force = (force_x**2 + force_y**2)**(1/2)

                theta_0_deg = 90 - (abs(np.arctan(force_x/force_y) * 180/np.pi))
                theta_0 = theta_0_deg * np.pi/180

                position_x = (kBx*dBx*(strain_ax_filtered-cAx) - kAx*dAx*(strain_bx_filtered-cBx))/(kBx*(strain_ax_filtered-cAx) - kAx*(strain_bx_filtered-cBx)) * 100
                position_y = (kBy*dBy*(strain_ay_filtered-cAy) - kAy*dAy*(strain_by_filtered-cBy))/(kBy*(strain_ay_filtered-cAy) - kAy*(strain_by_filtered-cBy)) * 100
                #position = position_x * (90 - theta_0_deg)/90 + position_y * theta_0_deg/90

                print(position_x)
                

                position = np.array([])
                for i in range(len(position_x)):
                    if position_x[i] > position_y[i]:
                        position = np.append(position, position_x[i] * (1 + (theta_0_deg[i])/90))
                    
                    if position_y[i] > position_x[i]:
                        position = np.append(position, position_y[i] * (1 + (90 - theta_0_deg[i])/90))

                phi = theta_0 + np.pi/2
                n = -1/np.tan(phi)
                n_table = np.array([-5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
                c_theta_table = np.array([1.1788, 1.1971, 1.2119, 1.2293, 1.2322, 1.2323, 1.2348, 1.2385, 1.243, 1.2467, 1.2492, 1.2511, 1.2534, 1.2548, 1.2557, 1.257, 1.2578])
                c_theta = np.array([])
                gamma = np.array([])
                k_theta = np.array([])

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
                    
                    c_theta = np.append(c_theta, c_theta_calc)

                    if n[i] > 0.5 and n[i] <= 10:
                        gamma_calc = 0.841655 - 0.0067807*n[i] + 0.000438*n[i]**2
                    elif n[i] > -1.8316 and n[i] <= 0.5:
                        gamma_calc = 0.852144 - 0.0182867*n[i]
                    elif n[i] > -5 and n[i] <= -1.8316:
                        gamma_calc = 0.912364 + 0.0145928*n[i]
                    
                    gamma = np.append(gamma, gamma_calc)

                    if n[i] > -5 and n[i] <= -2.5:
                        k_theta_calc = 3.024112 + 0.12129*n[i] + 0.003169*n[i]**2
                    elif n[i] > -2.5 and n[i] <= -1:
                        k_theta_calc = 1.967657 - 2.616021*n[i] - 3.738166*n[i]**2 - 2.649437*n[i]**3 \
                                        - 0.891906*n[i]**4 - 0.113063*n[i]**5
                    elif n[i] > -1 and n[i] <= 10:
                        k_theta_calc = 2.654855 - 0.509896*10**-1*n[i] + 0.126749*10**-1*n[i]**2 \
                                        - 0.142039*10**-2*n[i]**3 + 0.584525*10**-4*n[i]**4
                    
                    k_theta = np.append(k_theta, k_theta_calc)



                capital_theta = theta_0 / c_theta

                L = height/(1-gamma*(1-np.cos(capital_theta)))
                b = gamma*L*np.sin(capital_theta)
                stiffness = (force*height + n*force*b)*L / (capital_theta*gamma*k_theta) # units of N*m^2


                plt.figure(1)
                plt.plot(time_filtered, position_x, 'r', label='X-Position Vs Time')
                plt.plot(time_filtered, position_y, 'b', label='Y-Position Vs Time')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (cm)')
                plt.legend()
                plt.savefig(f'positions_vs_time_{date}_test_{test_num}')

                plt.figure(2)
                plt.plot(time_filtered, force_x, 'r', label='X-Axis Force Vs Time')
                plt.plot(time_filtered, force_y, 'b', label='Y-Axis Force Vs Time')
                plt.xlabel('Time (s)')
                plt.ylabel('Force (N)')
                plt.legend()
                plt.savefig(f'forces_vs_time_{date}_test_{test_num}')

                plt.figure(3)
                plt.plot(time_filtered, position)
                plt.xlabel('Time (s)')
                plt.ylabel('Position (cm)')
                plt.savefig(f'position_vs_time_{date}_test_{test_num}')

                plt.figure(4)
                plt.plot(time_filtered, force)
                plt.xlabel('Time (s)')
                plt.ylabel('Force (N)')
                plt.savefig(f'force_vs_time_{date}_test_{test_num}')

                plt.figure(5)
                plt.plot(time_filtered, stiffness)
                plt.xlabel('Time (s)')
                plt.ylabel('Stiffness (N*m^2)')
                plt.savefig(f'stiffness_vs_time_{date}_test_{test_num}')

                plt.figure(6)
                plt.plot(time_filtered, theta_0_deg)
                plt.xlabel('Time (s)')
                plt.ylabel('Theta_0 (deg)')
                plt.savefig(f'theta_0_vs_time_{date}_test_{test_num}')
                plt.show()
                return
            
            if data[0] == " " or data[1] == " ":
                return

            #if len(self.strain_ax) > 2:
            #        if abs(float(data[1]))*supply_voltage/(resolution*gain) < 5*abs(self.strain_ax[-1]) and \
            #            abs(float(data[2]))*supply_voltage/(resolution*gain) < 5*abs(self.strain_bx[-1]) and \
            #                abs(float(data[3]))*supply_voltage/(resolution*gain) < 5*abs(self.strain_ay[-1]) and \
            #                    abs(float(data[4]))*supply_voltage/(resolution*gain) < 5*abs(self.strain_by[-1]):
                        
                        
            # Write the data into the next row in the csv file.
            time_sec = float(data[0])*10**-6 #convert time to seconds
            strain_ax = float(data[1])*supply_voltage/(resolution*gain)
            strain_bx = float(data[2])*supply_voltage/(resolution*gain)
            strain_ay = float(data[3])*supply_voltage/(resolution*gain)
            strain_by = float(data[4])*supply_voltage/(resolution*gain)
            self.csvwriter.writerow([time_sec, strain_ax, strain_bx, strain_ay, strain_by])

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
    