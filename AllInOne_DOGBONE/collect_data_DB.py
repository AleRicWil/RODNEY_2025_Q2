import serial
import csv
import keyboard
import numpy as np
import pandas as pd
from datetime import datetime
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from multiprocessing import Queue
import time
import os

# Constants
SUPPLY_VOLTAGE = 5
RESOLUTION = 2**9
GAIN = 16

class RealTimePlotWindow(QtWidgets.QMainWindow):
    """Class to handle real-time strain, force, and position data collection and plotting from an Arduino.

    Attributes:
        ser (serial.Serial): Serial connection to the Arduino.
        csvfile (file): CSV file for data storage.
        csvwriter (csv.writer): Writer for CSV data.
        win_strain (pg.GraphicsLayoutWidget): PyQtGraph window for strain plotting.
        win_force_pos (pg.GraphicsLayoutWidget): PyQtGraph window for force and position plotting.
        plot_x (pg.PlotItem): Plot for A1 & B1 strains.
        plot_y (pg.PlotItem): Plot for A2 & B2 strains.
        plot_force (pg.PlotItem): Plot for force.
        plot_pos (pg.PlotItem): Plot for position.
        curve_a1 (pg.PlotDataItem): Plot curve for A1 strain.
        curve_b1 (pg.PlotDataItem): Plot curve for B1 strain.
        curve_a2 (pg.PlotDataItem): Plot curve for A2 strain.
        curve_b2 (pg.PlotDataItem): Plot curve for B2 strain.
        curve_force (pg.PlotDataItem): Plot curve for force.
        curve_pos (pg.PlotDataItem): Plot curve for position.
        time_sec (np.ndarray): Array of time values.
        strain_a1 (np.ndarray): Array of A1 strain values.
        strain_b1 (np.ndarray): Array of B1 strain values.
        strain_a2 (np.ndarray): Array of A2 strain values.
        strain_b2 (np.ndarray): Array of B2 strain values.
        force (np.ndarray): Array of force values.
        position (np.ndarray): Array of position values.
        time_offset (float): Time offset for data collection.
        time_offset_check (bool): Flag to set initial time offset.
        plot_time (float): Last time plotted to control update frequency.
        timer (QtCore.QTimer): Timer for updating plots.
        k_A1, k_B1, k_A2, k_B2, d_A1, d_B1, d_A2, d_B2, c_A1, c_B1, c_A2, c_B2: Calibration coefficients.
    """

    def __init__(self, port, config, status_queue):
        """Initialize the plot windows and serial connection.

        Args:
            port (str): Serial port for Arduino communication.
            config (dict): Configuration dictionary with test parameters.
            status_queue (Queue): Queue to send status messages to the UI.
        """
        super().__init__()
        self.status_queue = status_queue

        # Load calibration coefficients
        cal_csv_path = r'AllInOne_DOGBONE\calibration_history.csv'
        try:
            cal_data = pd.read_csv(cal_csv_path)
            latest_cal = cal_data.iloc[-1]
            self.k_1 = latest_cal['k_1']
            self.k_11 = latest_cal['k_11']
            self.k_2 = latest_cal['k_2']
            self.k_3 = latest_cal['k_3']
            self.d_1 = latest_cal['d_1']
            self.d_11 = latest_cal['d_11']
            self.d_2 = latest_cal['d_2']
            self.d_3 = latest_cal['d_3']
            self.c_1 = latest_cal['c_1']
            self.c_11 = latest_cal['c_11']
            self.c_2 = latest_cal['c_2']
            self.c_3 = latest_cal['c_3']
        except Exception as e:
            self.status_queue.put(f"Error loading calibration: {str(e)}")
            self.ser.close()
            return

        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
        except serial.SerialException as e:
            self.status_queue.put(f"Failed to connect to {port}: {str(e)}")
            return

        count = 0
        while True:
            count += 1
            incoming_data = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if count <= 1:
                time.sleep(2)
                self.status_queue.put("Press 'space' to sync with Arduino")
            if incoming_data == "#" or keyboard.is_pressed('space'):
                self.status_queue.put("Starting data collection")
                break

        # Create parent folder based on date
        parent_folder = os.path.join('Raw Data_DOGBONE', f'{config["date"]}')
        os.makedirs(parent_folder, exist_ok=True)

        # Open CSV file in the parent folder
        csv_path = os.path.join(parent_folder, f'{config["date"]}_test_{config["test_num"]}.csv')
        self.csvfile = open(csv_path, 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        pre_test_notes = [
            ["user_note", '', '', '', '', ''],
            ["rodney configuration", config["configuration"], '', '', '', ''],
            ["sensor calibration (k d c)", f'{self.k_1} {self.k_11} {self.k_2} {self.k_3}', f'{self.d_1} {self.d_11} {self.d_2} {self.d_3}', f'{self.c_1} {self.c_11} {self.c_2} {self.c_3}', '', ''],
            ["stalk array (lo med hi)", config["pvc_stiffness"], '', '', '', ''],
            ["sensor height (cm)", config["height"], '', '', '', ''],
            ["sensor yaw (degrees)", config["yaw"], '', '', '', ''],
            ["sensor pitch (degrees)", config["pitch"], '', '', '', ''],
            ["sensor roll (degrees)", config["roll"], '', '', '', ''],
            ["rate of travel (ft/min)", config["rate_of_travel"], '', '', '', ''],
            ["angle of travel (degrees)", config["angle_of_travel"], '', '', '', ''],
            ["sensor offset (cm to gauge 2)", config["offset_distance"], '', '', '', ''],
            ["====="]
        ]
        for note in pre_test_notes:
            self.csvwriter.writerow(note)

        headers = ['Time', 'Strain 1', 'Strain 11', 'Strain 2', 'Strain 3', 'Current Time']
        self.csvwriter.writerow(headers)

        pg.setConfigOptions(antialias=True)
        # Strain plot window (right)
        self.win_strain = pg.GraphicsLayoutWidget(show=True, title=f"Strain Data Plots - Test {config['test_num']}")
        self.win_strain.resize(1000, 1000)
        self.win_strain.move(1000, 0)
        self.plot_x = self.win_strain.addPlot(title='1 Strain Vs Time')
        self.curve_1 = self.plot_x.plot(pen='green', name='1 Strain')
        self.plot_y = self.win_strain.addPlot(title='2(blue) & 3(yellow) Strain Vs Time')
        self.curve_2 = self.plot_y.plot(pen='blue', name='2 Strain')
        self.curve_3 = self.plot_y.plot(pen='yellow', name='3 Strain')

        # Force and position plot window (left)
        self.win_force_pos = pg.GraphicsLayoutWidget(show=True, title=f"Force and Position - Test {config['test_num']}")
        self.win_force_pos.resize(1000, 1000)
        self.win_force_pos.move(0, 0)
        self.plot_force = self.win_force_pos.addPlot(title='Force Vs Time')
        self.curve_force = self.plot_force.plot(pen='g', name='Force')
        self.plot_pos = self.win_force_pos.addPlot(title='Position Vs Time')
        self.curve_pos = self.plot_pos.plot(pen='y', name='Position')

        self.time_sec = []
        self.strain_yeet = []
        self.strain_1 = []
        self.strain_2 = []
        self.strain_3 = []
        self.force = []
        self.position = []

        self.time_offset_check = True
        self.time_offset = 0
        self.plot_time = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def update_plot(self):
        """Update the strain, force, and position plots with new data from the serial port."""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            data = line.split(',')

            if len(data) < 5:
                self.status_queue.put("Invalid data received: incomplete data packet veicolo")
                return

            if data[0] == "$":
                self.status_queue.put(f"Reset at {float(data[1])*10**-6}")
                return

            now = datetime.now()
            current_time = now.time()

            if data[0] == "test ended" or keyboard.is_pressed('space'):
                self.status_queue.put("Data collection ended")
                self.ser.close()
                self.csvfile.close()
                self.timer.stop()
                self.win_strain.close()
                self.win_force_pos.close()
                return

            if data[0] == " " or data[1] == " ":
                return

            try:
                time_sec = float(data[0]) * 10**-6
                strain_yeet = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)   #1
                strain_1 = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)   #2
                strain_2 = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)   #3
                strain_3 = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)   #4
            except (ValueError, IndexError):
                self.status_queue.put("Invalid data received: cannot parse values")
                return

            if self.time_offset_check:
                self.time_offset = time_sec
                self.time_offset_check = False

            time_sec -= self.time_offset
            self.csvwriter.writerow([time_sec, strain_1, strain_1, strain_2, strain_3, current_time])
            self.csvfile.flush()

            # Calculate force and position
            force = (self.k_3 * (strain_1 - self.c_1) - self.k_1 * (strain_3 - self.c_3)) / (self.k_1 * self.k_3 * (self.d_3 - self.d_1))
            position = (self.k_3 * self.d_3 * (strain_1 - self.c_1) - self.k_1 * self.d_1 * (strain_3 - self.c_3)) / (self.k_3 * (strain_1 - self.c_1) - self.k_1 * (strain_3 - self.c_3))
            position = np.clip(position, -0.5, 0.5)

            increment = 0.05
            if time_sec - self.plot_time > increment:
                self.time_sec.append(time_sec)
                self.strain_1.append(strain_1)
                self.strain_2.append(strain_2)
                self.strain_3.append(strain_3)
                self.force.append(force)
                self.position.append(position * 100)  # Convert to cm

                # Update strain plots
                # self.curve_a1.setData(self.time_sec, self.strain_a1)
                self.curve_1.setData(self.time_sec, self.strain_1)
                self.curve_2.setData(self.time_sec, self.strain_2)
                self.curve_3.setData(self.time_sec, self.strain_3)

                # Update force and position plots
                self.curve_force.setData(self.time_sec, self.force)
                self.curve_pos.setData(self.time_sec, self.position)

                if len(self.time_sec) > 0:
                    self.plot_x.setXRange(max(0, self.time_sec[-1] - 5), self.time_sec[-1])
                    self.plot_y.setXRange(max(0, self.time_sec[-1] - 5), self.time_sec[-1])
                    self.plot_force.setXRange(max(0, self.time_sec[-1] - 5), self.time_sec[-1])
                    self.plot_pos.setXRange(max(0, self.time_sec[-1] - 5), self.time_sec[-1])

                self.plot_time = time_sec

            self.status_queue.put(f"Press 'space' to end data collection")

        except KeyboardInterrupt:
            self.status_queue.put("Interrupted by user")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win_strain.close()
            self.win_force_pos.close()
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win_strain.close()
            self.win_force_pos.close()

def run_collection(port, config, status_queue):
    """Run the real-time plot window in a separate process.

    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with test parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = RealTimePlotWindow(port, config, status_queue)
    app.exec_()