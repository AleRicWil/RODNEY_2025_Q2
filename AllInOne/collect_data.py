import serial
import csv
import keyboard
import numpy as np
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
    """Class to handle real-time strain data collection and plotting from an Arduino.

    Attributes:
        ser (serial.Serial): Serial connection to the Arduino.
        csvfile (file): CSV file for data storage.
        csvwriter (csv.writer): Writer for CSV data.
        win (pg.GraphicsLayoutWidget): PyQtGraph window for plotting.
        plot_x (pg.PlotItem): Plot for X-axis strains.
        plot_y (pg.PlotItem): Plot for Y-axis strains.
        curve_ax (pg.PlotDataItem): Plot curve for Ax strain.
        curve_bx (pg.PlotDataItem): Plot curve for Bx strain.
        curve_ay (pg.PlotDataItem): Plot curve for Ay strain.
        curve_by (pg.PlotDataItem): Plot curve for By strain.
        time_sec (np.ndarray): Array of time values.
        strain_ax (np.ndarray): Array of Ax strain values.
        strain_bx (np.ndarray): Array of Bx strain values.
        strain_ay (np.ndarray): Array of Ay strain values.
        strain_by (np.ndarray): Array of By strain values.
        time_offset (float): Time offset for data collection.
        time_offset_check (bool): Flag to set initial time offset.
        plot_time (float): Last time plotted to control update frequency.
        timer (QtCore.QTimer): Timer for updating plots.
    """

    def __init__(self, port, config, status_queue):
        """Initialize the plot window and serial connection.

        Args:
            port (str): Serial port for Arduino communication.
            config (dict): Configuration dictionary with test parameters.
            status_queue (Queue): Queue to send status messages to the UI.
        """
        super().__init__()
        self.status_queue = status_queue

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
        parent_folder = 'Raw Data'
        os.makedirs(parent_folder, exist_ok=True)

        # Open CSV file in the parent folder
        csv_path = os.path.join(parent_folder, f'{config["date"]}_test_{config["test_num"]}.csv')
        self.csvfile = open(csv_path, 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile)

        pre_test_notes = [
            ["pre_test_note_1", '', '', '', '', ''],
            ["rodney configuration", config["configuration"], '', '', '', ''],
            ["medium stiffness (Nm^2)", config["pvc_stiffness"], '', '', '', ''],
            ["height (cm)", config["height"], '', '', '', ''],
            ["yaw angle (degrees)", config["yaw"], '', '', '', ''],
            ["pitch angle (degrees)", config["pitch"], '', '', '', ''],
            ["roll angle (degrees)", config["roll"], '', '', '', ''],
            ["rate of travel (ft/min)", config["rate_of_travel"], '', '', '', ''],
            ["angle of travel (degrees)", config["angle_of_travel"], '', '', '', ''],
            ["offset distance (cm)", config["offset_distance"], '', '', '', ''],
            ["=================================================================="]
        ]
        for note in pre_test_notes:
            self.csvwriter.writerow(note)

        headers = ['Time', 'Strain Ax', 'Strain Bx', 'Strain Ay', 'Strain By', 'Current Time']
        self.csvwriter.writerow(headers)

        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget(show=True, title=f"Strain Data Plots - Test {config['test_num']}")
        self.win.resize(1500, 1000)
        self.win.move(0, 0)
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

        self.time_offset_check = True
        self.time_offset = 0
        self.plot_time = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def update_plot(self):
        """Update the plot with new data from the serial port."""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            data = line.split(',')

            if len(data) < 5:
                self.status_queue.put("Invalid data received: incomplete data packet")
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
                self.win.close()
                return

            if data[0] == " " or data[1] == " ":
                return

            try:
                time_sec = float(data[0]) * 10**-6
                strain_ax = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_bx = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_ay = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                strain_by = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
            except (ValueError, IndexError):
                self.status_queue.put("Invalid data received: cannot parse values")
                return

            if self.time_offset_check:
                self.time_offset = time_sec
                self.time_offset_check = False

            time_sec -= self.time_offset
            self.csvwriter.writerow([time_sec, strain_ax, strain_bx, strain_ay, strain_by, current_time])
            self.csvfile.flush()

            increment = 0.05
            if time_sec - self.plot_time > increment:
                self.time_sec = np.append(self.time_sec, time_sec)
                self.strain_ax = np.append(self.strain_ax, strain_ax)
                self.strain_bx = np.append(self.strain_bx, strain_bx)
                self.strain_ay = np.append(self.strain_ay, strain_ay)
                self.strain_by = np.append(self.strain_by, strain_by)

                self.curve_ax.setData(self.time_sec, self.strain_ax)
                self.curve_bx.setData(self.time_sec, self.strain_bx)
                self.curve_ay.setData(self.time_sec, self.strain_ay)
                self.curve_by.setData(self.time_sec, self.strain_by)

                if len(self.time_sec) > 0:
                    self.plot_x.setXRange(max(0, self.time_sec[-1] - 4), self.time_sec[-1])
                    self.plot_y.setXRange(max(0, self.time_sec[-1] - 4), self.time_sec[-1])

                self.plot_time = time_sec

            self.status_queue.put(f"Press 'space' to end data collection")

        except KeyboardInterrupt:
            self.status_queue.put("Interrupted by user")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win.close()
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
            self.ser.close()
            self.csvfile.close()
            self.timer.stop()
            self.win.close()

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