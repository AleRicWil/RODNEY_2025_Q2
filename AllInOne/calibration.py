# Name: Christian Shamo
# Date: 06/24/2025
# Description: This program handles calibration for an Arduino Uno that prints strain data for 4 wheatstone half-bridges,
#              separated by commas, to a serial port. The user inputs weights, positions, and a date once in the UI.
#              The program collects data for each weight-position pair in sequence (all positions for one weight,
#              then the next weight), saving each collection to a separate CSV file in a 'Raw Data' folder (e.g.,
#              Raw Data/MM_DD_calibration_weight_X_pos_Y.csv). The average strain for each pair is stored in a summary
#              CSV file (e.g., Raw Data/MM_DD_calibration_summary.csv). The process mimics the data collection in
#              collect_data.py but without real-time plotting.
#
# Dynamic Inputs:
#         - date (MM_DD)
#         - weights (list of grams)
#         - positions (list of cm)
#         - COM port (selected in UI; must match the Arduino program)
# Other Inputs:
#         - Incoming data from the serial port, printed by an Arduino Uno.
#         - Baud rate (must match the Arduino program)
#
# Outputs:
#         - Multiple .csv files in 'Raw Data', one per weight-position pair, containing strain data.
#         - A summary .csv file with average strains for each weight-position pair.

import serial
import csv
import keyboard
import numpy as np
from datetime import datetime
from multiprocessing import Queue
import time
import os

# Constants
SUPPLY_VOLTAGE = 5
RESOLUTION = 2**9
GAIN = 16

def run_calibration(port, config, status_queue):
    """Run calibration by collecting strain data for each weight-position pair.

    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with calibration parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    parent_folder = 'Raw Data'
    os.makedirs(parent_folder, exist_ok=True)
    summary_path = os.path.join(parent_folder, f'{config["date"]}_calibration_summary.csv')
    summary_data = []

    weights = config["weights"]
    positions = config["positions"]

    for weight in weights:
        for position in positions:
            try:
                ser = serial.Serial(port, 115200, timeout=1)
            except serial.SerialException as e:
                status_queue.put(f"Failed to connect to {port}: {str(e)}")
                return

            count = 0
            while True:
                count += 1
                incoming_data = ser.readline().decode('utf-8', errors='ignore').strip()
                if count <= 1:
                    time.sleep(2)
                    status_queue.put(f"Press 'space' to start collection for weight {weight}g at {position}cm")
                if incoming_data == "#" or keyboard.is_pressed('space'):
                    status_queue.put(f"Starting data collection for weight {weight}g at {position}cm")
                    break

            csv_path = os.path.join(parent_folder, f'{config["date"]}_calibration_weight_{weight}_pos_{position}.csv')
            strains = {'ax': [], 'bx': [], 'ay': [], 'by': []}

            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                pre_test_notes = [
                    ["pre_test_note_1", '', '', '', '', ''],
                    ["weight (g)", weight, '', '', '', ''],
                    ["position (cm)", position, '', '', '', ''],
                    ["=================================================================="]
                ]
                for note in pre_test_notes:
                    csvwriter.writerow(note)

                headers = ['Time', 'Strain Ax', 'Strain Bx', 'Strain Ay', 'Strain By', 'Current Time']
                csvwriter.writerow(headers)

                time_offset_check = True
                time_offset = 0

                time.sleep(1)
                while True:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        data = line.split(',')

                        if len(data) < 5:
                            status_queue.put("Invalid data received: incomplete data packet")
                            continue

                        if data[0] == "$":
                            status_queue.put(f"Reset at {float(data[1])*10**-6}")
                            continue

                        now = datetime.now()
                        current_time = now.time()

                        if data[0] == "test ended" or keyboard.is_pressed('space'):
                            status_queue.put(f"Data collection ended for weight {weight}g at {position}cm")
                            break

                        if data[0] == " " or data[1] == " ":
                            continue

                        try:
                            time_sec = float(data[0]) * 10**-6
                            strain_ax = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_bx = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_ay = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_by = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                        except (ValueError, IndexError):
                            status_queue.put("Invalid data received: cannot parse values")
                            continue

                        if time_offset_check:
                            time_offset = time_sec
                            time_offset_check = False

                        time_sec -= time_offset
                        csvwriter.writerow([time_sec, strain_ax, strain_bx, strain_ay, strain_by, current_time])
                        csvfile.flush()

                        strains['ax'].append(strain_ax)
                        strains['bx'].append(strain_bx)
                        strains['ay'].append(strain_ay)
                        strains['by'].append(strain_by)

                        status_queue.put(f"Press 'space' to end data collection for weight {weight}g at {position}cm")

                    except KeyboardInterrupt:
                        status_queue.put("Interrupted by user")
                        ser.close()
                        break
                    except Exception as e:
                        status_queue.put(f"Error: {str(e)}")
                        ser.close()
                        break

                ser.close()

            avg_strains = {
                'ax': np.mean(strains['ax']) if strains['ax'] else 0,
                'bx': np.mean(strains['bx']) if strains['bx'] else 0,
                'ay': np.mean(strains['ay']) if strains['ay'] else 0,
                'by': np.mean(strains['by']) if strains['by'] else 0
            }
            summary_data.append([weight, position, avg_strains['ax'], avg_strains['bx'], avg_strains['ay'], avg_strains['by']])

    with open(summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        pre_test_notes = [
            ["=================================================================="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)

        headers = ['Weight (g)', 'Position (cm)', 'Avg Strain Ax', 'Avg Strain Bx', 'Avg Strain Ay', 'Avg Strain By']
        csvwriter.writerow(headers)
        for row in summary_data:
            csvwriter.writerow(row)

    status_queue.put("Calibration ended")