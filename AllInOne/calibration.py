import serial
import csv
import keyboard
import numpy as np
from datetime import datetime
from multiprocessing import Queue
import time
import os
from sklearn.linear_model import LinearRegression

# Constants
SUPPLY_VOLTAGE = 5
RESOLUTION = 2**9
GAIN = 16

def run_calibration(port, config, status_queue):
    """Run calibration by collecting strain data for each mass-position pair.

    Args:
        port (str): Serial port for Arduino communication.
        config (dict): Configuration dictionary with calibration parameters.
        status_queue (Queue): Queue to send status messages to the UI.
    """
    parent_folder = 'Raw Data'
    os.makedirs(parent_folder, exist_ok=True)
    summary_path = os.path.join(parent_folder, f'{config["date"]}_calibration_summary.csv')
    summary_data = []

    masses = config["masses"]
    positions = config["positions"]

    for mass in masses:
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
                    status_queue.put(f"Press 'space' to start collection for mass {mass}g at {position}cm")
                if incoming_data == "#" or keyboard.is_pressed('space'):
                    status_queue.put(f"Starting data collection for mass {mass}g at {position}cm")
                    break

            csv_path = os.path.join(parent_folder, f'{config["date"]}_calibration_mass_{mass}_pos_{position}.csv')
            strains = {'ax': [], 'bx': [], 'ay': [], 'by': []}

            with open(csv_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                pre_test_notes = [
                    ["pre_test_note_1", '', '', '', '', ''],
                    ["mass (g)", mass, '', '', '', ''],
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
                            status_queue.put(f"Data collection ended for mass {mass}g at {position}cm")
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

                        status_queue.put(f"Press 'space' to end data collection for mass {mass}g at {position}cm")

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
            summary_data.append([mass, position, avg_strains['ax'], avg_strains['bx'], avg_strains['ay'], avg_strains['by']])

    with open(summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        pre_test_notes = [
            ["=================================================================="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)

        headers = ['Mass (g)', 'Position (cm)', 'Avg Strain Ax', 'Avg Strain Bx', 'Avg Strain Ay', 'Avg Strain By']
        csvwriter.writerow(headers)
        for row in summary_data:
            csvwriter.writerow(row)

    status_queue.put("Calibration ended")

def calculate_coefficients(calibration_data, cal_status_var):
    """Calculate calibration coefficients using multiple linear regression to directly fit k, d, and c.

    Args:
        calibration_data (list): List of rows from the calibration summary CSV.
        cal_status_var (tk.StringVar): Tkinter variable to update the UI status.

    Returns:
        str: Formatted string of calculated coefficients.
    """
    if not calibration_data:
        cal_status_var.set("Status: No calibration data loaded")
        return ""

    try:
        # Extract and filter data, ensuring all columns are valid numbers
        valid_data = []
        for row in calibration_data:
            try:
                mass = float(row[0])
                position = float(row[1])
                strain_ax = float(row[2])
                strain_bx = float(row[3])
                strain_ay = float(row[4])
                strain_by = float(row[5])
                valid_data.append((mass, position, strain_ax, strain_bx, strain_ay, strain_by))
            except ValueError:
                continue

        if not valid_data:
            cal_status_var.set("Status: No valid data for calculation")
            return ""

        # Unpack filtered data
        masses, positions, strains_ax, strains_bx, strains_ay, strains_by = zip(*valid_data)
        masses = np.array(masses)
        positions = np.array(positions)
        strains_ax = np.array(strains_ax)
        strains_bx = np.array(strains_bx)
        strains_ay = np.array(strains_ay)
        strains_by = np.array(strains_by)

        # Match units
        masses = masses * 1e-3  # g to kg
        positions = positions * 1e-2  # cm to m
        g = 9.80  # m/s^2
        forces = masses * g  # Newtons

        # Create design matrix A = [F*x, -F] for multiple linear regression
        A = np.column_stack((forces * positions, -forces))

        # Fit models for each strain type: V = c + k*(F*x) + beta2*(-F), where beta2 = k*d
        model_ax = LinearRegression().fit(A, strains_ax)
        c_ax = model_ax.intercept_
        k_ax = model_ax.coef_[0]
        beta2_ax = model_ax.coef_[1]
        d_ax = beta2_ax / k_ax if k_ax != 0 else 0

        model_bx = LinearRegression().fit(A, strains_bx)
        c_bx = model_bx.intercept_
        k_bx = model_bx.coef_[0]
        beta2_bx = model_bx.coef_[1]
        d_bx = beta2_bx / k_bx if k_bx != 0 else 0

        model_ay = LinearRegression().fit(A, strains_ay)
        c_ay = model_ay.intercept_
        k_ay = model_ay.coef_[0]
        beta2_ay = model_ay.coef_[1]
        d_ay = beta2_ay / k_ay if k_ay != 0 else 0

        model_by = LinearRegression().fit(A, strains_by)
        c_by = model_by.intercept_
        k_by = model_by.coef_[0]
        beta2_by = model_by.coef_[1]
        d_by = beta2_by / k_by if k_by != 0 else 0

        # Format result
        result = (f"Ax: k={k_ax:.6f}, d={d_ax:.6f}, c={c_ax:.6f}\n"
                  f"Bx: k={k_bx:.6f}, d={d_bx:.6f}, c={c_bx:.6f}\n"
                  f"Ay: k={k_ay:.6f}, d={d_ay:.6f}, c={c_ay:.6f}\n"
                  f"By: k={k_by:.6f}, d={d_by:.6f}, c={c_by:.6f}")

        # Optional plotting for Ax strain
        V_ax_pred = model_ax.predict(A)
        V_ay_pred = model_ay.predict(A)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(forces * positions, strains_ax, label='Original')
        plt.scatter(forces * positions, V_ax_pred, label='Predicted')
        plt.legend()

        plt.figure(2)
        plt.scatter(forces * positions, strains_ay, label='Original')
        plt.scatter(forces * positions, V_ay_pred, label='Predicted')
        plt.legend()
        plt.show()

        return result

    except Exception as e:
        cal_status_var.set(f"Status: Error calculating coefficients: {str(e)}")
        return ""