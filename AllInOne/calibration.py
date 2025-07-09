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
            strains = {'a1': [], 'b1': [], 'a2': [], 'b2': []}

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

                headers = ['Time', 'Strain A1', 'Strain B1', 'Strain A2', 'Strain B2', 'Current Time']
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
                            strain_a1 = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_b1 = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_a2 = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_b2 = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                        except (ValueError, IndexError):
                            status_queue.put("Invalid data received: cannot parse values")
                            continue

                        if time_offset_check:
                            time_offset = time_sec
                            time_offset_check = False

                        time_sec -= time_offset
                        csvwriter.writerow([time_sec, strain_a1, strain_b1, strain_a2, strain_b2, current_time])
                        csvfile.flush()

                        strains['a1'].append(strain_a1)
                        strains['b1'].append(strain_b1)
                        strains['a2'].append(strain_a2)
                        strains['b2'].append(strain_b2)

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
                'a1': np.mean(strains['a1']) if strains['a1'] else 0,
                'b1': np.mean(strains['b1']) if strains['b1'] else 0,
                'a2': np.mean(strains['a2']) if strains['a2'] else 0,
                'b2': np.mean(strains['b2']) if strains['b2'] else 0
            }
            summary_data.append([mass, position, avg_strains['a1'], avg_strains['b1'], avg_strains['a2'], avg_strains['b2']])

    with open(summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        pre_test_notes = [
            ["=================================================================="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)

        headers = ['Mass (g)', 'Position (cm)', 'Avg Strain A1', 'Avg Strain B1', 'Avg Strain A2', 'Avg Strain B2']
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
                strain_a1 = float(row[3])   #2
                strain_b1 = float(row[3])   #3
                strain_a2 = float(row[5])   #4
                strain_b2 = float(row[5])   #5
                valid_data.append((mass, position, strain_a1, strain_b1, strain_a2, strain_b2))
            except ValueError:
                continue

        if not valid_data:
            cal_status_var.set("Status: No valid data for calculation")
            return ""

        # Unpack filtered data
        masses, positions, strains_a1, strains_b1, strains_a2, strains_b2 = zip(*valid_data)
        masses = np.array(masses)
        positions = np.array(positions)
        strains_a1 = np.array(strains_a1)
        strains_b1 = np.array(strains_b1)
        strains_a2 = np.array(strains_a2)
        strains_b2 = np.array(strains_b2)

        # Match units
        masses = masses * 1e-3  # g to kg
        positions = positions * 1e-2  # cm to m
        g = 9.80  # m/s^2
        forces = masses * g  # Newtons

        # Create design matrix A = [F*x, -F] for multiple linear regression
        A = np.column_stack((forces * positions, -forces))

        # Fit models for each strain type: V = c + k*(F*x) + beta2*(-F), where beta2 = k*d
        model_a1 = LinearRegression().fit(A, strains_a1)
        c_a1 = model_a1.intercept_
        k_a1 = model_a1.coef_[0]
        beta2_a1 = model_a1.coef_[1]
        d_a1 = beta2_a1 / k_a1 if k_a1 != 0 else 0

        model_b1 = LinearRegression().fit(A, strains_b1)
        c_b1 = model_b1.intercept_
        k_b1 = model_b1.coef_[0]
        beta2_b1 = model_b1.coef_[1]
        d_b1 = beta2_b1 / k_b1 if k_b1 != 0 else 0

        model_a2 = LinearRegression().fit(A, strains_a2)
        c_a2 = model_a2.intercept_
        k_a2 = model_a2.coef_[0]
        beta2_a2 = model_a2.coef_[1]
        d_a2 = beta2_a2 / k_a2 if k_a2 != 0 else 0

        model_b2 = LinearRegression().fit(A, strains_b2)
        c_b2 = model_b2.intercept_
        k_b2 = model_b2.coef_[0]
        beta2_b2 = model_b2.coef_[1]
        d_b2 = beta2_b2 / k_b2 if k_b2 != 0 else 0

        # Format result
        result = (f"A1: k={k_a1:.6f}, d={d_a1:.6f}, c={c_a1:.6f}\n"
                  f"B1: k={k_b1:.6f}, d={d_b1:.6f}, c={c_b1:.6f}\n"
                  f"A2: k={k_a2:.6f}, d={d_a2:.6f}, c={c_a2:.6f}\n"
                  f"B2: k={k_b2:.6f}, d={d_b2:.6f}, c={c_b2:.6f}")

        # Optional plotting for A1 strain
        V_a1_pred = model_a1.predict(A)
        V_a2_pred = model_a2.predict(A)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(forces * positions, strains_a1, label='Original')
        plt.scatter(forces * positions, V_a1_pred, label='Predicted')
        plt.legend()

        plt.figure(2)
        plt.scatter(forces * positions, strains_a2, label='Original')
        plt.scatter(forces * positions, V_a2_pred, label='Predicted')
        plt.legend()
        plt.show()

        return result

    except Exception as e:
        cal_status_var.set(f"Status: Error calculating coefficients: {str(e)}")
        return ""