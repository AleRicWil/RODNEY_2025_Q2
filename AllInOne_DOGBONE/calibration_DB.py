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
    parent_folder = os.path.join('Raw Data_DOGBONE', f'{config["date"]}')
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
            strains = {'1': [], '11': [], '2': [], '3': []}

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

                headers = ['Time', 'Strain 1', 'Strain 11', 'Strain 2', 'Strain 3', 'Current Time']
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
                            strain_yeet = float(data[1]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_1 = float(data[2]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_2 = float(data[3]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                            strain_3 = float(data[4]) * SUPPLY_VOLTAGE / (RESOLUTION * GAIN)
                        except (ValueError, IndexError):
                            status_queue.put("Invalid data received: cannot parse values")
                            continue

                        if time_offset_check:
                            time_offset = time_sec
                            time_offset_check = False

                        time_sec -= time_offset
                        csvwriter.writerow([time_sec, strain_1, strain_1, strain_2, strain_3, current_time])
                        csvfile.flush()

                        strains['1'].append(strain_1)
                        strains['11'].append(strain_1)
                        strains['2'].append(strain_2)
                        strains['3'].append(strain_3)

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
                '1': np.mean(strains['1']) if strains['1'] else 0,
                '11': np.mean(strains['11']) if strains['11'] else 0,
                '2': np.mean(strains['2']) if strains['2'] else 0,
                '3': np.mean(strains['3']) if strains['3'] else 0
            }
            summary_data.append([mass, position, avg_strains['1'], avg_strains['11'], avg_strains['2'], avg_strains['3']])

    with open(summary_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        pre_test_notes = [
            ["=================================================================="]
        ]
        for note in pre_test_notes:
            csvwriter.writerow(note)

        headers = ['Mass (g)', 'Position (cm)', 'Avg Strain 1', 'Avg Strain 11', 'Avg Strain 2', 'Avg Strain 3']
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
                strain_1 = float(row[2])   #2
                strain_11 = float(row[3])   #3
                strain_2 = float(row[4])   #4
                strain_3 = float(row[5])   #5
                valid_data.append((mass, position, strain_1, strain_11, strain_2, strain_3))
            except ValueError:
                continue

        if not valid_data:
            cal_status_var.set("Status: No valid data for calculation")
            return ""

        # Unpack filtered data
        masses, positions, strains_1, strains_11, strains_2, strains_3 = zip(*valid_data)
        masses = np.array(masses)
        positions = np.array(positions)
        strains_1 = np.array(strains_1)
        strains_11 = np.array(strains_11)
        strains_2 = np.array(strains_2)
        strains_3 = np.array(strains_3)

        # Match units
        masses = masses * 1e-3  # g to kg
        positions = positions * 1e-2  # cm to m
        g = 9.80  # m/s^2
        forces = masses * g  # Newtons

        # Create design matrix A = [F*x, -F] for multiple linear regression
        A = np.column_stack((forces * positions, -forces))

        # Fit models for each strain type: V = c + k*(F*x) + beta2*(-F), where beta2 = k*d
        model_1 = LinearRegression().fit(A, strains_1)
        c_1 = model_1.intercept_
        k_1 = model_1.coef_[0]
        beta2_1 = model_1.coef_[1]
        d_1 = beta2_1 / k_1 if k_1 != 0 else 0

        model_11 = LinearRegression().fit(A, strains_11)
        c_11 = model_11.intercept_
        k_11 = model_11.coef_[0]
        beta2_11 = model_11.coef_[1]
        d_11 = beta2_11 / k_11 if k_11 != 0 else 0

        model_2 = LinearRegression().fit(A, strains_2)
        c_2 = model_2.intercept_
        k_2 = model_2.coef_[0]
        beta2_2 = model_2.coef_[1]
        d_2 = beta2_2 / k_2 if k_2 != 0 else 0

        model_3 = LinearRegression().fit(A, strains_3)
        c_3 = model_3.intercept_
        k_3 = model_3.coef_[0]
        beta2_3 = model_3.coef_[1]
        d_3 = beta2_3 / k_3 if k_3 != 0 else 0

        # Format result
        result = (f"1: k={k_1:.6f}, d={d_1:.6f}, c={c_1:.6f}\n"
                  f"11: k={k_11:.6f}, d={d_11:.6f}, c={c_11:.6f}\n"
                  f"2: k={k_2:.6f}, d={d_2:.6f}, c={c_2:.6f}\n"
                  f"3: k={k_3:.6f}, d={d_3:.6f}, c={c_3:.6f}")

        # Optional plotting for A1 strain
        V_1_pred = model_1.predict(A)
        V_2_pred = model_2.predict(A)
        V_3_pred = model_3.predict(A)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(forces * positions, strains_1, label='Original')
        plt.scatter(forces * positions, V_1_pred, label='Predicted')
        plt.xlabel('Force*Position')
        plt.ylabel('Voltage')
        plt.legend()

        plt.figure(2)
        plt.scatter(forces * positions, strains_2, label='Original')
        plt.scatter(forces * positions, V_2_pred, label='Predicted')
        plt.xlabel('Force*Position')
        plt.ylabel('Voltage')
        plt.legend()

        plt.figure(3)
        plt.scatter(forces * positions, strains_3, label='Original')
        plt.scatter(forces * positions, V_3_pred, label='Predicted')
        plt.xlabel('Force*Position')
        plt.ylabel('Voltage')
        plt.legend()
        plt.show()

        return result

    except Exception as e:
        cal_status_var.set(f"Status: Error calculating coefficients: {str(e)}")
        return ""
    
    