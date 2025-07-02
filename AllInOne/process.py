import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process_data(date, test_num):
    # Load calibration coefficients
    cal_csv_path = r'AllInOne\calibration_history.csv'
    cal_data = pd.read_csv(cal_csv_path)
    latest_cal = cal_data.iloc[-1]  # Get the most recent calibration
    k_A1 = latest_cal['k_A1']
    k_B1 = latest_cal['k_B1']
    k_A2 = latest_cal['k_A2']
    k_B2 = latest_cal['k_B2']
    d_A1 = latest_cal['d_A1']
    d_B1 = latest_cal['d_B1']
    d_A2 = latest_cal['d_A2']
    d_B2 = latest_cal['d_B2']
    c_A1 = latest_cal['c_A1']
    c_B1 = latest_cal['c_B1']
    c_A2 = latest_cal['c_A2']
    c_B2 = latest_cal['c_B2']

    # Load strain data
    csv_path = f'Raw Data\{date}_test_{test_num}.csv'
    data = pd.read_csv(csv_path, skiprows=11)
    time_sec = data['Time'].to_numpy()
    strain_a1 = data['Strain A1'].to_numpy()
    strain_b1 = data['Strain B1'].to_numpy()
    strain_a2 = data['Strain A2'].to_numpy()
    strain_b2 = data['Strain B2'].to_numpy()

    # Smooth strain data
    strain_a1_smooth = savgol_filter(strain_a1, 100, 2)
    strain_b1_smooth = savgol_filter(strain_b1, 100, 2)
    strain_a2_smooth = savgol_filter(strain_a2, 100, 2)
    strain_b2_smooth = savgol_filter(strain_b2, 100, 2)

    # Derivatives
    strainDT_a1_smooth = savgol_filter(np.gradient(strain_a1_smooth), 100, 2)
    strainDT_a2_smooth = savgol_filter(np.gradient(strain_a2_smooth), 100, 2)
    strainDT_a1_smooth2 = savgol_filter(strainDT_a1_smooth, 300, 2)
    strainDT_a2_smooth2 = savgol_filter(strainDT_a2_smooth, 300, 2)

    # Find indices where strainDT_a1_smooth2 is within 0.0001 of 0
    mask = np.abs(strainDT_a1_smooth2) < 0.00003
    selected_time_a1 = time_sec[mask]
    selected_strain_a1 = strain_a1_smooth[mask]

    mask = np.abs(strainDT_a2_smooth2) < 0.00003
    selected_time_a2 = time_sec[mask]
    selected_strain_a2 = strain_a2_smooth[mask]

    # Initials and drift lines
    t_start = time_sec[0]
    t_end = time_sec[-1]
    strain_a1_ini = np.mean(strain_a1_smooth[0:500])
    strain_a2_ini = np.mean(strain_a2_smooth[0:500])
    strain_a1_end = np.mean(strain_a1_smooth[-500:])
    strain_a2_end = np.mean(strain_a2_smooth[-500:])
    
    a1_drift_slope = (strain_a1_end - strain_a1_ini) / (t_end - t_start)
    a2_drift_slope = (strain_a2_end - strain_a2_ini) / (t_end - t_start)
    a1_drift = a1_drift_slope * time_sec + strain_a1_ini
    a2_drift = a2_drift_slope * time_sec + strain_a2_ini

    corrected_a1 = strain_a1_smooth - (a1_drift - strain_a1_ini)
    corrected_a2 = strain_a2_smooth - (a2_drift - strain_a2_ini)
    strain_a1_smooth = corrected_a1
    strain_a2_smooth = corrected_a2

    # Calculate force and position
    force = (k_A2 * (strain_a1_smooth - c_A1) - k_A1 * (strain_a2_smooth - c_A2)) / (k_A1 * k_A2 * (d_A2 - d_A1))
    force = savgol_filter(force, 100, 2)
    position = (k_A2*d_A2*(strain_a1_smooth - c_A1) - k_A1*d_A1*(strain_a2_smooth - c_A2)) / (k_A2*(strain_a1_smooth - c_A1) - k_A1*(strain_a2_smooth - c_A2))
    position = savgol_filter(position, 100, 2)

    # Plot strains (Figure 1)
    fig1, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
    axs[0, 0].plot(time_sec, strain_a1, linewidth=0.3)
    axs[0, 0].plot(time_sec, strain_a1_smooth, label='A1')
    axs[0, 0].scatter(selected_time_a1, selected_strain_a1, c='red')
    axs[0, 0].axhline(strain_a1_ini, c='red', linewidth=0.5)
    axs[0, 0].plot(time_sec, a1_drift, c='green', linewidth=0.5)
    axs[0, 0].plot(time_sec, corrected_a1, c='purple', linewidth=0.5)
    axs[0, 0].legend()
    axs[0, 1].plot(time_sec, strain_a2, linewidth=0.3)
    axs[0, 1].plot(time_sec, strain_a2_smooth, label='A2')
    axs[0, 1].scatter(selected_time_a2, selected_strain_a2, c='red')
    axs[0, 1].axhline(strain_a2_ini, c='red', linewidth=0.5)
    axs[0, 1].plot(time_sec, a2_drift, c='green', linewidth=0.5)
    axs[0, 1].plot(time_sec, corrected_a2, c='purple', linewidth=0.5)
    axs[0, 1].legend()
    axs[1, 0].plot(time_sec, strainDT_a1_smooth, linewidth=0.3)
    axs[1, 0].plot(time_sec, strainDT_a1_smooth2, label='A1_dt')
    axs[1, 0].legend()
    axs[1, 1].plot(time_sec, strainDT_a2_smooth, linewidth=0.3)
    axs[1, 1].plot(time_sec, strainDT_a2_smooth2, label='A2_dt')
    axs[1, 1].legend()
    plt.tight_layout()

    # Plot force and position (Figure 2)
    fig2, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
    ax[0].plot(time_sec, force, label='Force')
    ax[0].set_ylabel('Force (N)')
    ax[1].plot(time_sec, position*100, label='Position')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Position (cm)')
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    process_data(date='06_27', test_num='2')