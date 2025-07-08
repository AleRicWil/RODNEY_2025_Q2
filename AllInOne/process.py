import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LabStalkRow:
    def __init__(self, date, test_num):
        # Load calibration coefficients
        cal_csv_path = r'AllInOne\calibration_history.csv'
        cal_data = pd.read_csv(cal_csv_path)
        latest_cal = cal_data.iloc[-1]  # Get the most recent calibration
        self.k_A1 = latest_cal['k_A1']
        self.k_B1 = latest_cal['k_B1']
        self.k_A2 = latest_cal['k_A2']
        self.k_B2 = latest_cal['k_B2']
        self.d_A1 = latest_cal['d_A1']
        self.d_B1 = latest_cal['d_B1']
        self.d_A2 = latest_cal['d_A2']
        self.d_B2 = latest_cal['d_B2']
        self.c_A1 = latest_cal['c_A1']
        self.c_B1 = latest_cal['c_B1']
        self.c_A2 = latest_cal['c_A2']
        self.c_B2 = latest_cal['c_B2']

        # Load strain data
        csv_path = f'Raw Data\{date}_test_{test_num}.csv'
        data = pd.read_csv(csv_path, skiprows=11)
        self.time = data['Time'].to_numpy()
        self.strain_a1 = self.strain_a1_raw = data['Strain A1'].to_numpy()
        self.strain_b1 = self.strain_b1_raw = data['Strain B1'].to_numpy()
        self.strain_a2 = self.strain_a2_raw = data['Strain A2'].to_numpy()
        self.strain_b2 = self.strain_b2_raw = data['Strain B2'].to_numpy()

    def smooth_strains(self, window=100, order=2):
        self.strain_a1 = savgol_filter(self.strain_a1, window, order)
        self.strain_b1 = savgol_filter(self.strain_b1, window, order)
        self.strain_a2 = savgol_filter(self.strain_a2, window, order)
        self.strain_b2 = savgol_filter(self.strain_b2, window, order)

    def smooth_strain_DTs(self, window=100, order=2):
        self.strainDT_a1 = savgol_filter(self.strainDT_a1, window, order)
        self.strainDT_b1 = savgol_filter(self.strainDT_b1, window, order)
        self.strainDT_a2 = savgol_filter(self.strainDT_a2, window, order)
        self.strainDT_b2 = savgol_filter(self.strainDT_b2, window, order)

    def differentiate_strains(self):
        self.strainDT_a1 = np.gradient(self.strain_a1)
        self.strainDT_b1 = np.gradient(self.strain_b1)
        self.strainDT_a2 = np.gradient(self.strain_a2)
        self.strainDT_b2 = np.gradient(self.strain_b2)

    def correct_linear_drift(self, zero_cutoff=500):
        t_start = self.time[0]
        t_end = self.time[-1]

        self.strain_a1_ini = np.mean(self.strain_a1[0:zero_cutoff])
        self.strain_b1_ini = np.mean(self.strain_b1[0:zero_cutoff])
        self.strain_a2_ini = np.mean(self.strain_a2[0:zero_cutoff])
        self.strain_b2_ini = np.mean(self.strain_b2[0:zero_cutoff])

        strain_a1_end = np.mean(self.strain_a1[-zero_cutoff:])
        strain_b1_end = np.mean(self.strain_b1[-zero_cutoff:])
        strain_a2_end = np.mean(self.strain_a2[-zero_cutoff:])
        strain_b2_end = np.mean(self.strain_b2[-zero_cutoff:])

        a1_drift_slope = (strain_a1_end - self.strain_a1_ini) / (t_end - t_start)
        b1_drift_slope = (strain_b1_end - self.strain_b1_ini) / (t_end - t_start)
        a2_drift_slope = (strain_a2_end - self.strain_a2_ini) / (t_end - t_start)
        b2_drift_slope = (strain_b2_end - self.strain_b2_ini) / (t_end - t_start)

        a1_drift = a1_drift_slope * self.time + self.strain_a1_ini
        b1_drift = b1_drift_slope * self.time + self.strain_b1_ini
        a2_drift = a2_drift_slope * self.time + self.strain_a2_ini
        b2_drift = b2_drift_slope * self.time + self.strain_b2_ini

        corrected_a1 = self.strain_a1 - (a1_drift - self.strain_a1_ini)
        corrected_b1 = self.strain_b1 - (b1_drift - self.strain_b1_ini)
        corrected_a2 = self.strain_a2 - (a2_drift - self.strain_a2_ini)
        corrected_b2 = self.strain_b2 - (b2_drift - self.strain_b2_ini)

        self.strain_a1 = corrected_a1
        self.strain_b1 = corrected_b1
        self.strain_a2 = corrected_a2
        self.strain_b2 = corrected_b2

    def calculate_force_position(self, smooth=True, window=100, order=2):
        self.force = (self.k_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * (self.strain_a2 - self.c_A2)) / (self.k_A1 * self.k_A2 * (self.d_A2 - self.d_A1))
        self.position = (self.k_A2*self.d_A2*(self.strain_a1 - self.c_A1) - self.k_A1*self.d_A1*(self.strain_a2 - self.c_A2)) / (self.k_A2*(self.strain_a1 - self.c_A1) - self.k_A1*(self.strain_a2 - self.c_A2))
        
        if smooth:
            self.force = savgol_filter(self.force, window, order)
            self.position = savgol_filter(self.position, window, order)

    def shift_initials(self, initial_force=0, initial_position=0, zero_cutoff=500):
        '''This function adjusts the calibration coefficients so the initial force from
        calculate_force_position() comes out to equal initial_force'''
        

    def plot_force_position(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.force, label='Force')
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time, self.position*100, label='Position')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')
        plt.tight_layout()

    def plot_raw_strain(self):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        axs[0, 0].plot(self.time, self.strain_a1_raw, linewidth=0.3)
        axs[0, 0].plot(self.time, self.strain_a1, label='A1')
        axs[0, 0].axhline(self.strain_a1_ini, c='red', linewidth=0.5)
        axs[0, 0].legend()
        axs[0, 1].plot(self.time, self.strain_a2_raw, linewidth=0.3)
        axs[0, 1].plot(self.time, self.strain_a2, label='A2')
        axs[0, 1].axhline(self.strain_a2_ini, c='red', linewidth=0.5)
        axs[0, 1].legend()
        axs[1, 0].plot(self.time, self.strain_b1_raw, linewidth=0.3)
        axs[1, 0].plot(self.time, self.strain_b1, label='B1')
        axs[1, 0].axhline(self.strain_b1_ini, c='red', linewidth=0.5)
        axs[1, 0].legend()
        axs[1, 1].plot(self.time, self.strain_b2_raw, linewidth=0.3)
        axs[1, 1].plot(self.time, self.strain_b2, label='B2')
        axs[1, 1].axhline(self.strain_b2_ini, c='red', linewidth=0.5)
        axs[1, 1].legend()
        plt.tight_layout()

def process_data(date, test_num):
    test = LabStalkRow(date=date, test_num=test_num)
    test.smooth_strains()
    test.correct_linear_drift()
    test.calculate_force_position(smooth=True)
    test.plot_force_position()
    test.shift_initials()
    test.plot_force_position()
    test.plot_raw_strain()
    plt.show()

if __name__ == "__main__":
    process_data(date='07_03', test_num='21')
