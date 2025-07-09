import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

local_run_flag = False

class LabStalkRow:
    def __init__(self, date, test_num, color):
        self.result_color = color
        self.angle = np.radians(-20)
        self.height = 80 * 1e-2 # centimeters
        self.min_position = 6 * 1e-2    # centimeters
        self.max_position = 16 * 1e-2   # centimeters
        self.min_force = 2  # Newtons
        self.min_force_rate = 0.5 # Newton/second
        self.min_sequential = 20
        
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
        csv_path = rf'Raw Data\{date}_test_{test_num}.csv'
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
        self.strainDT_a1 = np.gradient(self.strain_a1, self.time)
        self.strainDT_b1 = np.gradient(self.strain_b1, self.time)
        self.strainDT_a2 = np.gradient(self.strain_a2, self.time)
        self.strainDT_b2 = np.gradient(self.strain_b2, self.time)

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

    def shift_initials(self, initial_force=0, initial_position=0, zero_cutoff=500):
        '''Adjusts the calibration coefficients so the initial force from
        calculate_force_position() equals initial_force'''
        self.strain_a1_ini = np.mean(self.strain_a1[0:zero_cutoff])
        self.strain_b1_ini = np.mean(self.strain_b1[0:zero_cutoff])
        self.strain_a2_ini = np.mean(self.strain_a2[0:zero_cutoff])
        self.strain_b2_ini = np.mean(self.strain_b2[0:zero_cutoff])

        self.c_A1 = self.strain_a1_ini
        self.c_B1 = self.strain_b1_ini
        self.c_A2 = self.strain_a2_ini
        self.c_B2 = self.strain_b2_ini

    def calculate_force_position(self, smooth=True, window=100, order=2, small_den_cutoff=0.000035):
        self.force_num = self.k_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * (self.strain_a2 - self.c_A2)
        self.force_den = self.k_A1 * self.k_A2 * (self.d_A2 - self.d_A1)
        self.force = self.force_num / self.force_den
        
        self.pos_num = self.k_A2 * self.d_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * self.d_A1 * (self.strain_a2 - self.c_A2)
        self.pos_den = self.k_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * (self.strain_a2 - self.c_A2)
        self.position = np.where(np.abs(self.pos_den) < small_den_cutoff, 0, self.pos_num / self.pos_den)
        
        if smooth:
            self.force = savgol_filter(self.force, window, order)
            self.position = savgol_filter(self.position, window, order)

    def differentiate_force_position(self, smooth=True, window=100, order=2):
        self.forceDT = np.gradient(self.force, self.time)
        self.positionDT = np.gradient(self.position, self.time)

        if smooth:
            self.forceDT = savgol_filter(self.forceDT, window, order)
            self.positionDT = savgol_filter(self.positionDT, window, order)

    def differentiate_force_position_DT(self, smooth=True, window=100, order=2):
        self.forceDDT = np.gradient(self.forceDT, self.time)
        self.positionDDT = np.gradient(self.positionDT, self.time)

        if smooth:
            self.forceDDT = savgol_filter(self.forceDDT, window, order)
            self.positionDDT = savgol_filter(self.positionDDT, window, order)

    def find_stalk_interaction(self, tolerance):
        self.near_zero_accel_indices = np.where(np.abs(self.positionDDT) < tolerance)[0]
        self.valid_position_indices = np.where((self.position > self.min_position) & (self.position < self.max_position))[0]
        self.interaction_indices = np.intersect1d(self.near_zero_accel_indices, self.valid_position_indices)
        
        self.valid_force_indices = np.where((self.force > self.min_force) & (self.forceDT > self.min_force_rate))[0]
        self.interaction_indices = np.intersect1d(self.interaction_indices, self.valid_force_indices)
        
        # Identify sequential groups and filter out blips (groups with fewer than min_sequential indices)
        if len(self.interaction_indices) > 0:
            diffs = np.diff(self.interaction_indices)
            group_starts = np.where(diffs > 1)[0] + 1
            group_indices = np.split(self.interaction_indices, group_starts)
            self.non_blip_indices = np.concatenate([group for group in group_indices if len(group) >= self.min_sequential])
        else:
            self.non_blip_indices = np.array([])
        
        self.interaction_indices = np.intersect1d(self.interaction_indices, self.non_blip_indices)
        
        # Determine gaps and reconnect sections with gaps < 30% of average gap
        gaps = np.diff(self.interaction_indices)
        gaps = gaps[gaps>1]
        avg_gap = np.mean(gaps)
        threshold = 0.3*avg_gap

        new_indices = []
        gap_index = 0
        for i in range(len(self.interaction_indices)-1):
            new_indices.append(self.interaction_indices[i])
            if self.interaction_indices[i+1] - self.interaction_indices[i] > 1:
                if gaps[gap_index] < threshold:
                    gap_start = self.interaction_indices[i] + 1
                    gap_end = gap_start + gaps[gap_index]
                    for new_index in range(gap_start, gap_end):
                        new_indices.append(new_index)
                gap_index += 1

        self.interaction_indices = np.array(new_indices)
        self.stalk_force = self.force[self.interaction_indices]
        self.stalk_position = self.position[self.interaction_indices]
        self.stalk_time = self.time[self.interaction_indices]

    def collect_stalk_sections(self):
        gaps = np.concatenate(([0], np.diff(self.interaction_indices)))

        self.stalk_forces = []
        self.stalk_positions = []
        self.stalk_times = []
        stalk_force_section = []
        stalk_position_section = []
        stalk_time_section = []
        
        for i in range(len(self.interaction_indices)):
            if gaps[i] <= 1:
                stalk_force_section.append(self.stalk_force[i])
                stalk_position_section.append(self.stalk_position[i])
                stalk_time_section.append(self.stalk_time[i])
            else:
                self.stalk_forces.append(np.array(stalk_force_section))
                self.stalk_positions.append(np.array(stalk_position_section))
                self.stalk_times.append(np.array(stalk_time_section))
                stalk_force_section = []
                stalk_position_section = []
                stalk_time_section = []
        self.stalk_forces.append(np.array(stalk_force_section))
        self.stalk_positions.append(np.array(stalk_position_section))
        self.stalk_times.append(np.array(stalk_time_section))
            
    def calc_stalk_stiffness(self):
        self.force_fits = []
        self.position_fits = []
        self.flex_stiffs = []
        for i in range(len(self.stalk_times)):
            time = self.stalk_times[i]
            force = self.stalk_forces[i]
            position = self.stalk_positions[i]
            
            # Fit linear regression to force vs time
            force_coeffs = np.polyfit(time, force, 1)
            force_slope = force_coeffs[0]  # Slope of force over time
            force_fit = np.polyval(force_coeffs, time)
            self.force_fits.append(force_fit)
            
            # Fit linear regression to position vs time
            pos_coeffs = np.polyfit(time, position, 1)
            pos_slope = pos_coeffs[0]  # Slope of position over time
            pos_fit = np.polyval(pos_coeffs, time)
            self.position_fits.append(pos_fit)

            # Calulate fluxual stiffness from slopes and system parameters
            num = force_slope*self.height**3
            den = 3*pos_slope*np.sin(self.angle)
            flexural_stiffness = num/den
            self.flex_stiffs.append(flexural_stiffness)
            
    def plot_force_position(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time, self.force, label='Force')
        # if hasattr(self, 'near_zero_accel_indices'):
        #     ax[0].plot(self.time[self.interaction_indices], self.force[self.interaction_indices], 'ro', markersize=2, label='Near-zero accel')
        ax[0].set_ylabel('Force (N)')
        # ax[0].legend()
        
        ax[1].plot(self.time, self.position*100, label='Position')
        # if hasattr(self, 'near_zero_accel_indices'):
        #     ax[1].plot(self.time[self.interaction_indices], self.position[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')
        # ax[1].legend()
        
        for i in range(len(self.stalk_times)):
            ax[0].plot(self.stalk_times[i], self.stalk_forces[i], c='red')
            ax[0].plot(self.stalk_times[i], self.force_fits[i], c='green')
            ax[1].plot(self.stalk_times[i], self.stalk_positions[i]*100, c='red')
            ax[1].plot(self.stalk_times[i], self.position_fits[i]*100, c='green')
        plt.tight_layout()

    def plot_force_position_DT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.forceDT, label='Force Rate')
        ax[0].set_ylabel('Force Rate (N/s)')
        ax[1].plot(self.time, self.positionDT*100, label='Position Rate')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.positionDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Rate (cm/s)')
        ax[1].legend()
        plt.tight_layout()

    def plot_force_position_DDT(self):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))
        ax[0].plot(self.time, self.forceDDT, label='Force Accel')
        ax[0].set_ylabel('Force Accel (N/s2)')
        ax[1].plot(self.time, self.positionDDT*100, label='Position Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.positionDDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Accel (cm/s2)')
        ax[1].legend()
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

    def plot_results(self):
        plt.figure(20)
        plt.scatter(range(len(self.flex_stiffs)), self.flex_stiffs, c=self.result_color, s=2)
        plt.xlabel('Stalk Number')
        plt.ylabel('Flexural Stiffness')

def process_data(date, test_num, accel_tolerance=1.0, color='red'):
    test = LabStalkRow(date=date, test_num=test_num, color=color)
    test.smooth_strains()
    test.correct_linear_drift()
    test.shift_initials()
    test.calculate_force_position(smooth=True, small_den_cutoff=0.00006)
    test.differentiate_force_position(smooth=True, window=100)
    test.differentiate_force_position_DT(smooth=True, window=100)
    test.find_stalk_interaction(tolerance=accel_tolerance)
    test.collect_stalk_sections()
    test.calc_stalk_stiffness()

    test.plot_force_position()
    test.plot_results()
    # test.plot_force_position_DT()
    # test.plot_force_position_DDT()
    # test.plot_raw_strain()
    if not local_run_flag:
        plt.show()

if __name__ == "__main__":
    local_run_flag = True
    colors = ['red'] * 10 + ['green'] * 10 + ['blue'] * 10
    accel_tols = [0.3]*20 + [1.0]*10
    for i in range(11,40+1):
        process_data(date='07_03', test_num=f'{i}', accel_tolerance=accel_tols[i-11], color=colors[i-11])
        

    # process_data(date='07_03', test_num='25')

    plt.show()