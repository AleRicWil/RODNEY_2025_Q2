import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import os
from scipy.signal import savgol_filter
from itertools import product
from scipy.optimize import minimize

local_run_flag = False
results = []

class LabStalkRow:
    def __init__(self, date, test_num, min_force_rate=0.3,
                 lo_pos_accel_tol=1.0, med_pos_accel_tol=0.5, hi_pos_accel_tol=0.5,
                 lo_force_accel_tol=25.0, med_force_accel_tol=30.0, hi_force_accel_tol=40.0):
        
        self.min_position = 6 * 1e-2    # centimeters
        self.max_position = 16 * 1e-2   # centimeters
        self.min_force = 2  # Newtons
        self.min_force_rate = min_force_rate  # Newton/second
        self.min_sequential = 20
        # Create results folder
        parent_folder = os.path.join('Results')
        os.makedirs(parent_folder, exist_ok=True)
        self.results_path = os.path.join(parent_folder, f'results.csv')
        self.date = date

        # Load strain data and calibration coefficients from CSV header
        self.csv_path = rf'Raw Data\{date}\{date}_test_{test_num}.csv'
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            self.header_rows = []
            for row in reader:
                if row[0] == "=====":
                    break
                self.header_rows.append(row)
            # Find test parameter rows
            param_rows = 0
            for row in self.header_rows:
                if row[0] == "rodney configuration":
                    self.configuration = row[1]
                    param_rows += 1
                if row[0] == "sensor calibration (k d c)":
                    k_str = row[1]
                    d_str = row[2]
                    k_values = [float(v) for v in k_str.split()]
                    d_values = [float(v) for v in d_str.split()]
                    self.k_A1, self.k_B1, self.k_A2, self.k_B2 = k_values
                    self.d_A1, self.d_B1, self.d_A2, self.d_B2 = d_values
                    param_rows += 1
                if row[0] == "stalk array (lo med hi)":
                    self.stalk_type = row[1]
                    if row[1] == "lo":
                        self.result_color = 'red'
                        self.pos_accel_tol = lo_pos_accel_tol
                        self.force_accel_tol = lo_force_accel_tol
                    elif row[1] == "med":
                        self.result_color = 'green'
                        self.pos_accel_tol = med_pos_accel_tol
                        self.force_accel_tol = med_force_accel_tol
                    elif row[1] == "hi":
                        self.result_color = 'blue'
                        self.pos_accel_tol = hi_pos_accel_tol
                        self.force_accel_tol = hi_force_accel_tol
                    param_rows += 1
                if row[0] == "sensor height (cm)":
                    self.height = float(row[1])*1e-2
                    param_rows += 1
                if row[0] == "sensor yaw (degrees)":
                    self.yaw = np.radians(-float(row[1]))
                    param_rows += 1
                if row[0] == "sensor offset (cm to gauge 2)":
                    self.sensor_offset = float(row[1])
                    param_rows += 1
            if not param_rows >= 6:
                raise ValueError("Test parameter rows missing in header")
            # Read the data
            data = pd.read_csv(f)

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

        self.a1_drift = a1_drift_slope * self.time + self.strain_a1_ini
        self.b1_drift = b1_drift_slope * self.time + self.strain_b1_ini
        self.a2_drift = a2_drift_slope * self.time + self.strain_a2_ini
        self.b2_drift = b2_drift_slope * self.time + self.strain_b2_ini

        corrected_a1 = self.strain_a1 - (self.a1_drift - self.strain_a1_ini)
        corrected_b1 = self.strain_b1 - (self.b1_drift - self.strain_b1_ini)
        corrected_a2 = self.strain_a2 - (self.a2_drift - self.strain_a2_ini)
        corrected_b2 = self.strain_b2 - (self.b2_drift - self.strain_b2_ini)

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

    def find_stalk_interaction(self):
        self.near_zero_accel_indices = np.where(np.abs(self.positionDDT) < self.pos_accel_tol)[0]
        self.valid_position_indices = np.where((self.position > self.min_position) & (self.position < self.max_position))[0]
        self.interaction_indices = np.intersect1d(self.near_zero_accel_indices, self.valid_position_indices)
        
        self.valid_force_indices = np.where((self.force > self.min_force) & (self.forceDT > self.min_force_rate))[0]
        self.interaction_indices = np.intersect1d(self.interaction_indices, self.valid_force_indices)

        self.valid_force_accel_indices = np.where(np.abs(self.forceDDT) < self.force_accel_tol)[0]
        self.interaction_indices = np.intersect1d(self.interaction_indices, self.valid_force_accel_indices)
        
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
        big_gaps = gaps[gaps>1]
        avg_gap = np.average(big_gaps)

        self.stalk_forces = []
        self.stalk_positions = []
        self.stalk_times = []
        stalk_force_section = []
        stalk_position_section = []
        stalk_time_section = []
        
        for i in range(len(self.interaction_indices)):
            if gaps[i] <= 1:    # accumulate point on current stalk
                stalk_force_section.append(self.stalk_force[i])
                stalk_position_section.append(self.stalk_position[i])
                stalk_time_section.append(self.stalk_time[i])
            else:               # store current stalk and reset accumulation
                self.stalk_forces.append(np.array(stalk_force_section))
                self.stalk_positions.append(np.array(stalk_position_section))
                self.stalk_times.append(np.array(stalk_time_section))
                stalk_force_section = []
                stalk_position_section = []
                stalk_time_section = []
                
                if gaps[i] >= avg_gap*1.6:  # if the gap is very large, skip next stalk number
                    self.stalk_forces.append(np.nan)
                    self.stalk_positions.append(np.nan)
                    self.stalk_times.append(np.nan)
        
        # add the last stalk
        self.stalk_forces.append(np.array(stalk_force_section))
        self.stalk_positions.append(np.array(stalk_position_section))
        self.stalk_times.append(np.array(stalk_time_section))
            
    def calc_stalk_stiffness(self):
        self.force_fits = []
        self.position_fits = []
        self.flex_stiffs = []
        for i in range(len(self.stalk_times)):
            if not np.isnan(self.stalk_times[i]).all():
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
                den = 3*pos_slope*np.sin(self.yaw)
                flexural_stiffness = num/den
                self.flex_stiffs.append(flexural_stiffness)
            else:
                self.force_fits.append(np.nan)
                self.position_fits.append(np.nan)
                self.flex_stiffs.append(np.nan)
            
        results.append(self.flex_stiffs)
            
    def plot_force_position(self, view_flag=False):
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
        
        if not view_flag:
            for i in range(len(self.stalk_times)):
                if not np.isnan(self.stalk_times[i]).all():
                    ax[0].plot(self.stalk_times[i], self.stalk_forces[i], c='red')
                    ax[0].plot(self.stalk_times[i], self.force_fits[i], c='green')
                    ax[1].plot(self.stalk_times[i], self.stalk_positions[i]*100, c='red')
                    ax[1].plot(self.stalk_times[i], self.position_fits[i]*100, c='green')
            plt.tight_layout()
        # For verifying calibration coefficients
        else:
            g = 9.8
            ax[0].axhline(1*g, c='red', linewidth=0.5)
            ax[0].axhline(0.5*g, c='red', linewidth=0.5)
            ax[0].axhline(0.2*g, c='red', linewidth=0.5)
            ax[1].axhline(15, c='red', linewidth=0.5)
            ax[1].axhline(10, c='red', linewidth=0.5)

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
        if hasattr(self, 'near_zero_accel_indices'):
            ax[0].plot(self.time[self.interaction_indices], self.forceDDT[self.interaction_indices], 'ro', markersize=2, label='Near-zero accel')
        ax[0].set_ylabel('Force Accel (N/s2)')
        ax[1].plot(self.time, self.positionDDT*100, label='Position Accel')
        if hasattr(self, 'near_zero_accel_indices'):
            ax[1].plot(self.time[self.interaction_indices], self.positionDDT[self.interaction_indices]*100, 'ro', markersize=2, label='Near-zero accel')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position Accel (cm/s2)')
        plt.tight_layout()

    def plot_raw_strain(self):
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
        axs[0, 0].plot(self.time, self.strain_a1_raw, linewidth=0.3)
        axs[0, 0].plot(self.time, self.a1_drift, linewidth=0.3)
        axs[0, 0].plot(self.time, self.strain_a1, label='A1')
        axs[0, 0].axhline(self.strain_a1_ini, c='red', linewidth=0.5)
        axs[0, 0].legend()
        axs[0, 1].plot(self.time, self.strain_a2_raw, linewidth=0.3)
        axs[0, 1].plot(self.time, self.a2_drift, linewidth=0.3)
        axs[0, 1].plot(self.time, self.strain_a2, label='A2')
        axs[0, 1].axhline(self.strain_a2_ini, c='red', linewidth=0.5)
        axs[0, 1].legend()
        axs[1, 0].plot(self.time, self.strain_b1_raw, linewidth=0.3)
        axs[1, 0].plot(self.time, self.b1_drift, linewidth=0.3)
        axs[1, 0].plot(self.time, self.strain_b1, label='B1')
        axs[1, 0].axhline(self.strain_b1_ini, c='red', linewidth=0.5)
        axs[1, 0].legend()
        axs[1, 1].plot(self.time, self.strain_b2_raw, linewidth=0.3)
        axs[1, 1].plot(self.time, self.b2_drift, linewidth=0.3)
        axs[1, 1].plot(self.time, self.strain_b2, label='B2')
        axs[1, 1].axhline(self.strain_b2_ini, c='red', linewidth=0.5)
        axs[1, 1].legend()
        plt.tight_layout()

    def plot_results(self):
        plt.figure(20)
        plt.scatter(range(1,len(self.flex_stiffs)+1), self.flex_stiffs, c=self.result_color, s=2)
        # plt.xlabel('Stalk Number')
        plt.ylabel('Flexural Stiffness')

    def save_results(self, overwrite_result=False):
        # Prepare row data
        row = [self.date, self.csv_path]
        numeric_headers = ['sensor height (cm)', 'sensor yaw (degrees)', 'sensor pitch (degrees)', 'sensor roll (degrees)',
                           'rate of travel (ft/min)', 'angle of travel (degrees)', 'sensor offset (cm to gauge 2)']
        for header_row in self.header_rows:
            if header_row[0] == "sensor calibration (k d c)":
                row.append(header_row[1] + ' : ' + header_row[2] + ' : ' + header_row[3])  # Keep as string
            else:
                value = header_row[1] if len(header_row) > 1 else ''
                if header_row[0] in numeric_headers and value == '':
                    row.append(np.nan)  # Use np.nan for empty numeric fields
                elif header_row[0] in numeric_headers:
                    row.append(float(value))  # Cast numeric strings to float
                else:
                    row.append(value)  # Keep as string or empty string
        row += [float(self.pos_accel_tol), float(self.force_accel_tol)]
        for stalk in self.flex_stiffs:
            row.append(float(stalk) if not np.isnan(stalk) else np.nan)

        # Check if file exists
        file_exists = os.path.isfile(self.results_path)
        headers = ['Date', 'File'] + [row[0] for row in self.header_rows if row[0] != "====="]
        headers += ['positionDDT_tol', 'forceDDT_tol']
        headers += ['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']

        # Define dtypes for columns
        dtype_dict = {
            'Date': 'object',
            'File': 'object',
            'user_note': 'object',
            'rodney configuration': 'object',
            'sensor calibration (k d c)': 'object',
            'stalk array (lo med hi)': 'object',
            'sensor height (cm)': 'float64',
            'sensor yaw (degrees)': 'float64',
            'sensor pitch (degrees)': 'float64',
            'sensor roll (degrees)': 'float64',
            'rate of travel (ft/min)': 'float64',
            'angle of travel (degrees)': 'float64',
            'sensor offset (cm to gauge 2)': 'float64',
            'positionDDT_tol': 'float64',
            'forceDDT_tol': 'float64',
            'Stalk1': 'float64',
            'Stalk2': 'float64',
            'Stalk3': 'float64',
            'Stalk4': 'float64',
            'Stalk5': 'float64',
            'Stalk6': 'float64',
            'Stalk7': 'float64',
            'Stalk8': 'float64',
            'Stalk9': 'float64'
        }

        if file_exists:
            df = pd.read_csv(self.results_path)
            # Ensure DataFrame has correct dtypes
            for col, dtype in dtype_dict.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
            if 'File' in df.columns and self.csv_path in df['File'].values and overwrite_result:
                # Overwrite existing row
                for col, val in zip(headers, row):
                    df.loc[df['File'] == self.csv_path, col] = val
                df.to_csv(self.results_path, index=False)
                return

        # Write new row
        with open(self.results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(headers)
            csvwriter.writerow(row)

def boxplot_data(rodney_config, plot_num=20):
    results_df = pd.read_csv(r'Results\results.csv')
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    plt.figure(plot_num)
    plt.title(rodney_config)
    for _, row in lo_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='red', s=2)
    lo_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='red'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='red'),
                   medianprops=dict(color='black'))
    for _, row in med_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='green', s=2)
    med_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='green'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='green'),
                   medianprops=dict(color='black'))
    for _, row in hi_EIs.iloc[2:].iterrows():
        plt.scatter(range(1, len(row) + 1), row, c='blue', s=2)
    hi_EIs.boxplot(grid=False, patch_artist=True, boxprops=dict(facecolor='none', color='blue'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='blue'),
                   medianprops=dict(color='black'))
    plt.ylim(0, None)

def get_stats(rodney_config, plot_num=None):
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)
        margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * sem
        rel_margin = margin / type_mean if type_mean != 0 else 0  # Relative margin
        return {'interval': ci, 'mean': mean, 'margin': margin, 'rel_margin': rel_margin}
    
    results_df = pd.read_csv(r'Results\results.csv')
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    lo_stats = lo_EIs.describe()
    lo_mean = np.mean(lo_stats.loc['mean'])
    lo_ci = {col: get_ci(lo_EIs[col], lo_mean) for col in lo_EIs.columns} 
    lo_relMargins = np.array([lo_ci[col]['rel_margin'] for col in lo_EIs.columns])

    med_stats = med_EIs.describe()
    med_mean = np.mean(med_stats.loc['mean'])
    med_ci = {col: get_ci(med_EIs[col], med_mean) for col in med_EIs.columns}
    med_relMargins = np.array([med_ci[col]['rel_margin'] for col in med_EIs.columns])

    hi_stats = hi_EIs.describe()
    hi_mean = np.mean(hi_stats.loc['mean'])
    hi_ci = {col: get_ci(hi_EIs[col], hi_mean) for col in hi_EIs.columns}
    hi_relMargins = np.array([hi_ci[col]['rel_margin'] for col in hi_EIs.columns])

    all_relMargins = np.append(np.append(lo_relMargins, med_relMargins), hi_relMargins)
    all_relMargins_mean = np.nanmean(all_relMargins)
    medhi_relMargins = np.append(med_relMargins, hi_relMargins)
    medhi_relMargins_mean = np.nanmean(medhi_relMargins)


    if plot_num is not None:
        plt.figure(plot_num)
        plt.scatter(range(1, len(lo_relMargins)+1), lo_relMargins*100, label='lo', c='red')
        plt.scatter(range(1, len(med_relMargins)+1), med_relMargins*100, label='med', c='green')
        plt.scatter(range(1, len(hi_relMargins)+1), hi_relMargins*100, label='hi', c='blue')
        plt.axhline(all_relMargins_mean*100, c='black')
        plt.axhline(medhi_relMargins_mean*100, c='brown')
    
        plt.ylabel('% Relative Error Margin')
        plt.title('Error Margin Relative to Mean of Stalk Type')
        plt.legend()

    return all_relMargins_mean

# Optimization functions
def get_all_tests(date):
    test_nums = []
    for file in os.listdir(f'Raw Data/{date}'):
        if file.endswith('.csv'):
            test_num = file.split('_test_')[1].split('.csv')[0]
            test_nums.append(test_num)
    return test_nums

def get_config(date, test_num):
    csv_path = f'Raw Data/{date}/{date}_test_{test_num}.csv'
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "rodney configuration":
                return row[1]
    return None

def get_tests_for_config(date, rodney_config):
    tests = []
    for test_num in get_all_tests(date):
        config = get_config(date, test_num)
        if config == rodney_config:
            tests.append(test_num)
    return tests

# def optimize_parameters(dates, rodney_config):
#     all_tests = []
#     for date in dates:
#         test_nums = get_tests_for_config(date, rodney_config)
#         for test_num in test_nums:
#             all_tests.append((date, test_num))
#     print(f'Found {len(all_tests)} files for {rodney_config}')
    
#     # Define objective function
#     def objective(params):
#         min_force_rate, lo_pos_accel_tol, med_pos_accel_tol, hi_pos_accel_tol, \
#         lo_force_accel_tol, med_force_accel_tol, hi_force_accel_tol = params

#         # Process all tests with these parameters
#         for date, test_num in all_tests:
#             test = LabStalkRow(date=date, test_num=test_num,
#                                min_force_rate=min_force_rate,
#                                lo_pos_accel_tol=lo_pos_accel_tol,
#                                med_pos_accel_tol=med_pos_accel_tol,
#                                hi_pos_accel_tol=hi_pos_accel_tol,
#                                lo_force_accel_tol=lo_force_accel_tol,
#                                med_force_accel_tol=med_force_accel_tol,
#                                hi_force_accel_tol=hi_force_accel_tol)
#             test.smooth_strains()
#             test.correct_linear_drift()
#             test.shift_initials()
#             test.calculate_force_position(smooth=True, small_den_cutoff=0.00006)
#             test.differentiate_force_position(smooth=True, window=100)
#             test.differentiate_force_position_DT(smooth=True, window=100)
#             test.find_stalk_interaction()
#             test.collect_stalk_sections()
#             test.calc_stalk_stiffness()
#             test.save_results(overwrite_result=True)

#         # Compute all_relMargins_mean
#         score = get_stats(rodney_config)
#         print(f"Parameters: {params}, Score: {score}")
#         return score

#     # Define parameter bounds
#     bounds = [
#         (0.2, 0.6),  # min_force_rate
#         (0.5, 2.0),  # lo_pos_accel_tol
#         (0.4, 2.0),  # med_pos_accel_tol
#         (0.4, 2.0),  # hi_pos_accel_tol
#         (20, 50),    # lo_force_accel_tol
#         (20, 50),    # med_force_accel_tol
#         (20, 50)     # hi_force_accel_tol
#     ]

#     # Initial guess (midpoint of bounds)
#     initial_guess = [(b[0] + b[1]) / 2 for b in bounds]

#     # Run optimization
#     result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)

#     best_params = result.x
#     best_score = result.fun

#     print(f"Best parameters for {rodney_config}: "
#           f"min_force_rate={best_params[0]:.4f}, "
#           f"lo_pos_accel_tol={best_params[1]:.4f}, "
#           f"med_pos_accel_tol={best_params[2]:.4f}, "
#           f"hi_pos_accel_tol={best_params[3]:.4f}, "
#           f"lo_force_accel_tol={best_params[4]:.4f}, "
#           f"med_force_accel_tol={best_params[5]:.4f}, "
#           f"hi_force_accel_tol={best_params[6]:.4f}, "
#           f"Best score: {best_score:.6f}")

# Helper function to load optimized parameters
def load_optimized_parameters(rodney_config):
    param_file = 'Results/optimized_parameters.csv'
    default_params = {
        'min_force_rate': 0.3,
        'lo_pos_accel_tol': 1.0,
        'med_pos_accel_tol': 0.5,
        'hi_pos_accel_tol': 0.5,
        'lo_force_accel_tol': 25.0,
        'med_force_accel_tol': 30.0,
        'hi_force_accel_tol': 40.0
    }
    if os.path.isfile(param_file):
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            row = df[df['rodney_config'] == rodney_config].iloc[0]
            return {
                'min_force_rate': float(row['min_force_rate']),
                'lo_pos_accel_tol': float(row['lo_pos_accel_tol']),
                'med_pos_accel_tol': float(row['med_pos_accel_tol']),
                'hi_pos_accel_tol': float(row['hi_pos_accel_tol']),
                'lo_force_accel_tol': float(row['lo_force_accel_tol']),
                'med_force_accel_tol': float(row['med_force_accel_tol']),
                'hi_force_accel_tol': float(row['hi_force_accel_tol'])
            }
    return default_params

# Modified process_data to use optimized parameters
def process_data(date, test_num, view=False, overwrite=False):
    # Load optimized filtering parameters for the configuration
    config = get_config(date, test_num)
    params = load_optimized_parameters(config)
    
    test = LabStalkRow(date=date, test_num=test_num,
                       min_force_rate=params['min_force_rate'],
                       lo_pos_accel_tol=params['lo_pos_accel_tol'],
                       med_pos_accel_tol=params['med_pos_accel_tol'],
                       hi_pos_accel_tol=params['hi_pos_accel_tol'],
                       lo_force_accel_tol=params['lo_force_accel_tol'],
                       med_force_accel_tol=params['med_force_accel_tol'],
                       hi_force_accel_tol=params['hi_force_accel_tol'])
    test.smooth_strains()
    test.correct_linear_drift()
    test.shift_initials()
    test.calculate_force_position(smooth=True, small_den_cutoff=0.00006)
    test.differentiate_force_position(smooth=True, window=100)
    test.differentiate_force_position_DT(smooth=True, window=100)
    test.find_stalk_interaction()
    test.collect_stalk_sections()
    test.calc_stalk_stiffness()

    if view:
        test.plot_force_position(view_flag=False)
        # test.plot_raw_strain()
        # test.plot_force_position_DT()
        # test.plot_force_position_DDT()
    else:
        test.save_results(overwrite_result=True)
    
    if not local_run_flag:
        plt.show()

def optimize_parameters(dates, rodney_config):
    all_tests = []
    for date in dates:
        test_nums = get_tests_for_config(date, rodney_config)
        for test_num in test_nums:
            all_tests.append((date, test_num))
    print(f'Found {len(all_tests)} files for {rodney_config}')
    
    # Define objective function
    def objective(params):
        min_force_rate, lo_pos_accel_tol, med_pos_accel_tol, hi_pos_accel_tol, \
        lo_force_accel_tol, med_force_accel_tol, hi_force_accel_tol = params
        for date, test_num in all_tests:
            test = LabStalkRow(date=date, test_num=test_num,
                               min_force_rate=min_force_rate,
                               lo_pos_accel_tol=lo_pos_accel_tol,
                               med_pos_accel_tol=med_pos_accel_tol,
                               hi_pos_accel_tol=hi_pos_accel_tol,
                               lo_force_accel_tol=lo_force_accel_tol,
                               med_force_accel_tol=med_force_accel_tol,
                               hi_force_accel_tol=hi_force_accel_tol)
            test.smooth_strains()
            test.correct_linear_drift()
            test.shift_initials()
            test.calculate_force_position(smooth=True, small_den_cutoff=0.00006)
            test.differentiate_force_position(smooth=True, window=100)
            test.differentiate_force_position_DT(smooth=True, window=100)
            test.find_stalk_interaction()
            test.collect_stalk_sections()
            test.calc_stalk_stiffness()
            test.save_results(overwrite_result=True)
        score = get_stats(rodney_config)
        print(f"Parameters: {params}, Score: {score}")
        return score

    # Initial bounds (first round)
    initial_bounds = [
        (0.2, 0.6),  # min_force_rate
        (0.5, 2.0),  # lo_pos_accel_tol
        (0.4, 0.8),  # med_pos_accel_tol
        (0.4, 0.8),  # hi_pos_accel_tol
        (20, 50),    # lo_force_accel_tol
        (25, 50),    # med_force_accel_tol
        (35, 50)     # hi_force_accel_tol
    ]

    n_starts = 10
    best_score = float('inf')
    best_params = None

    # Function to run optimization for one round
    def run_optimization_round(bounds, round_num):
        nonlocal best_score, best_params
        print(f"Starting optimization round {round_num}")
        for i in range(n_starts):
            initial_guess = [np.random.uniform(b[0], b[1]) for b in bounds]
            print(f"Round {round_num}, Start {i+1}/{n_starts}, Initial guess: {initial_guess}")
            result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 1500, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        print(f"Round {round_num} best parameters: {best_params}, Score: {best_score}")

    # Run first round
    run_optimization_round(initial_bounds, 1)

    # Second round: ±30% of first round's optimal values
    second_bounds = []
    for i, (lower, upper) in enumerate(initial_bounds):
        opt_val = best_params[i]
        delta = 0.3 * abs(opt_val)
        new_lower = max(lower, opt_val - delta)
        new_upper = min(upper, opt_val + delta)
        second_bounds.append((new_lower, new_upper))
    run_optimization_round(second_bounds, 2)

    # Third round: ±10% of second round's optimal values
    third_bounds = []
    for i, (lower, upper) in enumerate(initial_bounds):
        opt_val = best_params[i]
        delta = 0.1 * abs(opt_val)
        new_lower = max(lower, opt_val - delta)
        new_upper = min(upper, opt_val + delta)
        third_bounds.append((new_lower, new_upper))
    run_optimization_round(third_bounds, 3)

    # Save optimized parameters
    param_file = 'Results/optimized_parameters.csv'
    headers = ['rodney_config', 'min_force_rate', 'lo_pos_accel_tol', 'med_pos_accel_tol',
               'hi_pos_accel_tol', 'lo_force_accel_tol', 'med_force_accel_tol',
               'hi_force_accel_tol', 'best_score']
    row = [rodney_config] + list(best_params) + [best_score]

    file_exists = os.path.isfile(param_file)
    if file_exists:
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            df.loc[df['rodney_config'] == rodney_config, headers] = row
            df.to_csv(param_file, index=False)
        else:
            with open(param_file, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row)
    else:
        with open(param_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(row)

    print(f"Best parameters for {rodney_config}: "
          f"min_force_rate={best_params[0]:.4f}, "
          f"lo_pos_accel_tol={best_params[1]:.4f}, "
          f"med_pos_accel_tol={best_params[2]:.4f}, "
          f"hi_pos_accel_tol={best_params[3]:.4f}, "
          f"lo_force_accel_tol={best_params[4]:.4f}, "
          f"med_force_accel_tol={best_params[5]:.4f}, "
          f"hi_force_accel_tol={best_params[6]:.4f}, "
          f"Best score: {best_score:.6f}")

if __name__ == "__main__":
    local_run_flag = True
    
    '''Batch run of same configuration'''
    # for i in range(11, 40+1):
    #     process_data(date='07_03', test_num=f'{i}', view=True, overwrite=True)

    # boxplot_data(rodney_config='Integrated Beam Prototype 1', plot_num=101)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', plot_num=102)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', plot_num=103)
    '''end batch run'''

    '''Statistics'''
    # print('1', get_stats(rodney_config='Integrated Beam Prototype 1', plot_num=201))
    # print('2', get_stats(rodney_config='Integrated Beam Prototype 2', plot_num=202))
    # print('3', get_stats(rodney_config='Integrated Beam Prototype 3', plot_num=203))
    '''end statistics'''

    '''Single file run and view full file. Does not save result'''
    # process_data(date='07_10', test_num='1', view=True)
    '''end single file run'''

    # Optimize parameters for a specific configuration
    # optimize_parameters(dates=['07_03', '07_10'], rodney_config='Integrated Beam Prototype 1')

    plt.show()