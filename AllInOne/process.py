import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import os
from skopt import gp_minimize
from skopt.space import Real
from scipy.signal import savgol_filter

local_run_flag = False
results = []

class LabStalkRow:
    def __init__(self, date, test_num, min_force_rate=0.01,
                 lo_pos_accel_tol=1.0, med_pos_accel_tol=0.5, hi_pos_accel_tol=0.5,
                 lo_force_accel_tol=25.0, med_force_accel_tol=30.0, hi_force_accel_tol=40.0):
        
        self.min_position = 5 * 1e-2    # centimeters
        self.max_position = 18 * 1e-2   # centimeters
        self.min_force = 2  # Newtons
        self.min_force_rate = min_force_rate  # Newton/second
        self.min_sequential = 10
        # Create results folder
        parent_folder = os.path.join('Results')
        os.makedirs(parent_folder, exist_ok=True)
        self.results_path = os.path.join(parent_folder, f'results.csv')
        self.date = date
        self.test_num = test_num

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

    def calc_force_position(self, smooth=True, window=100, order=2, small_den_cutoff=0.000035):
        self.force_num = self.k_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * (self.strain_a2 - self.c_A2)
        self.force_den = self.k_A1 * self.k_A2 * (self.d_A2 - self.d_A1)
        self.force = self.force_num / self.force_den
        
        self.pos_num = self.k_A2 * self.d_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * self.d_A1 * (self.strain_a2 - self.c_A2)
        self.pos_den = self.k_A2 * (self.strain_a1 - self.c_A1) - self.k_A1 * (self.strain_a2 - self.c_A2)
        self.position = np.where(np.abs(self.pos_den) < small_den_cutoff, 0, self.pos_num / self.pos_den)
        
        if smooth:
            self.raw_force = self.force
            self.raw_position = self.position
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
        # Combine conditions into a single boolean mask
        mask = (np.abs(self.positionDDT) < self.pos_accel_tol) & \
            (self.position > self.min_position) & \
            (self.position < self.max_position) & \
            (self.positionDT < 0.1) & \
            (self.force > self.min_force) & \
            (self.forceDT > self.min_force_rate) & \
            (np.abs(self.forceDDT) < self.force_accel_tol)
        
        # Get initial interaction indices
        interaction_indices = np.where(mask)[0]
        
        # Filter out blips (groups with fewer than min_sequential indices)
        if len(interaction_indices) > 0:
            diffs = np.diff(interaction_indices)
            group_starts = np.where(diffs > 1)[0] + 1
            groups = np.split(interaction_indices, group_starts)
            interaction_indices = np.concatenate([g for g in groups if len(g) >= self.min_sequential]) if groups else np.array([])
        
        # Reconnect gaps < 30% of average gap
        if len(interaction_indices) > 1:
            gaps = np.diff(interaction_indices)
            valid_gaps = gaps[gaps > 1]
            if len(valid_gaps) > 0:
                threshold = 0.3 * np.mean(valid_gaps)
                new_indices = []
                for i in range(len(interaction_indices) - 1):
                    new_indices.append(interaction_indices[i])
                    gap = interaction_indices[i + 1] - interaction_indices[i]
                    if gap > 1 and gap < threshold:
                        new_indices.extend(range(interaction_indices[i] + 1, interaction_indices[i + 1]))
                new_indices.append(interaction_indices[-1])
                interaction_indices = np.array(new_indices, dtype=np.int64)
        
        # Assign results
        self.interaction_indices = interaction_indices
        self.stalk_force = self.force[interaction_indices]
        self.stalk_position = self.position[interaction_indices]
        self.stalk_time = self.time[interaction_indices]

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
        # print(f'Computing stiffness for {self.stalk_type} stalks')
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
            
    def plot_force_position(self, view_stalks=False, plain=True):
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.8))
        ax[0].plot(self.time, self.force, label='Force')
        ax[0].set_ylabel('Force (N)')
        ax[1].plot(self.time, self.position*100, label='Position')
        if hasattr(self, 'raw_force'):
            ax[0].plot(self.time, self.raw_force, label='raw', linewidth=0.5)
            ax[1].plot(self.time, self.raw_position, label='raw', linewidth=0.5)
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Position (cm)')

        plt.suptitle(f'{self.configuration}, Date:{self.date}, Test #{self.test_num}\nStalks:{self.stalk_type}')

        
        if view_stalks:
            for i in range(len(self.stalk_times)):
                if not np.isnan(self.stalk_times[i]).all():
                    ax[0].plot(self.stalk_times[i], self.stalk_forces[i], c='red')
                    ax[0].plot(self.stalk_times[i], self.force_fits[i], c='green')
                    ax[1].plot(self.stalk_times[i], self.stalk_positions[i]*100, c='red')
                    ax[1].plot(self.stalk_times[i], self.position_fits[i]*100, c='green')
            plt.tight_layout()
        # For verifying calibration coefficients
        elif not plain:
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

    def clear_intermediate_data(self):
        self.strain_a1 = None
        self.strain_b1 = None
        self.strain_a2 = None
        self.strain_b2 = None
        self.strain_a1_raw = None
        self.strain_b1_raw = None
        self.strain_a2_raw = None
        self.strain_b2_raw = None
        self.force = None
        self.position = None
        self.forceDT = None
        self.positionDT = None
        self.forceDDT = None
        self.positionDDT = None
        self.interaction_indices = None
        self.stalk_force = None
        self.stalk_position = None
        self.stalk_time = None
        self.stalk_forces = None
        self.stalk_positions = None
        self.stalk_times = None
        self.force_fits = None
        self.position_fits = None

def boxplot_data(rodney_config, date=None, stalk_type=None, plot_num=20):
    results_df = pd.read_csv(r'Results\results.csv')
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    plt.figure(plot_num)
    plt.title(rodney_config)
    plt.ylabel('Flexural Stiffness (N/m^2)')
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
    plt.ylim(0, 50)

def get_stats(rodney_config, date=None, stalk_type=None, plot_num=None):
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        # sem = stats.sem(data)
        std = np.std(data, ddof=1)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std)
        margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * std
        rel_margin = margin / type_mean if type_mean != 0 else 0  # Relative margin
        return {'interval': ci, 'mean': mean, 'margin': margin, 'rel_margin': rel_margin}
    
    results_df = pd.read_csv(r'Results\results.csv')
    if not date == None:
        results_df = results_df[results_df['Date'] == date]
    if not stalk_type == None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
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
    all_relMargins_median = np.nanmedian(all_relMargins)

    medhi_relMargins = np.append(med_relMargins, hi_relMargins)
    medhi_relMargins_mean = np.nanmean(medhi_relMargins)
    medhi_relMargins_median = np.nanmedian(medhi_relMargins)
    # print(all_relMargins_median)

    if plot_num is not None:
        plt.figure(plot_num)
        plt.scatter(range(1, len(lo_relMargins)+1), lo_relMargins*100, label='lo', c='red')
        plt.scatter(range(1, len(med_relMargins)+1), med_relMargins*100, label='med', c='green')
        plt.scatter(range(1, len(hi_relMargins)+1), hi_relMargins*100, label='hi', c='blue')
        plt.axhline(all_relMargins_median*100, c='black', label='Median')
        plt.axhline(medhi_relMargins_median*100, c='brown', label='M/H Median')
        plt.ylim(0, 80)
        plt.ylabel('% Relative Error Margin')
        plt.title(rodney_config + f', Median: {all_relMargins_median*100:.1f}')
        plt.legend()

    return all_relMargins_mean, all_relMargins_median

# Optimization functions
def get_all_tests(date):
    test_nums = []
    for file in os.listdir(f'Raw Data/{date}'):
        if file.endswith('.csv') and 'test' in file:
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

# Helper function to load optimized parameters
def load_optimized_parameters(rodney_config):
    param_file = 'Results/optimized_parameters.csv'
    default_params = {
        'lo_pos_accel_tol': 1.0,
        'med_pos_accel_tol': 0.5,
        'hi_pos_accel_tol': 4.0,
        'lo_force_accel_tol': 25.0,
        'med_force_accel_tol': 30.0,
        'hi_force_accel_tol': 150.0
    }
    if os.path.isfile(param_file):
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            row = df[df['rodney_config'] == rodney_config].iloc[0]
            return {
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
                       lo_pos_accel_tol=params['lo_pos_accel_tol'],
                       med_pos_accel_tol=params['med_pos_accel_tol'],
                       hi_pos_accel_tol=params['hi_pos_accel_tol'],
                       lo_force_accel_tol=params['lo_force_accel_tol'],
                       med_force_accel_tol=params['med_force_accel_tol'],
                       hi_force_accel_tol=params['hi_force_accel_tol'])
    test.smooth_strains(window=20, order=1)
    test.correct_linear_drift()
    test.shift_initials()
    test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
    test.differentiate_force_position(smooth=True, window=40, order=1)
    test.differentiate_force_position_DT(smooth=True, window=40, order=1)
    test.find_stalk_interaction()
    test.collect_stalk_sections()
    test.calc_stalk_stiffness()

    if view:
        if overwrite:
            test.save_results(overwrite_result=True)
        test.plot_force_position(view_stalks=True)
        # test.plot_raw_strain()
        # test.plot_force_position_DT()
        # test.plot_force_position_DDT()
    else:
        test.save_results(overwrite_result=True)
    
    if not local_run_flag:
        plt.show()

def optimize_parameters(dates, rodney_config):
    '''This optimizes the filtering paramters to provide the lowest average relative margin of error for all stalk types
    of a specified configuration. However, the median value is reported in other places. This is because using the 
    median as the objective score allows the optimizer to give bad results for troublesome data and effectively ignore
    it. In the worst case, it can choose values that result in non-detections of bad stalk interactions, whose results
    are recorded as np.nan and do not affect later statistics used to provide the objective score.
    
    There are 6 parameters to optimize. The optimizer samples the field with 30 combinations, and then selectively targets
    areas to find the optimal combination. This exploration takes a while, and notable imporovements don't happen until a 
    while into the process. This function will report the best value after each objective call, and after 120 combinations, 
    it will start counting how many calls were made without improvement, stopping after 30 calls with no improvement. So, 
    this will run at least 150 combinations, up to 200'''
    
    import gc
    from psutil import Process, HIGH_PRIORITY_CLASS
    from os import getpid

    inst = Process(getpid())
    inst.nice(HIGH_PRIORITY_CLASS)


    # Precompute data for all tests
    class PrecomputedTest:
        def __init__(self, date, test_num):
            test = LabStalkRow(date=date, test_num=test_num)
            test.smooth_strains(window=20, order=1)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.differentiate_force_position(smooth=True, window=40, order=1)
            test.differentiate_force_position_DT(smooth=True, window=40, order=1)
            self.time = test.time
            self.force = test.force
            self.position = test.position
            self.forceDT = test.forceDT
            self.positionDT = test.positionDT
            self.forceDDT = test.forceDDT
            self.positionDDT = test.positionDDT
            self.stalk_type = test.stalk_type
            self.min_position = test.min_position
            self.max_position = test.max_position
            self.min_force = test.min_force
            self.min_force_rate = test.min_force_rate
            self.min_sequential = test.min_sequential
            self.height = test.height
            self.yaw = test.yaw
            self.results_path = test.results_path
            self.csv_path = test.csv_path
            self.header_rows = test.header_rows
    
    all_tests = []
    for date in dates:
        test_nums = get_tests_for_config(date, rodney_config)
        for test_num in test_nums:
            all_tests.append((date, test_num))
    print(f'Found {len(all_tests)} files for {rodney_config}')
    
    precomputed_tests = [PrecomputedTest(date, test_num) for date, test_num in all_tests]
    
    # Define the search space
    search_space = [
        Real(0.5, 3.5, name='lo_pos_accel_tol'),
        Real(0.5, 3.5, name='med_pos_accel_tol'),
        Real(1.0, 5.0, name='hi_pos_accel_tol'),
        Real(30, 70, name='lo_force_accel_tol'),
        Real(40, 100, name='med_force_accel_tol'),
        Real(40, 100, name='hi_force_accel_tol')
    ]
    
    # Objective function using precomputed data
    call = [0]
    scores = []
    best = [10]
    count = [0]
    def objective(params):
        call[0] += 1
        global results
        results.clear()
        lo_pos_accel_tol, med_pos_accel_tol, hi_pos_accel_tol, \
        lo_force_accel_tol, med_force_accel_tol, hi_force_accel_tol = params
        
        for precomp in precomputed_tests:
            # Extract date and test_num using os.path
            base_name = os.path.basename(precomp.csv_path)
            date = base_name.split('_test_')[0]
            test_num = base_name.split('_test_')[1].split('.csv')[0]
            
            test = LabStalkRow(date=date, test_num=test_num)
            test.time = precomp.time
            test.force = precomp.force
            test.position = precomp.position
            test.forceDT = precomp.forceDT
            test.positionDT = precomp.positionDT
            test.forceDDT = precomp.forceDDT
            test.positionDDT = precomp.positionDDT
            test.stalk_type = precomp.stalk_type
            test.min_position = precomp.min_position
            test.max_position = precomp.max_position
            test.min_force = precomp.min_force
            test.min_force_rate = precomp.min_force_rate
            test.min_sequential = precomp.min_sequential
            test.height = precomp.height
            test.yaw = precomp.yaw
            test.results_path = precomp.results_path
            test.csv_path = precomp.csv_path
            test.header_rows = precomp.header_rows
            
            if test.stalk_type == 'lo':
                test.pos_accel_tol = lo_pos_accel_tol
                test.force_accel_tol = lo_force_accel_tol
            elif test.stalk_type == 'med':
                test.pos_accel_tol = med_pos_accel_tol
                test.force_accel_tol = med_force_accel_tol
            elif test.stalk_type == 'hi':
                test.pos_accel_tol = hi_pos_accel_tol
                test.force_accel_tol = hi_force_accel_tol
            
            test.find_stalk_interaction()
            test.collect_stalk_sections()
            test.calc_stalk_stiffness()
            test.save_results(overwrite_result=True)
            test.clear_intermediate_data()
            del test
            gc.collect()
        
        score = get_stats(rodney_config)[0]
        scores.append(score)
        if call[0] > 120:
            count[0] += 1
        if score < best[0] and score > 0:
            best[0] = score
            count[0] = 0
        print(f"Parameters: {[f'{p:.3f}' for p in params]}, Score: {score:.6f}, Best: {best[0]:.6f}")
        return score
    
    # Early stopping callback
    def early_stopping(res):
        if len(scores) >= 30:
            print(call[0], count[0])
            if count[0] > 30 and call[0] >= 120:
                print("Early stopping: Score improvement too low")
                return True
        return False
    
    # Perform Bayesian optimization
    print("Starting Bayesian optimization")
    result = gp_minimize(objective, search_space, n_initial_points=30, n_calls=300, callback=[early_stopping])
    
    # Extract and save best parameters
    best_params = result.x
    best_score = result.fun
    param_file = 'Results/optimized_parameters.csv'
    headers = ['rodney_config', 'lo_pos_accel_tol', 'med_pos_accel_tol',
               'hi_pos_accel_tol', 'lo_force_accel_tol', 'med_force_accel_tol',
               'hi_force_accel_tol', 'best_score']
    row = [rodney_config] + best_params + [best_score]
    
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
          f"lo_pos_accel_tol={best_params[0]:.4f}, "
          f"med_pos_accel_tol={best_params[1]:.4f}, "
          f"hi_pos_accel_tol={best_params[2]:.4f}, "
          f"lo_force_accel_tol={best_params[3]:.4f}, "
          f"med_force_accel_tol={best_params[4]:.4f}, "
          f"hi_force_accel_tol={best_params[5]:.4f}, "
          f"Best score: {best_score:.6f}")

def optimize_parameters2(dates, rodney_config, stalk_type):
    from psutil import Process, HIGH_PRIORITY_CLASS
    from os import getpid
    import gc
    '''Optimizes filtering parameters for a specified stalk type to minimize the average relative margin of error.
    
    Args:
        dates: List of dates to process.
        rodney_config: Configuration identifier.
        stalk_type: Type of stalk to optimize ('lo', 'med', or 'hi').
    
    The function optimizes two parameters for the specified stalk type, using existing or default values for others.
    It updates only those two parameters and the corresponding score in optimized_parameters.csv.
    '''
    
    inst = Process(getpid())
    inst.nice(HIGH_PRIORITY_CLASS)

    # Precompute data for all tests
    class PrecomputedTest:
        def __init__(self, date, test_num):
            test = LabStalkRow(date=date, test_num=test_num)
            test.smooth_strains(window=20, order=1)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.differentiate_force_position(smooth=True, window=40, order=1)
            test.differentiate_force_position_DT(smooth=True, window=40, order=1)
            self.time = test.time
            self.force = test.force
            self.position = test.position
            self.forceDT = test.forceDT
            self.positionDT = test.positionDT
            self.forceDDT = test.forceDDT
            self.positionDDT = test.positionDDT
            self.stalk_type = test.stalk_type
            self.min_position = test.min_position
            self.max_position = test.max_position
            self.min_force = test.min_force
            self.min_force_rate = test.min_force_rate
            self.min_sequential = test.min_sequential
            self.height = test.height
            self.yaw = test.yaw
            self.results_path = test.results_path
            self.csv_path = test.csv_path
            self.header_rows = test.header_rows
    
    all_tests = []
    for date in dates:
        test_nums = get_tests_for_config(date, rodney_config)
        for test_num in test_nums:
            all_tests.append((date, test_num))
    print(f'Found {len(all_tests)} files for {rodney_config}')
    
    precomputed_tests = [PrecomputedTest(date, test_num) for date, test_num in all_tests]
    
    # Define default parameters
    DEFAULT_POS_ACCEL_TOL = 2.0
    DEFAULT_FORCE_ACCEL_TOL = 50
    
    # Get existing parameters if available
    def get_existing_params(rodney_config):
        param_file = 'Results/optimized_parameters.csv'
        if os.path.isfile(param_file):
            df = pd.read_csv(param_file)
            if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
                row = df[df['rodney_config'] == rodney_config].iloc[0]
                return {
                    'lo_pos_accel_tol': row.get('lo_pos_accel_tol', np.nan),
                    'med_pos_accel_tol': row.get('med_pos_accel_tol', np.nan),
                    'hi_pos_accel_tol': row.get('hi_pos_accel_tol', np.nan),
                    'lo_force_accel_tol': row.get('lo_force_accel_tol', np.nan),
                    'med_force_accel_tol': row.get('med_force_accel_tol', np.nan),
                    'hi_force_accel_tol': row.get('hi_force_accel_tol', np.nan)
                }
        return {
            'lo_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'med_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'hi_pos_accel_tol': DEFAULT_POS_ACCEL_TOL,
            'lo_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL,
            'med_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL,
            'hi_force_accel_tol': DEFAULT_FORCE_ACCEL_TOL
        }
    
    existing_params = get_existing_params(rodney_config)
    
    # Define the search space based on stalk_type
    if stalk_type == 'lo':
        search_space = [
            Real(0.5, 3.5, name='lo_pos_accel_tol'),
            Real(30, 70, name='lo_force_accel_tol')
        ]
    elif stalk_type == 'med':
        search_space = [
            Real(0.5, 3.5, name='med_pos_accel_tol'),
            Real(40, 100, name='med_force_accel_tol')
        ]
    elif stalk_type == 'hi':
        search_space = [
            Real(1.0, 5.0, name='hi_pos_accel_tol'),
            Real(40, 100, name='hi_force_accel_tol')
        ]
    else:
        raise ValueError("Invalid stalk_type. Must be 'lo', 'med', or 'hi'.")
    
    # Objective function using precomputed data
    call = [0]
    scores = []
    best = [10]
    count = [0]
    def objective(params):
        call[0] += 1
        global results
        results.clear()
        
        for precomp in precomputed_tests:
            base_name = os.path.basename(precomp.csv_path)
            date = base_name.split('_test_')[0]
            test_num = base_name.split('_test_')[1].split('.csv')[0]
            
            test = LabStalkRow(date=date, test_num=test_num)
            test.time = precomp.time
            test.force = precomp.force
            test.position = precomp.position
            test.forceDT = precomp.forceDT
            test.positionDT = precomp.positionDT
            test.forceDDT = precomp.forceDDT
            test.positionDDT = precomp.positionDDT
            test.stalk_type = precomp.stalk_type
            test.min_position = precomp.min_position
            test.max_position = precomp.max_position
            test.min_force = precomp.min_force
            test.min_force_rate = precomp.min_force_rate
            test.min_sequential = precomp.min_sequential
            test.height = precomp.height
            test.yaw = precomp.yaw
            test.results_path = precomp.results_path
            test.csv_path = precomp.csv_path
            test.header_rows = precomp.header_rows
            
            if test.stalk_type == stalk_type:
                test.pos_accel_tol = params[0]
                test.force_accel_tol = params[1]
                test.find_stalk_interaction()
                test.collect_stalk_sections()
                test.calc_stalk_stiffness()
                test.save_results(overwrite_result=True)
            else:
                pos_key = f"{test.stalk_type}_pos_accel_tol"
                force_key = f"{test.stalk_type}_force_accel_tol"
                pos_tol = existing_params.get(pos_key, DEFAULT_POS_ACCEL_TOL)
                force_tol = existing_params.get(force_key, DEFAULT_FORCE_ACCEL_TOL)
                test.pos_accel_tol = pos_tol if not np.isnan(pos_tol) else DEFAULT_POS_ACCEL_TOL
                test.force_accel_tol = force_tol if not np.isnan(force_tol) else DEFAULT_FORCE_ACCEL_TOL
            
            
            test.clear_intermediate_data()
            del test
            gc.collect()
        
        score = get_stats2(rodney_config, stalk_type=stalk_type)[0]
        scores.append(score)
        if call[0] > 20:
            count[0] += 1
        if score < best[0]:
            best[0] = score
            count[0] = 0
        print(f"Parameters: {[f'{p:.3f}' for p in params]}, Score: {score:.6f}, Best: {best[0]:.6f}")
        return score
    
    # Early stopping callback (corrected to stop after 30 calls without improvement)
    def early_stopping(res):
        print(call[0], count[0])
        if len(scores) >= 20:
            if count[0] > 20 and call[0] >= 50:
                print("Early stopping: Score improvement too low")
                return True
        return False
    
    # Perform Bayesian optimization
    print(f"Starting Bayesian optimization for {stalk_type}")
    result = gp_minimize(objective, search_space, n_initial_points=20, n_calls=200, callback=[early_stopping])
    
    # Extract and save best parameters
    best_params = result.x
    best_score = result.fun
    param_file = 'Results/optimized_parameters.csv'
    headers = ['rodney_config', 'lo_pos_accel_tol', 'med_pos_accel_tol', 'hi_pos_accel_tol',
               'lo_force_accel_tol', 'med_force_accel_tol', 'hi_force_accel_tol',
               'lo_score', 'med_score', 'hi_score']
    
    file_exists = os.path.isfile(param_file)
    if file_exists:
        df = pd.read_csv(param_file)
        if 'rodney_config' in df.columns and rodney_config in df['rodney_config'].values:
            idx = df[df['rodney_config'] == rodney_config].index[0]
            if stalk_type == 'lo':
                df.at[idx, 'lo_pos_accel_tol'] = best_params[0]
                df.at[idx, 'lo_force_accel_tol'] = best_params[1]
                df.at[idx, 'lo_score'] = best_score
            elif stalk_type == 'med':
                df.at[idx, 'med_pos_accel_tol'] = best_params[0]
                df.at[idx, 'med_force_accel_tol'] = best_params[1]
                df.at[idx, 'med_score'] = best_score
            elif stalk_type == 'hi':
                df.at[idx, 'hi_pos_accel_tol'] = best_params[0]
                df.at[idx, 'hi_force_accel_tol'] = best_params[1]
                df.at[idx, 'hi_score'] = best_score
            df.to_csv(param_file, index=False)
        else:
            new_row = pd.DataFrame([[rodney_config] + [np.nan]*9], columns=headers)
            if stalk_type == 'lo':
                new_row.at[0, 'lo_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'lo_force_accel_tol'] = best_params[1]
                new_row.at[0, 'lo_score'] = best_score
            elif stalk_type == 'med':
                new_row.at[0, 'med_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'med_force_accel_tol'] = best_params[1]
                new_row.at[0, 'med_score'] = best_score
            elif stalk_type == 'hi':
                new_row.at[0, 'hi_pos_accel_tol'] = best_params[0]
                new_row.at[0, 'hi_force_accel_tol'] = best_params[1]
                new_row.at[0, 'hi_score'] = best_score
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(param_file, index=False)
    else:
        new_row = pd.DataFrame([[rodney_config] + [np.nan]*9], columns=headers)
        if stalk_type == 'lo':
            new_row.at[0, 'lo_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'lo_force_accel_tol'] = best_params[1]
            new_row.at[0, 'lo_score'] = best_score
        elif stalk_type == 'med':
            new_row.at[0, 'med_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'med_force_accel_tol'] = best_params[1]
            new_row.at[0, 'med_score'] = best_score
        elif stalk_type == 'hi':
            new_row.at[0, 'hi_pos_accel_tol'] = best_params[0]
            new_row.at[0, 'hi_force_accel_tol'] = best_params[1]
            new_row.at[0, 'hi_score'] = best_score
        new_row.to_csv(param_file, index=False)
    
    print(f"Best parameters for {rodney_config} ({stalk_type}): "
          f"{stalk_type}_pos_accel_tol={best_params[0]:.4f}, "
          f"{stalk_type}_force_accel_tol={best_params[1]:.4f}, "
          f"Best score: {best_score:.6f}")

def get_stats2(rodney_config, date=None, stalk_type=None, plot_num=None):
    '''Computes statistics for stalk stiffness measurements.
    
    Args:
        rodney_config: Configuration identifier.
        date: Optional date to filter results.
        stalk_type: Optional stalk type to filter results ('lo', 'med', or 'hi').
        plot_num: Optional plot number for visualization.
    
    Returns:
        Tuple of mean and median of relative margins of error.
    '''
    def get_ci(data, type_mean, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=std / np.sqrt(n))
        margin = stats.t.ppf((1 + confidence) / 2, df=n-1) * std / np.sqrt(n)
        rel_margin = margin / type_mean if type_mean != 0 else 0
        return {'interval': ci, 'mean': mean, 'margin': margin, 'rel_margin': rel_margin}
    
    results_df = pd.read_csv(r'Results\results.csv')
    if date is not None:
        results_df = results_df[results_df['Date'] == date]
    if stalk_type is not None:
        results_df = results_df[results_df['stalk array (lo med hi)'] == stalk_type]
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    
    if config_results.empty:
        return np.nan, np.nan
    
    EIs = config_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 
                          'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    stats_df = EIs.describe()
    mean_EI = np.mean(stats_df.loc['mean'])
    ci_dict = {col: get_ci(EIs[col], mean_EI) for col in EIs.columns}
    rel_margins = np.array([ci_dict[col]['rel_margin'] for col in EIs.columns])
    
    all_relMargins_mean = np.nanmean(rel_margins)
    all_relMargins_median = np.nanmedian(rel_margins)
    
    if plot_num is not None:
        plt.figure(plot_num)
        plt.scatter(range(1, len(rel_margins)+1), rel_margins*100, 
                    label=stalk_type, 
                    c='red' if stalk_type=='lo' else 'green' if stalk_type=='med' else 'blue')
        plt.axhline(all_relMargins_median*100, c='black', label='Median')
        plt.ylim(0, 80)
        plt.ylabel('% Relative Error Margin')
        plt.title(f'{rodney_config} ({stalk_type}), Median: {all_relMargins_median*100:.1f}')
        plt.legend()
    
    return all_relMargins_mean, all_relMargins_median

def show_force_position(dates, test_nums, stalk_id=None, rodney_config=None):
    for date in dates:
        for test_num in test_nums:
            config = get_config(date, test_num)
            params = load_optimized_parameters(config)
            test = LabStalkRow(date=date, test_num=test_num,
                       lo_pos_accel_tol=params['lo_pos_accel_tol'],
                       med_pos_accel_tol=params['med_pos_accel_tol'],
                       hi_pos_accel_tol=params['hi_pos_accel_tol'],
                       lo_force_accel_tol=params['lo_force_accel_tol'],
                       med_force_accel_tol=params['med_force_accel_tol'],
                       hi_force_accel_tol=params['hi_force_accel_tol'])
            test.smooth_strains(window=20, order=1)
            test.correct_linear_drift()
            test.shift_initials()
            test.calc_force_position(smooth=True, window=40, order=1, small_den_cutoff=0.00006)
            test.plot_force_position(view_stalks=False, plain=True)
            test.plot_raw_strain()
    plt.show()

if __name__ == "__main__":
    local_run_flag = True
    
    '''Batch run of same configuration'''
    for i in range(1, 45+1):
        process_data(date='07_24', test_num=f'{i}', view=True, overwrite=True)
    # show_force_position(dates=['07_24'], test_nums=range(1, 15+1))

    # boxplot_data(rodney_config='Integrated Beam Prototype 1', date='07_03', plot_num=104)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=105)
    # boxplot_data(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=106)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=107)
    # boxplot_data(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=108)
    # boxplot_data(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=108)
    boxplot_data(rodney_config='Integrated Beam Fillet 1', date='07_24', plot_num=108)
    '''end batch run'''

    '''Statistics'''
    # print('1 mean, median', get_stats(rodney_config='Integrated Beam Prototype 1', plot_num=204))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_10', plot_num=205))
    # print('2 mean, median', get_stats(rodney_config='Integrated Beam Prototype 2', date='07_14', plot_num=206))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_10', plot_num=207))
    # print('3 mean, median', get_stats(rodney_config='Integrated Beam Prototype 3', date='07_11', plot_num=208))
    # print('mean, median', get_stats(rodney_config='Integrated Beam Printed Guide 1', date='07_16', plot_num=208))
    print('mean, median', get_stats(rodney_config='Integrated Beam Fillet 1', date='07_24', plot_num=208))
    '''end statistics'''

    '''Single file run and view full file. Does not save result'''
    # process_data(date='07_14', test_num='1', view=True)
    '''end single file run'''

    # Optimize parameters for a specific configuration
    # optimize_parameters(dates=['07_16'], rodney_config='Integrated Beam Printed Guide 1')

    plt.show()