import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from scipy.signal import savgol_filter

local_run_flag = False
results = []

class LabStalkRow:
    def __init__(self, date, test_num):
        self.min_position = 6 * 1e-2    # centimeters
        self.max_position = 16 * 1e-2   # centimeters
        self.min_force = 2  # Newtons
        self.min_force_rate = 0.5  # Newton/second
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
                        self.accel_tol = 1.0
                    elif row[1] == "med":
                        self.result_color = 'green'
                        self.accel_tol = 0.3
                    elif row[1] == "hi":
                        self.result_color = 'blue'
                        self.accel_tol = 0.3
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
        self.near_zero_accel_indices = np.where(np.abs(self.positionDDT) < self.accel_tol)[0]
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

    def save_results(self):
        # Prepare row data
        row = [self.date, self.csv_path]
        for header_row in self.header_rows:
            if header_row[0] == "sensor calibration (k d c)":
                row.append(header_row[1] + ' : ' + header_row[2] + ' : ' + header_row[3])
            else:
                row.append(header_row[1] if len(header_row) > 1 else '')
        for stalk in self.flex_stiffs:
            row.append(stalk)

        # Check if file exists and if csv_path is already in the second column
        file_exists = os.path.isfile(self.results_path)
        write_row = True
        if file_exists:
            df = pd.read_csv(self.results_path)
            if 'File' in df.columns and self.csv_path in df['File'].values:
                write_row = False

        # Write to CSV if row should be written
        if write_row:
            with open(self.results_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if not file_exists:
                    headers = ['Date', 'File'] + [row[0] for row in self.header_rows if row[0] != "====="]
                    headers += ['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']
                    csvwriter.writerow(headers)
                csvwriter.writerow(row)

def process_data(date, test_num, view=False):
    test = LabStalkRow(date=date, test_num=test_num)
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
        test.plot_force_position(view_flag=view)
        test.plot_raw_strain()
        # test.plot_force_position_DT()
        # test.plot_force_position_DDT()
    else:
        test.save_results()
    
    
    if not local_run_flag:
        plt.show()

def boxplot_data(rodney_config):
    results_df = pd.read_csv(r'Results\results.csv')
    config_results = results_df[results_df['rodney configuration'] == rodney_config]
    lo_results = config_results[config_results['stalk array (lo med hi)'] == 'lo']
    med_results = config_results[config_results['stalk array (lo med hi)'] == 'med']
    hi_results = config_results[config_results['stalk array (lo med hi)'] == 'hi']

    lo_EIs = lo_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    med_EIs = med_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]
    hi_EIs = hi_results[['Stalk1', 'Stalk2', 'Stalk3', 'Stalk4', 'Stalk5', 'Stalk6', 'Stalk7', 'Stalk8', 'Stalk9']]

    plt.figure(20)
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
    
if __name__ == "__main__":
    local_run_flag = True
    
    '''Batch run of same configuration'''
    for i in range(11, 40+1):
        process_data(date='07_03', test_num=f'{i}')

    boxplot_data(rodney_config='Integrated Beam Prototype 1')
    '''end batch run'''


    '''Single file run and view full file. Does not save result'''
    # process_data(date='07_10', test_num='1', view=True)
    '''end single file run'''

    plt.show()