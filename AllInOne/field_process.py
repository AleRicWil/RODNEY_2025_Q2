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

class FieldStalkSection:
    def __init__(self, date, test_num, min_force_rate=0.01, pos_accel_tol=5.0, force_accel_tol=100):
        # These params set the filter bounds for identifying which portions of the data are stalk interactions. These are ideally straight lines, increasing in
        # force and decreasing in position. 
            # this window should be a bit wider than the physical sensor
        self.min_position = 5*1e-2  # centimeters, location of 2nd strain gauge
        self.max_position = 18*1e-2 # centimeters, location of beam end
        self.min_force = 1 # newton, easily cut out the spaces between stalks. Rejects noisy detection regime
        self.min_force_rate = min_force_rate    # newton/sec, only look at data where the stalk is being pushed outward
        self.pos_accel_tol = pos_accel_tol  # m/s^2, reject curvature on either side of good stalk interaction 
        self.force_accel_tol = force_accel_tol # newton/s^2, reject curvature on either side of good stalk interaction
        self.min_seq_points = 10    # don't consider little portions of data that pass the initial filter
        self.stitch_gap_limit = 20  # if two good segments are close together, stitch into one segment (including the gap)

        # Results folder
        parent_folder = r'Results'
        os.makedirs(parent_folder, exist_ok=True)   # create the folder if it doesn't already exist
        self.results_path = os.path.join(parent_folder, r'field_results.csv')   # one file to store all results from all dates/sections
            # these are used to find the file, and written alongside the results in the results file
        self.date = date   
        self.test_num = test_num

        # Load data and stored calibration from data collection CSV header
        self.csv_path = rf'Raw Data\{date}\{date}_test_{test_num}.csv'
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)  # create an object which handles the CSV operations
            self.header_rows = []
            for row in reader:  # grab all the header rows
                if row[0] == '=====':   # this string (in the first column) divides the header from the data
                    break
                self.header_rows.append(row)
            
            params_read = 0 # track how many parameters have been read
            for row in self.header_rows:
                    # the first column of each row is the parameter name. The second column is the parameter's value
                if row[0] == "rodney configuration":
                    self.configuration = row[1]
                    params_read += 1
                if row[0] == "sensor calibration (k d c)":
                        # c values are not used, instead calculated from initial idle values of each data collection
                        # read in the strings and convert to floats
                    k_str = row[1]
                    d_str = row[2]
                    k_values = [float(v) for v in k_str.split()]
                    d_values = [float(v) for v in d_str.split()]
                    self.k_A1, self.k_B1, self.k_A2, self.k_B2 = k_values
                    self.d_A1, self.d_B1, self.d_A2, self.d_B2 = d_values
                    params_read += 1
                if row[0] == "stalk array (lo med hi)":
                        # this is which block or section in the field (or synthetic stalk type in the lab)
                    self.stalk_type = row[1]
                    params_read += 1
                if row[0] == "sensor height (cm)":
                    self.height = float(row[1])*1e-2    # meters in this code, stored as cm in header
                    params_read += 1
                if row[0] == "sensor yaw (degrees)":
                    self.yaw = np.radians(float(row[1]))    # radians in this code
                    params_read += 1
                if row[0] == "sensor offset (cm to gauge 2)":
                    self.sensor_offset = float(row[1])*1e-2 # meters in this code, but it isn't used anywhere
                    params_read += 1
            if not params_read >= 6:
                raise ValueError("Test parameter rows missing in header")
            
            # Read the data
            data = pd.read_csv(f)   # load the CSV with pandas from where the reader object left off. Must be done within 'with open() as f:' scope

            # store each column by its title and covert the pandas object to a numpy array
        self.time = data['Time'].to_numpy()
        self.strain_1 = self.strain_1_raw = data['Strain A1'].to_numpy()    # CSV labels as A1/A2 for backwards compatability with 4 channel sensors. This code
        self.strain_2 = self.strain_2_raw = data['Strain A2'].to_numpy()    # only uses the two channels
            # check if this data collection had an accelerometer
        self.acce_flag = False  #
        if 'AcX1' in data.columns:
            self.acce_flag = True
            self.acX = self.acX_raw = data['AcX1'].to_numpy()   # CSV is set up to allow two accelerometers
            self.acY = self.acY_raw = data['AcY1'].to_numpy()
            self.acZ = self.acZ_raw = data['AcZ1'].to_numpy()

    def smooth_raw_data(self, strain_window=20, strain_order=1, accel_window=50, accel_order=1):
        self.strain_1 = savgol_filter(self.strain_1, strain_window, strain_order)
        self.strain_2 = savgol_filter(self.strain_2, strain_window, strain_order)
        if self.accel_flag:
            self.acX = savgol_filter(self.acX, accel_window, accel_order)
            self.acY = savgol_filter(self.acY, accel_window, accel_order)
            self.acZ = savgol_filter(self.acZ, accel_window, accel_order)

    def differentiate_accels(self, smooth=False, window = 1000, order=2):
        self.acX_DT = np.gradient(self.acX, self.time)
        self.acY_DT = np.gradient(self.acY, self.time)
        self.acZ_DT = np.gradient(self.acZ, self.time)
        if smooth:
            self.smooth_accel_DTs(window, order)
        
    def smooth_accel_DTs(self, window=1000, order=2):
        self.acX_DT = savgol_filter(self.acX_DT, window, order)
        self.acY_DT = savgol_filter(self.acY_DT, window, order)
        self.acZ_DT = savgol_filter(self.acZ_DT, window, order)

