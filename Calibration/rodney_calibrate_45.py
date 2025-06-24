import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

class RODNEY_Calibration():
    def __init__(self, directory, defaults_flag=True):
        self.directory = directory
        self.get_filenames_in_directory()
        self.load_data_from_csv_files()

        self.defaults_flag = defaults_flag
        self.set_test_arrays()

    def get_filenames_in_directory(self):
        self.filenames = []
        for filename in os.listdir(f'{os.getcwd()}/{self.directory}'):
            if filename.endswith(".csv"):
                self.filenames.append(f'{self.directory}/{filename}')
 
    def load_data_from_csv_files(self):
        """Reads the first 100 rows of data after the header of each CSV."""
        self.data = []  # Initialize list to store DataFrames
        for filename in self.filenames:
            header_row = None
            # Search for the row starting with "Time (Microseconds)"
            with open(filename, 'r') as f:
                for i, line in enumerate(f):
                    if line.strip().startswith("Time (Microseconds)"):
                        header_row = i
                        break
            # If header row not found, skip the file with a warning
            if header_row is None:
                print(f"Warning: Header row not found in {filename}")
                continue
            # Read 100 rows of data using the found row as column names
            df = pd.read_csv(filename, header=header_row, nrows=100)
            self.data.append(df)

    def set_test_arrays(self):
        if self.defaults_flag:
            self.num_weights = 4
            self.num_positions = 5
            self.weight_vals = [0.25, 0.5, 1.0, 2.0]
            self.position_vals = [10, 12, 14, 16, 18]
        else:
            self.num_weights = int(input("Enter number of weights: "))
            self.weight_vals = [float(x) for x in input("Enter weight values (space-separated): ").split()]
            self.num_positions = int(input("Enter number of positions: "))
            self.position_vals = [float(x) for x in input("Enter position values (space-separated): ").split()]
        
        self.test_weights = np.array([w for w in self.weight_vals for _ in range(self.num_positions)])
        self.test_positions = np.array([p for _ in range(self.num_weights) for p in self.position_vals])

    def pre_process_data(self):
        if not hasattr(self, 'data'):
            self.get_data_from_csv_files()
        self.averages = []
        for df in self.data:
            avg_dict = {
                'Strain Ax': df['Strain Ax'].mean(),
                'Strain Bx': df['Strain Bx'].mean(),
                'Strain Ay': df['Strain Ay'].mean(),
                'Strain By': df['Strain By'].mean()
            }
            self.averages.append(avg_dict)
    
    def calc_calibration_constants(self):
        if not hasattr(self, 'averages'):
            self.pre_process_data()
        
        # Prepare arrays based on test structure
        x = np.array([p / 100 for p in self.test_positions])  # Convert to meters
        mass = np.array(self.test_weights)
        F = mass * 9.7974  # Force in Newtons with g for Provo, UT
        print(F)
        ind = range(1, self.num_weights * self.num_positions + 1)
        
        # Extract average strains for all tests
        V_Ax_cal = np.array([avg['Strain Ax'] for avg in self.averages])
        V_Bx_cal = np.array([avg['Strain Bx'] for avg in self.averages])
        V_Ay_cal = np.array([avg['Strain Ay'] for avg in self.averages])
        V_By_cal = np.array([avg['Strain By'] for avg in self.averages])

        plt.scatter(ind, V_Ax_cal)
        plt.scatter(ind, V_Bx_cal)
        plt.scatter(ind, V_Ay_cal)
        plt.scatter(ind, V_By_cal)
        plt.show()

        # Process each strain type with grouped regression
        strains = {'Ax': V_Ax_cal, 'Bx': V_Bx_cal, 'Ay': V_Ay_cal, 'By': V_By_cal}
        results = {}
        for strain_name, V in strains.items():
            k_vals, d_vals, c_vals = [], [], []
            for i in range(0, len(V), self.num_positions):  # Step by num_positions for each weight group
                V_group = V[i:i+self.num_positions]
                x_group = x[i:i+self.num_positions]
                F_group = F[i:i+self.num_positions]
                A = np.vstack([F_group * x_group, -F_group]).T
                model = LinearRegression().fit(A, V_group)
                c = model.intercept_
                k = model.coef_[0]
                d = model.coef_[1] / k if k != 0 else 0
                k_vals.append(k)
                d_vals.append(d)
                c_vals.append(c)
            results[strain_name] = {
                'k': np.mean(k_vals),
                'd': np.mean(d_vals),
                'c': np.mean(c_vals)
            }

        # Save Calibration Data
        np.savez('RODNEY_Calibration_Values.npz',
                c_Ax=results['Ax']['c'], k_Ax=results['Ax']['k'], d_Ax=results['Ax']['d'],
                c_Bx=results['Bx']['c'], k_Bx=results['Bx']['k'], d_Bx=results['Bx']['d'],
                c_Ay=results['Ay']['c'], k_Ay=results['Ay']['k'], d_Ay=results['Ay']['d'],
                c_By=results['By']['c'], k_By=results['By']['k'], d_By=results['By']['d'])
        
        # Save to dictionary
        self.calibration_result = {
            'Ax': {'c': results['Ax']['c'], 'k': results['Ax']['k'], 'd': results['Ax']['d']},
            'Bx': {'c': results['Bx']['c'], 'k': results['Bx']['k'], 'd': results['Bx']['d']},
            'Ay': {'c': results['Ay']['c'], 'k': results['Ay']['k'], 'd': results['Ay']['d']},
            'By': {'c': results['By']['c'], 'k': results['By']['k'], 'd': results['By']['d']}
        }

    def show_raw_data(self, test_nums):
        if not hasattr(self, 'data'):
            self.load_data_from_csv_files()
        invalid_tests = [t for t in test_nums if t >= len(self.data)]
        if invalid_tests:
            print(f"Warning: Tests {invalid_tests} not found.")
            test_nums = [t for t in test_nums if t < len(self.data)]
            if not test_nums:
                return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for t in test_nums:
            df = self.data[t]
            time = df['Time (Microseconds)']
            ax1.plot(time, df['Strain Ax'], label=f'Ax Test {t}', alpha=0.5)
            ax1.plot(time, df['Strain Bx'], label=f'Bx Test {t}', alpha=0.5)
            ax2.plot(time, df['Strain Ay'], label=f'Ay Test {t}', alpha=0.5)
            ax2.plot(time, df['Strain By'], label=f'By Test {t}', alpha=0.5)
        
        ax1.set_ylabel('Strain')
        ax1.legend()
        ax1.grid(True)
        ax2.set_xlabel('Time (Microseconds)')
        ax2.set_ylabel('Strain')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    def print_calibration_constants(self):
        for bridge, params in self.calibration_result.items():
            print(f"{bridge}:")
            print(f"  c = {params['c']:.4f}")
            print(f"  k = {params['k']:.4f}")
            print(f"  d = {params['d']:.4f}")

    def calculate_and_plot_forces(self):
        if not hasattr(self, 'data') or not hasattr(self, 'calibration_result'):
            self.get_data_from_csv_files()
            self.calc_calibration_constants()
        
        for i, df in enumerate(self.data):
            time = df['Time (Microseconds)']
            cal = self.calibration_result
            
            # Calculate force for each strain using V = kF(x-d) + c
            x = self.position_vals[i]  # Example position (m), adjust as needed or use actual position
            F_Ax = (df['Strain Ax'] - cal['Ax']['c']) / (cal['Ax']['k'] * (x - cal['Ax']['d']))
            F_Bx = (df['Strain Bx'] - cal['Bx']['c']) / (cal['Bx']['k'] * (x - cal['Bx']['d']))
            F_Ay = (df['Strain Ay'] - cal['Ay']['c']) / (cal['Ay']['k'] * (x - cal['Ay']['d']))
            F_By = (df['Strain By'] - cal['By']['c']) / (cal['By']['k'] * (x - cal['By']['d']))
            
            # Plot forces
            plt.figure(figsize=(10, 6))
            plt.plot(time, F_Ax, label='Force Ax', color='blue')
            plt.plot(time, F_Bx, label='Force Bx', color='red')
            plt.plot(time, F_Ay, label='Force Ay', color='green')
            plt.plot(time, F_By, label='Force By', color='purple')
            plt.title(f'Force vs Time (File {i})')
            plt.xlabel('Time (Microseconds)')
            plt.ylabel('Force (N)')
            plt.legend()
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    myDirectory = r'Calibration/Current Calibration'
    myCalib = RODNEY_Calibration(myDirectory, defaults_flag=True)
    # myCalib.show_raw_data(test_nums=[1,2,3,4,5])
    myCalib.pre_process_data()
    myCalib.calc_calibration_constants()

    myCalib.print_calibration_constants()
    # myCalib.calculate_and_plot_forces()


