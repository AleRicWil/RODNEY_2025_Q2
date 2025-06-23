import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
from tkinter import Tk, filedialog, messagebox
import datetime

# Global variables for click events
clicks = []
fig = None
ax = None

def on_click(event):
    if event.inaxes == ax and event.button == 1:  # Left mouse click within axes
        clicks.append(event.xdata)
        ax.axvline(event.xdata, color='r', linewidth=2)
        plt.draw()

def process_file(filename, folder, operator, clicker, session_filename):
    global clicks, fig, ax

    # Step 1: Read Data
    raw_data = pd.read_csv(os.path.join(folder, filename), skiprows=11)
    print(raw_data)
    raw_data = raw_data.drop(columns=['Current Time'])
    raw_data = raw_data.rename(columns={'Time_Microseconds': 'Time'})
    raw_data = raw_data.astype({'Time': 'float32', 'Strain Ax': 'float32', 'Strain Bx': 'float32',
                                'Strain Ay': 'float32', 'Strain By': 'float32'})
    
    # Read header metadata
    header_data = pd.read_csv(os.path.join(folder, filename), header=None, nrows=10).fillna('').values

    # Step 2: Prepare Data for Visualization
    nbegin = 100
    c1 = raw_data.iloc[:nbegin, 1:5].mean().values
    s1 = raw_data.iloc[:nbegin, 1:5].std().values
    dat = raw_data.iloc[:, 1:5].values
    index = dat < c1 + 4 * s1
    dat[index] = np.nan
    c2 = np.nanmean(dat, axis=0)
    s2 = np.nanstd(dat, axis=0)
    dat = raw_data.iloc[:, 1:5].values
    dat = (dat - c1) / (c2 + 3 * s2)
    dat = dat + np.array([0, 1, 2, 3])

    # Step 3: Click on Cut Points
    nstalks = 9
    while True:
        clicks = []
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dat)
        ax.text(500, -0.25, 'Click twice to trim the ends of the data.', fontsize=12)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        print(len(clicks))
        if len(clicks) == 2:
            dstart, dend = map(int, clicks)
            ax.axvline(dstart, color='b')
            ax.axvline(dend, color='b')
            plt.draw()
            if messagebox.askyesno("Confirm", "Continue with these cuts?"):
                plt.close()
                break
            plt.close()

    while True:
        clicks = []
        fig, ax = plt.subplots(figsize=(10, 6))
        index = np.arange(dstart, dend + 1)
        ax.plot(index, dat[index])
        ax.set_ylim(-0.5, ax.get_ylim()[1])
        ax.text(500, -0.25, 'Click 8 times to cut between stalks.', fontsize=12)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        if len(clicks) == nstalks - 1:
            cuts = [dstart] + list(map(int, clicks)) + [dend]
            for cut in cuts:
                ax.axvline(cut, linewidth=3)
            plt.draw()
            if messagebox.askyesno("Confirm", "Continue with these cuts?"):
                plt.close()
                break
            plt.close()

    # Step 4: Labeling
    dat = dat - np.array([0, 1, 2, 3])
    cuts_matrix = np.full((nstalks, 7), np.nan)
    cuts_matrix[:, 0] = cuts[:-1]
    cuts_matrix[:, 6] = cuts[1:]

    for i in range(nstalks):
        while True:
            clicks = []
            fig, ax = plt.subplots(figsize=(10, 6))
            index = np.arange(int(cuts_matrix[i, 0]), int(cuts_matrix[i, 6]) + 1)
            ax.plot(index, dat[index])
            ax.text(cuts_matrix[i, 0] + 10, 1.5, 'Click 5 times to label data.', fontsize=12)
            ax.set_title(f'Stalk {i + 1}')
            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            if len(clicks) == 5:
                cuts_matrix[i, 1:6] = list(map(int, clicks))
                ax.plot(cuts_matrix[i], dat[cuts_matrix[i].astype(int)], 'ro', markersize=8)
                plt.draw()
                if messagebox.askyesno("Confirm", "Continue with these labels?"):
                    plt.close()
                    break
                plt.close()

    # Step 5: Convert to Table
    columns = ['Note', 'RodneyConfig', 'PVC', 'Height', 'Yaw', 'Pitch', 'Roll', 'RateOfTravel',
               'AngleOfTravel', 'Offset', 'Stalk', 'Time', 'Strain Ax', 'Strain Bx', 'Strain Ay',
               'Strain By', 'DataStartOG', 'P1', 'P2', 'P3', 'P4', 'P5', 'DataEndOG', 'Operator',
               'Clicker', 'Filename', 'Directory']
    single_test_run = pd.DataFrame(columns=columns)
    
    for i in range(nstalks):
        index = np.arange(int(cuts_matrix[i, 0]), int(cuts_matrix[i, 6]) + 1)
        row = {
            'Stalk': i + 1,
            'Time': [raw_data['Time'].iloc[index].values],
            'Strain Ax': [raw_data['Strain Ax'].iloc[index].values],
            'Strain Bx': [raw_data['Strain Bx'].iloc[index].values],
            'Strain Ay': [raw_data['Strain Ay'].iloc[index].values],
            'Strain By': [raw_data['Strain By'].iloc[index].values],
            'DataStartOG': cuts_matrix[i, 0],
            'P1': cuts_matrix[i, 1] - cuts_matrix[i, 0],
            'P2': cuts_matrix[i, 2] - cuts_matrix[i, 0],
            'P3': cuts_matrix[i, 3] - cuts_matrix[i, 0],
            'P4': cuts_matrix[i, 4] - cuts_matrix[i, 0],
            'P5': cuts_matrix[i, 5] - cuts_matrix[i, 0],
            'DataEndOG': cuts_matrix[i, 6]
        }
        single_test_run = pd.concat([single_test_run, pd.DataFrame([row])], ignore_index=True)

    # Step 6: Add Metadata
    single_test_run['Note'] = header_data[0, 1]
    single_test_run['RodneyConfig'] = header_data[1, 1]
    single_test_run['PVC'] = header_data[2, 1]
    single_test_run['Height'] = float(header_data[3, 1])
    single_test_run['Yaw'] = float(header_data[4, 1])
    single_test_run['Pitch'] = float(header_data[5, 1])
    single_test_run['Roll'] = float(header_data[6, 1])
    single_test_run['RateOfTravel'] = float(header_data[7, 1])
    single_test_run['AngleOfTravel'] = float(header_data[8, 1])
    single_test_run['Offset'] = float(header_data[9, 1])
    single_test_run['Operator'] = operator
    single_test_run['Clicker'] = clicker
    single_test_run['Filename'] = filename
    single_test_run['Directory'] = folder

    return single_test_run

def main():
    # Step 0: User Input and Setup
    root = Tk()
    root.withdraw()  # Hide the main window

    # Choose processing option
    option = messagebox.askyesno("Processing Options", "Process a single file? (Yes = Single File, No = Folder)")
    if option:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        folder = os.path.dirname(file_path)
        files = [os.path.basename(file_path)]
    else:
        folder = filedialog.askdirectory()
        files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    if not files:
        return

    # Operator and Clicker
    operator = messagebox.askquestion("Test Operator", "Is the operator CS? (Yes = CS, No = MH)") == 'yes' and 'CS' or 'MH'
    clicker = messagebox.askquestion("Test Processor", "Is the processor CS? (Yes = CS, No = MH)") == 'yes' and 'CS' or 'MH'

    # Session Filename
    session_filename = filedialog.asksaveasfilename(defaultextension=".pkl", initialfile=files[0])
    if not session_filename:
        return

    # Create output directory
    os.makedirs(os.path.join(folder, 'ProcessedFiles'), exist_ok=True)

    # Initialize MultiTestRun
    multi_test_run = pd.DataFrame()

    # Loop Through Files
    for filename in files:
        single_test_run = process_file(filename, folder, operator, clicker, session_filename)
        
        # Step 7: Save and Consolidate
        run_save_file = os.path.join(folder, 'ProcessedFiles', filename + '.pkl')
        single_test_run.to_pickle(run_save_file)
        os.rename(os.path.join(folder, filename), os.path.join(folder, 'ProcessedFiles', filename))
        
        multi_test_run = pd.concat([multi_test_run, single_test_run], ignore_index=True)
        multi_test_run.to_pickle(session_filename)

        if not messagebox.askyesno("Continue", "Continue processing?"):
            break

    # Append to Existing Table
    if messagebox.askyesno("Append Data", "Append to an existing data table?"):
        append_file = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if append_file:
            pvc_tests = pd.read_pickle(append_file)
            n_stalks_prev = len(pvc_tests)
            pvc_tests = pd.concat([pvc_tests, multi_test_run], ignore_index=True)
            n_stalks_new = len(pvc_tests)
            n_increased = n_stalks_new - n_stalks_prev

            confirmation_message = f"Stalks processed: {len(multi_test_run)}\nMaster Table increased by: {n_increased}\nOk to save results?"
            if messagebox.askyesno("Append Confirmation", confirmation_message):
                save_append_filename = f"PVC_MASTER_TABLE_{datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.pkl"
                pvc_tests.to_pickle(save_append_filename)

if __name__ == "__main__":
    main()