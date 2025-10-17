import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv

# Need to put Darliing results from 8_07 and 8_22
# Need to put Hi-STIFS results from 8_07 and 8_22

def collect_results():
    all_results_path = r'Results/Field/all_results.csv'
    if not os.path.exists(all_results_path):
        headers = ['Date', 'Method', 'Stalk Interactions', 'Variety', 'Section', 'Stalk Number', 'Mean', 'Median', 'Std_Dev', 'Coeff. Var.']
        df = pd.DataFrame(columns=headers)
        df.to_csv(all_results_path, index=False)

    varieties = ['Ornamental', 'Vigor Root', 'Popcorn', 'Xtra Early 320']
    # sections = ['11-B WE', '12-C WE', '13-B WE', '15-A WE']
    # sections = ['6-A Iso', '7-A Iso', '7-B Iso', '8-C Iso', '10-A Iso']
    sections = ['7-B', '8-C', '10-A']
    date = '08_22'
    method = 'DARLING'
    interaction = 'Interaction'
    variety = varieties[3]
    section = sections[2]

    if method == 'DARLING':
        filepath = rf'Results/Field/{date}/{section}/darling_{date}_{section}.csv'
        df = pd.read_csv(filepath)
        print(df); print('\n')

        with open(all_results_path, 'a' if os.path.exists(all_results_path) else 'w', newline='') as f:
            writer = csv.writer(f)
            for row in df.iloc:
                new_row = [date, method, interaction, variety, section, row['Stalk'], row['Mean'], row['Median'], 
                           row['Std_Dev'], round(row['Std_Dev']/row['Median'], 4)]
                writer.writerow(new_row)
          
    method = 'Hi-STIFS'
    if method == 'Hi-STIFS':
        filepath = rf'Results/Field/{date}/{section}/stiffness_{date}_{section}.csv'
        df = pd.read_csv(filepath)
        print(df); print('\n')

        with open(all_results_path, 'a' if os.path.exists(all_results_path) else 'w', newline='') as f:
            writer = csv.writer(f)
            for row in df.iloc:
                new_row = [date, method, interaction, variety, section, row['Stalk'], round(row['Mean'], 4), 
                           round(row['Median'], 4), round(row['Std_Dev'], 4) , round(row['Std_Dev']/row['Median'], 4)]
                writer.writerow(new_row)

def work_CV():
    all_results_path = r'Results/Field/all_results.csv'

    if not os.path.exists(all_results_path):
        raise ValueError(f'No results file at {all_results_path}')
    
    df = pd.read_csv(all_results_path)

    DARLING_df = df[df['Method'] == 'DARLING']
    D_CV = DARLING_df['Coeff. Var.'].to_numpy()
    D_Median = DARLING_df['Median'].to_numpy()
    
    HiSTIFS_df = df[df['Method'] == 'Hi-STIFS']
    H_CV = HiSTIFS_df['Coeff. Var.'].dropna().to_numpy()
    H_Median = HiSTIFS_df.loc[HiSTIFS_df['Coeff. Var.'].notna(), 'Median'].to_numpy()

    plt.figure()
    plt.hist(D_CV, bins=50, label='DARLING')
    plt.axvline(np.median(D_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'DARLING Median CV: {np.median(D_CV):.3f}')
    # plt.legend()

    plt.figure()
    plt.hist(H_CV, bins=50, label='Hi-STIFS')
    plt.axvline(np.median(H_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'Hi-STIFS Median CV: {np.median(H_CV):.3f}')

    t_stat, p_value = stats.ttest_ind(H_CV, D_CV, nan_policy='omit')
    plt.figure()
    plt.boxplot([D_CV, H_CV], positions=[1, 2], tick_labels=['DARLING, n=5', 'Hi-STIFS, n=10'], notch=True)
    plt.title(f'Diff of Two Means p-value: {p_value:.2e}', fontsize=16)
    plt.ylabel('Single Stalk CV', fontsize=14)
    print(np.median(D_CV), np.median(H_CV))

    # # Create joint histogram for DARLING
    # plt.figure()
    # plt.hist2d(D_Median, D_CV, bins=50, cmap='Blues')
    # plt.colorbar(label='Count')
    # plt.xlabel('Median')
    # plt.ylabel('Coefficient of Variation')
    # plt.title('DARLING Joint Histogram')
    
    # # Create joint histogram for Hi-STIFS
    # plt.figure()
    # plt.hist2d(H_Median, H_CV, bins=50, cmap='Blues')
    # plt.colorbar(label='Count')
    # plt.xlabel('Median')
    # plt.ylabel('Coefficient of Variation')
    # plt.title('Hi-STIFS Joint Histogram')


    D_ISO_df = DARLING_df[DARLING_df['Stalk Interactions'] == 'Isolated']
    D_INT_df = DARLING_df[DARLING_df['Stalk Interactions'] == 'Interaction']
    H_ISO_df = HiSTIFS_df[HiSTIFS_df['Stalk Interactions'] == 'Isolated']
    H_INT_df = HiSTIFS_df[HiSTIFS_df['Stalk Interactions'] == 'Interaction']

    D_ISO_CV = D_ISO_df['Coeff. Var.'].to_numpy()
    D_INT_CV = D_INT_df['Coeff. Var.'].to_numpy()
    H_ISO_CV = H_ISO_df['Coeff. Var.'].dropna().to_numpy()
    H_INT_CV = H_INT_df['Coeff. Var.'].dropna().to_numpy()

    plt.figure()
    plt.hist(D_ISO_CV, bins=30, label='DARLING')
    plt.axvline(np.median(D_ISO_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'DARLING ISO Median CV: {np.median(D_ISO_CV):.3f}')

    plt.figure()
    plt.hist(D_INT_CV, bins=30, label='DARLING')
    plt.axvline(np.median(D_INT_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'DARLING INT Median CV: {np.median(D_INT_CV):.3f}')

    plt.figure()
    plt.hist(H_ISO_CV, bins=30, label='Hi-STIFS')
    plt.axvline(np.median(H_ISO_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'Hi-STIFS ISO Median CV: {np.median(H_ISO_CV):.3f}')

    plt.figure()
    plt.hist(H_INT_CV, bins=30, label='Hi-STIFS')
    plt.axvline(np.median(H_INT_CV), c='red')
    plt.xlabel('Stalk Coefficient of Variation')
    plt.ylabel('Frequency')
    plt.title(f'Hi-STIFS INT Median CV: {np.median(H_INT_CV):.3f}')



if __name__ == "__main__":
    # collect_results()
    work_CV()


    plt.show()