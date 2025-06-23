import pandas as pd

# Replace 'file_path.pkl' with your .pkl file path
file_path = r'ProcessedFiles\06_11_test_3.csv.pkl'
data = pd.read_pickle(file_path)

# Display the data
print(list(data.columns))
print(data['Strain Ax'])
print(data)