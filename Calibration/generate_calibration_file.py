import numpy as np
import pandas as pd
import os
import sys

DIRECTORY = 'Calibration\Current Calibration'
    
def get_files_in_directory(directory):
  files = []
  for file in os.listdir(f'{os.getcwd()}/{directory}'):
    if file.endswith(".csv"):
      files.append(f'{directory}/{file}')
  return files

def fill_data(data, files):
  for file in files:
    df = pd.DataFrame([(value for value in ln.rstrip().split(',')) for ln in open(f'{os.getcwd()}/{file}').readlines()])
    print(df[0:20])
    file_data = df.to_numpy()
    direction = file_data[15, 1].split('"')[1].strip().upper()
    mass = float(file_data[15, 2].split('kg')[0].strip())
    distance = int(file_data[15, 3].split('cm')[0].strip())
    
    if direction == 'X':
      if mass not in data['Ax']:
        data['Ax'][mass] = {distance: []}
      elif distance not in data['Ax'][mass]:
        data['Ax'][mass][distance] = []  
      
      if mass not in data['Bx']:
        data['Bx'][mass] = {distance: []}
      elif distance not in data['Bx'][mass]:
        data['Bx'][mass][distance] = []
        
      data['Ax'][mass][distance] = file_data[38:(38+50), 1].astype(float)
      data['Bx'][mass][distance] = file_data[38:(38+50), 2].astype(float)
      
      
    
    elif direction == 'Y':
      if mass not in data['Ay']:
        data['Ay'][mass] = {distance: []}
      elif distance not in data['Ay'][mass]:
        data['Ay'][mass][distance] = []  
      
      if mass not in data['By']:
        data['By'][mass] = {distance: []}
      elif distance not in data['By'][mass]:
        data['By'][mass][distance] = []
        
      data['Ay'][mass][distance] = file_data[38:(38+50), 3].astype(float)
      data['By'][mass][distance] = file_data[38:(38+50), 4].astype(float)
      
    #else:
    #  print(f'Error: {file} has an invalid direction: {direction}')
    #  sys.exit(1)

def write_to_file(file_name, data):
  header = ['Location', 'Mass (kg)', 'Distance (cm)']	
  for sample_id in range(1, 51):
    header.append(f'Sample {sample_id}')
  header += ['Sample Average', 'Standard Deviation']
  
  array_data = np.array([header]).astype('object')
  for direction in data:
    for mass in sorted(list(data[direction].keys()), reverse=False):
      for distance in sorted(list(data[direction][mass].keys()), reverse=False):
        samples = data[direction][mass][distance]
        sample_avg = np.mean(samples)
        sample_std = np.std(samples)
        row = [direction, mass, distance]
        row += samples.tolist()
        row += [sample_avg, sample_std]
        array_data = np.append(array_data, [row], axis=0)
  
  df = pd.DataFrame(array_data)     
  def convert_to_float_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value

  df = df.applymap(convert_to_float_or_str)
  df.to_excel(f'{os.getcwd()}/{file_name}.xlsx', index=False, header=False, sheet_name='Data')
  
def main():
  data = {
    'Bx': {},
    'By': {},
    'Ax': {},
    'Ay': {},
  }
  files = get_files_in_directory(DIRECTORY)
  fill_data(data, files)
  write_to_file(f'{DIRECTORY}_combined', data)
    
if __name__ == "__main__":
  main()
  print('Done generating calibration file')