# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:16:51 2024

@author: m.yaghi
"""
#code for the Bouwstoffen files
import pandas as pd
import os

# Define the directory containing your Excel files
directory_path = r'C:\Users\m.yaghi\OneDrive - Fugro\Desktop\python codes for different formats\XLS\Bouwstoffen_XLSM'

# List of sheet names to process
sheet_names = ['lab1', 'lab2', 'lab3', 'lab4']

# Initialize an empty DataFrame to store combined data
clay_inspection_XLS_Data = pd.DataFrame()

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xls') or filename.endswith('.xlsx'): # Check for Excel files
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {filename}")

        # Loop through each sheet name
        for sheet_name in sheet_names:
            try:
                # Load the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Find the row index for the cell that contains "RESULTATEN"
                resultaten_index = df[df.apply(lambda row: row.astype(str).str.contains('RESULTATEN').any(), axis=1)].index.min()

                # Check if "RESULTATEN" was found and define the number of rows to read below it
                if pd.notna(resultaten_index):
                    rows_to_read = 30 # Change this if the number of rows varies
                    # Read the rows under "RESULTATEN"
                    data_below_resultaten = df.iloc[resultaten_index + 1 : resultaten_index + 1 + rows_to_read]

                    # Drop the first two rows from 'data_below_resultaten'
                    data_below_resultaten = data_below_resultaten.iloc[2:]

                    # Drop rows 12 and 13 (which are now rows 10 and 11 after the initial drop)
                    #data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.index[[11]].union(data_below_resultaten.index[15:]))

                    # Keep only the first 8 columns
                    data_below_resultaten = data_below_resultaten.iloc[:, :8]

                    # Drop the first, second and third columns
                    data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.columns[[0, 1, 2]], axis=1)

                    # Rename columns with filename (without extension)
                    data_below_resultaten.columns = [f"{col}_{os.path.splitext(filename)[0]}" for col in data_below_resultaten.columns]

                    # Append the processed data to the combined DataFrame
                    clay_inspection_XLS_Data = pd.concat([clay_inspection_XLS_Data, data_below_resultaten], axis=1)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except ValueError:
                # Handle the case where a sheet does not exist
                print(f"Sheet {sheet_name} not found in {filename}")
            except Exception as e:
                print(f"An error occurred while processing {sheet_name} in {filename}: {e}")

# Print a message if no data was found
if clay_inspection_XLS_Data.empty:
    print("No data found in any of the specified sheets in the provided files.")

print(clay_inspection_XLS_Data)

#the general code of the clay inspection files
# =============================================================================
# import pandas as pd
# import os
# 
# # Define the directory containing your Excel files
# directory_path = r'C:\Users\m.yaghi\OneDrive - Fugro\Desktop\python codes for different formats\XLS\Bouwstoffen_XLSM'
# 
# # List of sheet names to process
# sheet_names = ['lab1', 'lab2', 'lab3', 'lab4']
# 
# # Initialize an empty DataFrame to store combined data
# clay_inspection_XLS_Data = pd.DataFrame()
# 
# # Loop through each file in the directory
# for filename in os.listdir(directory_path):
#   if filename.endswith('.xls') or filename.endswith('.xlsx'):  # Check for Excel files
#     file_path = os.path.join(directory_path, filename)
#     print(f"Processing file: {filename}")
# 
#     # Loop through each sheet name
#     for sheet_name in sheet_names:
#       try:
#         # Load the sheet
#         df = pd.read_excel(file_path, sheet_name=sheet_name)
# 
#         # Find the row index for the cell that contains "RESULTATEN"
#         resultaten_index = df[df.apply(lambda row: row.astype(str).str.contains('RESULTATEN').any(), axis=1)].index.min()
# 
#         # Check if "RESULTATEN" was found and define the number of rows to read below it
#         if pd.notna(resultaten_index):
#           rows_to_read = 17  # Change this if the number of rows varies
#           # Read the rows under "RESULTATEN"
#           data_below_resultaten = df.iloc[resultaten_index + 1 : resultaten_index + 1 + rows_to_read]
# 
#           # Drop the first two rows from 'data_below_resultaten'
#           data_below_resultaten = data_below_resultaten.iloc[2:]
# 
#           # Drop rows 12 and 13 (which are now rows 10 and 11 after the initial drop)
#           data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.index[[11]].union(data_below_resultaten.index[15:]))
# 
#           # Keep only the first 8 columns
#           data_below_resultaten = data_below_resultaten.iloc[:, :8]
# 
#           # Drop the first, second and third columns
#           data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.columns[[0, 1, 2]], axis=1)
# 
#           # Rename columns with filename (without extension)
#           data_below_resultaten.columns = [f"{col}_{os.path.splitext(filename)[0]}" for col in data_below_resultaten.columns]
# 
#           # Append the processed data to the combined DataFrame
#           clay_inspection_XLS_Data = pd.concat([clay_inspection_XLS_Data, data_below_resultaten], axis=1)
#       except FileNotFoundError:
#         print(f"File not found: {file_path}")
#       except ValueError:
#         # Handle the case where a sheet does not exist
#         print(f"Sheet {sheet_name} not found in {filename}")
#       except Exception as e:
#         print(f"An error occurred while processing {sheet_name} in {filename}: {e}")
# 
# =============================================================================





# =============================================================================
# clay_inspection_XLS_Data.index = ['watergehaalte %(m/m)', 'gehaalte > 63 %(m/m)',
#                                   'gehaalte < 2 %(m/m)', 'gehaalte organische stof %(m/m)' ,
#                                   'Massa verlies bij HCI-beh %(m/m)' , 'geleidingsvermogen uS/cm',
#                                   'vloeigrens %(m/m)', 'Uitrolgrens %(m/m)', 'plasticiteit_index %(m/m)',
#                                   'A-lijn %(m/m)', 'zoutgehalte bodemvocht (NaCl)' , 'W_max %(m/m)',
#                                   'Consistentie_index (-)' , 'vloeibaarheidsindex (-)']
# =============================================================================



clay_inspection_XLS_Data.to_csv(r'C:\Users\m.yaghi\OneDrive - Fugro\Desktop\python codes for different formats\XLS\clay inspection.XLS_raw_without_Bouwstoffen.csv')





#save the file into a csv format
#clay_inspection_XLS_Data.to_csv(r'C:\Users\m.yaghi\OneDrive - Fugro\Desktop\python codes for different formats\XLS\clay inspection\clay_inspection_data_XLS.csv')



#code for a single file:
    
# =============================================================================
#     
# =============================================================================
# import pandas as pd
# =============================================================================
# 
# # Define the path to your Excel file using a raw string
# file_path = r'C:\Users\m.yaghi\OneDrive - Fugro\Desktop\python codes for different formats\XLS\clay inspection\test.xls'
# 
# # List of sheet names to process
# sheet_names = ['lab1', 'lab2', 'lab3', 'lab4']
# 
# # Initialize an empty DataFrame to store combined data
# combined_data = pd.DataFrame()
# 
# # Loop through each sheet name
# for sheet_name in sheet_names:
#     # Load the entire sheet initially to locate the "RESULTATEN" cell
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
# 
#     # Find the row index for the cell that contains "RESULTATEN"
#     resultaten_index = df[df.apply(lambda row: row.astype(str).str.contains('RESULTATEN').any(), axis=1)].index.min()
# 
#     # Check if "RESULTATEN" was found and define the number of rows to read below it
#     if pd.notna(resultaten_index):
#         rows_to_read = 17  # Change this if the number of rows varies
#         # Read the rows under "RESULTATEN"
#         data_below_resultaten = df.iloc[resultaten_index + 1 : resultaten_index + 1 + rows_to_read]
# 
#         # Drop the first two rows from 'data_below_resultaten'
#         data_below_resultaten = data_below_resultaten.iloc[2:]
# 
#         # Drop rows 12 and 13 (which are now rows 10 and 11 after the initial drop)
#         data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.index[[10]])
# 
#         # Keep only the first 8 columns
#         data_below_resultaten = data_below_resultaten.iloc[:, :8]
# 
#         # Drop the second and third columns
#         data_below_resultaten = data_below_resultaten.drop(data_below_resultaten.columns[[ 1, 2]], axis=1)
# 
#         # Append the processed data to the combined DataFrame
#         combined_data = pd.concat([combined_data, data_below_resultaten], axis = 1)
#     else:
#         print(f"Cell with 'RESULTATEN' not found in sheet {sheet_name}.")
# 
# print(combined_data)
# 
# =============================================================================
