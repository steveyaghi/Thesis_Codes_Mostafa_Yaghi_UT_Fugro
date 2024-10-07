# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:33:19 2024

@author: m.yaghi
"""

# This code include all the columns. This is because the package does not seem to recognize the empty
#columns. Therefore, the approach is to extract the data and then remove the unwanted one in the cleaning phase.

import os
import camelot
import pandas as pd

# Function to extract Table 3 from PDF files in a folder
def extract_table3_from_folder(folder_path):
    dfs = []  # List to store DataFrames of Table 3 from each PDF
    total_files = 0  # To count the total number of PDF files processed
    extracted_files = 0  # To count files from which Table 3 was successfully extracted
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            total_files += 1
            filepath = os.path.join(folder_path, filename)
            # Using Camelot to read tables with 'lattice' method which works well for tables with clear grid lines
            tables = camelot.read_pdf(filepath, pages='all', flavor='lattice')
            
            # Check if at least 3 tables are detected, then extract the third table (index 2)
            if tables.n >= 3:
                table3 = tables[2].df.iloc[2:9, 1:10].reset_index(drop=True)
                dfs.append(table3)
                extracted_files += 1
    
    # Calculating the percentage of files that had Table 3 extracted
    extraction_percentage = (extracted_files / total_files) * 100 if total_files > 0 else 0
    print(f"Extraction success rate: {extraction_percentage:.2f}%")
    
    return dfs

# Folder containing the PDF files
folder_path = "C:/Users/m.yaghi/OneDrive - Fugro/Desktop/python codes for different formats/pdf/format 3 compaction test/test"

# Extract Table 3 from PDFs in the folder
table3_dfs = extract_table3_from_folder(folder_path)

# Combine all Table 3 DataFrames horizontally
if table3_dfs:
    combined_table3_df = pd.concat(table3_dfs, axis=1)
else:
    combined_table3_df = pd.DataFrame()  # Empty DataFrame if no tables were extracted

print("Shape of combined Table 3 DataFrame:", combined_table3_df.shape)



# =============================================================================
# import tabula
# import pandas as pd
# 
# 
# 
# 
# # Read the PDF file and extract tables
# tables = tabula.read_pdf("test.pdf", pages="all")
# 
# 
# results = tables[2].iloc[2:9, 1:10].reset_index(drop=True)
# print("Shape of results DataFrame:", results.shape)
# 
# 
# 
# results[['number1', 'number2']] = results['Unnamed: 0'].str.split(' ', expand=True)
# 
# results.drop(columns=['Unnamed: 0'], inplace =True)
# 
# results[['number3', 'number4']] = results['Unnamed: 3'].str.split(' ', expand=True)
# 
# results.drop(columns=['Unnamed: 3'], inplace =True)
# 
# =============================================================================
# =============================================================================
# import tabula
# import pandas as pd
# import os
# 
# # Define the directory where your PDF files are located
# pdf_directory = "C:/Users/m.yaghi/OneDrive - Fugro/Desktop/python codes for different formats/pdf/format 3"
# 
# # List all PDF files in the directory
# pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
# 
# # Initialize an empty list to store the final results DataFrames
# final_results_list = []
# 
# # Iterate over each PDF file
# for pdf_file in pdf_files:
#     # Read the PDF file and extract tables
#     tables = tabula.read_pdf(os.path.join(pdf_directory, pdf_file), pages="all")
#     
# # =============================================================================
# #     # Extract project name from the first row and first column of the first table
# #     project_name = tables[0].iloc[0, 0]
# # =============================================================================
# 
#     # Extract results from the third table (all rows except the first 6, and all columns except the first one)
#     results = tables[2].iloc[3:, :4].reset_index(drop=True)
#     print(f"Processing {pdf_file}: Shape of results DataFrame:", results.shape)
# 
# # =============================================================================
# #     # Print the project name
# #     print("Project Name:", project_name)
# # 
# # =============================================================================
#     # Function to extract the last number from a string
#     def extract_last_number(text):
#         if isinstance(text, str):
#             return text.split(' ')[-1]
#         else:
#             return None
# 
#     # Iterate through each cell in the first column, extract the last number, and create a new column
#     results['Last Number'] = results.iloc[:, 0].apply(extract_last_number)
# 
#     # Append the processed results to the final results list
#     final_results_list.append(results.iloc[:, 1:])
# 
# # Concatenate the final results horizontally (side by side)
# final_results = pd.concat(final_results_list, axis=1)
# 
# # Reset the index of the final results DataFrame
# final_results.reset_index(drop=True, inplace=True)
# 
# # Print the final results DataFrame
# print("\nFinal Results:")
# print(final_results)
# 
# =============================================================================


