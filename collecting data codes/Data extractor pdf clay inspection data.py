# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:33:19 2024

@author: m.yaghi
"""

import tabula
import pandas as pd
import os

# Define the directory where your PDF files are located
pdf_directory = "C:/Users/m.yaghi/OneDrive - Fugro/Desktop/python codes for different formats/pdf/format 1"

# List all PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Initialize an empty list to store the final results DataFrames
final_results_list = []

# Iterate over each PDF file
for pdf_file in pdf_files:
    # Read the PDF file and extract tables
    tables = tabula.read_pdf(os.path.join(pdf_directory, pdf_file), pages="all")
    
    # Extract project name from the first row and first column of the first table
    project_name = tables[0].iloc[0, 0]

    # Extract results from the third table (all rows except the first 6, and all columns except the first one)
    results = tables[2].iloc[4:, :4].reset_index(drop=True)
    print(f"Processing {pdf_file}: Shape of results DataFrame:", results.shape)

    # Print the project name
    print("Project Name:", project_name)

    # Function to extract the last number from a string
    def extract_last_number(text):
        if isinstance(text, str):
            return text.split(' ')[-1]
        else:
            return None

    # Iterate through each cell in the first column, extract the last number, and create a new column
    results['Last Number'] = results.iloc[:, 0].apply(extract_last_number)

    # Append the processed results to the final results list
    final_results_list.append(results.iloc[:, 1:])

# Concatenate the final results horizontally (side by side)
final_results = pd.concat(final_results_list, axis=1)

# Reset the index of the final results DataFrame
final_results.reset_index(drop=True, inplace=True)

# Print the final results DataFrame
print("\nFinal Results:")
print(final_results)
