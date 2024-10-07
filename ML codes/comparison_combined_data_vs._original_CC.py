# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:09:46 2024

@author: mosty
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

com = pd.read_excel(r'C:\Users\m.yaghi\OneDrive\Desktop\23_5_thesis files\Machine learning phase\CompA.xlsx')


def clip_and_replace(df, density_columns, unit_weight_columns, df_name, density_min=1094, density_max=1919, weight_min=10.90, weight_max=18.73):
    """
    Clips and replaces values in specified columns of a DataFrame, counts the replaced values,
    and prints the results along with the name of the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        density_columns (list): List of columns for dry density to clip and replace.
        unit_weight_columns (list): List of columns for dry unit weight to clip and replace.
        df_name (str): The name of the DataFrame (for printing purposes).
        density_min (float): Minimum value to clip dry density columns.
        density_max (float): Maximum value to replace in dry density columns.
        weight_min (float): Minimum value to clip dry unit weight columns.
        weight_max (float): Maximum value to replace in dry unit weight columns.
    
    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Counters for replaced values
    replaced_count = {col: 0 for col in density_columns + unit_weight_columns}

    # Clip and replace for dry density columns
    df[density_columns] = df[density_columns].clip(lower=density_min)
    for col in density_columns:
        # Count how many values are greater than the max
        replaced_count[col] += (df[col] > density_max).sum()
        # Replace values greater than the max
        df[col] = df[col].apply(lambda x: density_max if x > density_max else x)
    
    # Clip and replace for dry unit weight columns
    df[unit_weight_columns] = df[unit_weight_columns].clip(lower=weight_min)
    for col in unit_weight_columns:
        # Count how many values are greater than the max
        replaced_count[col] += (df[col] > weight_max).sum()
        # Replace values greater than the max
        df[col] = df[col].apply(lambda x: weight_max if x > weight_max else x)

    # Print the counts of replaced values for each column, including DataFrame name
    print(f"Results for DataFrame '{df_name}':")
    for col, count in replaced_count.items():
        print(f"Column '{col}': {count} values replaced")

    return df

dry_density_to_clip = ['predicted_dry_density_proctor_0.60_nn',
                     'predicted_dry_density_proctor_0.75_nn','predicted_dry_density_proctor_0.85_nn',
                     'predicted_dry_density_proctor_0.60_RF',
                                         'predicted_dry_density_proctor_0.75_RF','predicted_dry_density_proctor_0.85_RF',]
 
dry_unitweight_to_clip = ['predicted_dry_unit_weight_0.60_nn',
                     'predicted_dry_unit_weight_0.75_nn', 'predicted_dry_unit_weight_0.85_nn', 'predicted_dry_unit_weight_0.60_RF',
                                         'predicted_dry_unit_weight_0.75_RF','predicted_dry_unit_weight_0.85_RF']

# Applying the function to the com DataFrame
com = clip_and_replace(com, dry_density_to_clip, dry_unitweight_to_clip, 'com')

#%%

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_unit_weight_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_unit_weight_0.60_RF'], com['water_content_0.60_RF'], c='green', s=50, alpha=0.8, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_unit_weight_0.60_nn'], com['water_content_0.60_nn'], c='blue', s=50, alpha=0.5, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('dry unit weight')
plt.ylabel('water content')
plt.title(' (AutoEncoded) Dry unit weight vs. Water content 0.60')
plt.legend()

plt.show()  

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_unit_weight_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_unit_weight_0.75_RF'], com['water_content_0.75_RF'], c='green', s=50, alpha=0.8, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_unit_weight_0.75_nn'], com['water_content_0.75_nn'], c='blue', s=50, alpha=0.8, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('dry unit weight')
plt.ylabel('water content')
plt.title(' (AutoEncoded) Dry unit weight vs. Water content 0.75')
plt.legend()

plt.show()  

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_unit_weight_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_unit_weight_0.85_RF'], com['water_content_0.85_RF'], c='green', s=50, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_unit_weight_0.85_nn'], com['water_content_0.85_nn'], c='blue', s=50, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('dry unit weight')
plt.ylabel('water content')
plt.title(' (AutoEncoded)  unit weight vs. Water content 0.85')
plt.legend()

plt.show()  

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_density_proctor_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_density_proctor_0.60_RF'], com['water_content_0.60_RF'], c='green', s=50, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_density_proctor_0.60_nn'], com['water_content_0.60_nn'], c='blue', s=50, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('Dry Density proctor')
plt.ylabel('water content')
plt.title('(AutoEncoded) Dry Density proctor vs. Water content 0.60')
plt.legend()

plt.show()  

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_density_proctor_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_density_proctor_0.75_RF'], com['water_content_0.75_RF'], c='green', s=50, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_density_proctor_0.75_nn'], com['water_content_0.75_nn'], c='blue', s=50, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('Dry Density proctor')
plt.ylabel('water content')
plt.title(' (AutoEncoded) Dry Density proctor vs. Water content 0.75')
plt.legend()


plt.show()  

plt.figure(figsize=(8, 6))  # Optional: Adjust plot size
plt.scatter(com['dry_density_proctor_cc'], com['water_content_cc'], c='black', s=50, label= 'original data from compaction test')  # c: Colors, s: Marker size'
plt.scatter(com['predicted_dry_density_proctor_0.85_RF'], com['water_content_0.85_RF'], c='green', s=50, label= ' Random Forest predicted values')
plt.scatter(com['predicted_dry_density_proctor_0.85_nn'], com['water_content_0.85_nn'], c='blue', s=50, label= ' Neural Network predicted values')
#plt.scatter(dd['dry_unit_weight_CI'], dd['water_content_RF'], c='red', s=50, label= 'Random Forest predicted values')

#plt.scatter(com['dry_unit_weight_CI_NN'], com['linear_predicted_water_content'], c='green', s=50, label= 'linear predicted values')
#plt.scatter(com['dry_unit_weight_CI'], com['exponential_predicted_water_content'], c='yellow', s=50, label= 'exponential predicted values')
# Add Labels and Title
plt.xlabel('Dry Density proctor')
plt.ylabel('water content')
plt.title(' (AutoEncoded) Dry Density proctor vs. Water content 0.85')
plt.legend()

plt.show()  

