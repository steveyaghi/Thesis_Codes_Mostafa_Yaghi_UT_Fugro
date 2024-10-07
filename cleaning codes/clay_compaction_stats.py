# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:20:33 2024

@author: m.yaghi
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import norm, lognorm, gamma
import seaborn as sns
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

df = pd.read_csv(r'C:\Users\mosty\OneDrive\Desktop\23_5_thesis files\after cleaning\clay_compactionC.csv')

df.drop(columns=['meets req min >= % 95', 'Eis Wopt <Wn <Wmax' , 'dry unit weight ', 'mass', 'weight'], inplace=True) 

df['mass'] = df['dry density by present water content'] * 0.000944

#df['dry_unit_weight'] = ( df['mass'] * 9.8 ) /( 1 + ( df['Watercontent']/100 ))

df.rename(columns={'dry density': 'dry_density_in_situ',
                   'Watercontent': 'water_content',
                   'dry density by present water content': 'dry_density_proctor',
                   'liquid limit': 'liquid_limit',
                   'degree of compaction' : 'degree_of_compaction'}, inplace=True)

df['dry_unit_weight'] = (df['dry_density_in_situ'] * 9.81) / 1000

df['unit_weight'] = df['dry_unit_weight'] * (1 + df['water_content'] / 100)



df = df[df['degree_of_compaction'] >= 97]

df['wet_density'] = df['dry_density_in_situ'] * (1 + df['water_content'] / 100)

rho_water = 1000


Gs = 2.68
df['void_ratio'] = (Gs*(1+df['water_content']/100)
                    *(rho_water/df['dry_density_in_situ']))-1

df['porosity'] = df['void_ratio']/(1+df['void_ratio'])

df['Sr'] = ((df['water_content']/100)*Gs)/df['void_ratio']

df['air_content'] = df['porosity']*(1-df['Sr'])
df['air_content%'] = df['air_content'] * 100
df['bulk_density_in_situ'] = df['dry_density_in_situ'] *(1+ df['water_content']/100)
df['bulk_density_proctor'] = df['dry_density_proctor'] *(1+ df['water_content']/100)

columns_names = [
    "dry_density_in_situ",
    "water_content",
    "dry_density_proctor",
    "degree_of_compaction",
    "dry_unit_weight",
    "unit_weight",
    "wet_density",
    "air_content",
    "air_content%",
    "bulk_density_in_situ",
    "bulk_density_proctor"
]


from scipy.optimize import fsolve
# Constants
Gs = 2.68  # Specific gravity of solids
rho_water = 1000  # Density of water in kg/m^3

# Create a range of moisture content values
moisture_content = np.linspace(5, 50, 500)

# Function to calculate bulk density for a given air content
def calc_bulk_density(w, air_content, Gs=2.68):
    # Iteratively solve for bulk density
    bulk_density = np.zeros_like(w)
    for i, w_i in enumerate(w):
        def equation(rho_b):
            e = (Gs * (1 + w_i / 100) * rho_water / rho_b) - 1
            n = e / (1 + e)
            Sr = (w_i / 100) * Gs / e
            calc_air_content = n * (1 - Sr)
            return calc_air_content - (air_content / 100)
        
        # Initial guess for bulk density
        rho_b_guess = 1500
        bulk_density[i] = fsolve(equation, rho_b_guess)[0]
        
    return bulk_density

# Plot the air content lines
air_contents = [0, 2, 4, 6, 8, 10, 12, 15, 18 , 20]
plt.figure(figsize=(10, 7))
for air_content in air_contents:
    bulk_density = calc_bulk_density(moisture_content, air_content)
    plt.plot(moisture_content, bulk_density, '--', label=f'{air_content}% Air Content')

# Plot the actual data points
plt.scatter(df['water_content'], df['bulk_density_in_situ'], c='red', label='In-situ')
plt.scatter(df['water_content'], df['bulk_density_proctor'], c='black', label='Proctor')

# Add labels and legend
plt.xlabel('Moisture Content (%)')
plt.ylabel('Bulk Density (kg/m^3)')
plt.title('Compaction Curve with Air Content Lines')
plt.legend()
plt.grid(True)
plt.show()

#%%


# Constants
rho_s = 2650  # Density of solids in kg/m^3
rho_w = 1000  # Density of water in kg/m^3

# Create a range of moisture content values
moisture_content = np.linspace(5, 50, 500)

# Function to calculate dry density for a given air content
def calc_dry_density(w, air_content):
    # Iteratively solve for dry density
    dry_density = np.zeros_like(w)
    for i, w_i in enumerate(w):
        def equation(rho_d):
            volume_soil = rho_d / rho_s
            weight_water = rho_d * (w_i / 100)
            volume_water = weight_water / rho_w
            calc_air_content = 1 - volume_soil - volume_water
            return calc_air_content - (air_content / 100)
        
        # Initial guess for dry density
        rho_d_guess = 1500
        dry_density[i] = fsolve(equation, rho_d_guess)[0]
        
    return dry_density

# Plot the air content lines
air_contents = [0, 2, 4, 6, 8, 10, 12,14, 16, 18, 20,22,24,26]
plt.figure(figsize=(10, 7))
for air_content in air_contents:
    dry_density = calc_dry_density(moisture_content, air_content)
    plt.plot(moisture_content, dry_density, '--', label=f'{air_content}% Air Content')

# Plot the actual data points
plt.scatter(df['water_content'], df['dry_density_in_situ'], c='red', label='In-situ')
plt.scatter(df['water_content'], df['dry_density_proctor'], c='black', label='Proctor')

# Add labels and legend
plt.xlabel('Moisture Content (%)')
plt.ylabel('Dry Density (kg/m^3)')
plt.title('Compaction Curve with Air Content Lines')
plt.legend()
plt.grid(True)
plt.show()



columns_names = [
    "dry_density_in_situ",
    "water_content",
    "dry_density_proctor",
    "degree_of_compaction",
    "dry_unit_weight",
    "unit_weight",
    "wet_density",

]


#%%
# Create a subplot grid
num_rows = 2  # You can adjust the number of rows and columns as needed
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))


# Define the distribution fitting function
def fit_and_plot_distribution(data, ax):
    # Fit a normal distribution
    mu, std = norm.fit(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label='Normal fit')

    # Fit a log-normal distribution
    shape, loc, scale = lognorm.fit(data, floc=0)
    p = lognorm.pdf(x, shape, loc, scale)
    ax.plot(x, p, 'r', linewidth=2, label='LogNormal fit')
    
    # Fit a gamma distribution
    alpha, loc, beta = gamma.fit(data)
    p = gamma.pdf(x, alpha, loc, beta)
    ax.plot(x, p, 'g', linewidth=2, label='Gamma fit')

    ax.legend()

# Plot histograms and fitted distributions
for i, col in enumerate(columns_names):
    row = i // num_cols
    col_name = columns_names[i]  # Correctly access the column by its name
    ax = axes[row, i % num_cols]
    data = df[col_name].dropna()
    
    # Plot histogram
    ax.hist(data, bins=20, color='skyblue', edgecolor='black', density=True)
    ax.set_title(col_name)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Frequency')
    ax.grid(True)

    # Fit and plot distributions
    fit_and_plot_distribution(data, ax)
    
    # Summary statistics
    mean_val = np.mean(data)
    median_val = np.median(data)  
    max_val = np.max(data)
    min_val = np.min(data)
    std_val = np.std(data)
    summary_stats_str = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}\nStd: {std_val:.2f}"
    

    # Add summary statistics text at the bottom
    ax.text(0.02, 0.02, summary_stats_str, transform=ax.transAxes, fontsize=10, va='bottom', bbox=dict(facecolor='white', alpha=0.5))


# Adjust layout
plt.tight_layout()
plt.show()

#%%


columns_names = [
   "dry_density_in_situ",
   "water_content",
   "dry_density_proctor",
   "degree_of_compaction",
   "dry_unit_weight",
   "unit_weight"
]
df_selected = df[columns_names]

# Function to calculate R^2 and plot it
def annotate_r2(x, y, **kwargs):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value**2
    ax = plt.gca()
    ax.text(0.1, 0.9, f'$R^2={r2:.2f}$', transform=ax.transAxes, size=12, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8))
    sns.regplot(x=df_selected[x.name], y=df_selected[y.name], ax=ax, scatter=True, fit_reg=True, 
    line_kws={'color': 'red', 'linewidth': 1})
    
# Create the PairGrid and plots
pair_grid = sns.PairGrid(df_selected)
pair_grid.map_lower(annotate_r2)
pair_grid.map_diag(sns.histplot, kde=True)

# --- Calculate Correlation Matrix ---
corr_matrix = df_selected[columns_names].corr()

# --- Create the Correlation Heatmap ---
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 220, as_cmap=True)  
sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .8}, square=True)
plt.title('Correlation Matrix Heatmap')


# --- Display Plots ---
plt.tight_layout()
plt.show()


#%%


