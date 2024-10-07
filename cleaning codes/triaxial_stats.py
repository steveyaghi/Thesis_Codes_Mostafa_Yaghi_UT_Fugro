# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:17:10 2024

@author: mosty
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import norm, lognorm, gamma
import seaborn as sns
from scipy.stats import linregress
from matplotlib.patches import Patch


df =pd.read_excel('triaxial.xlsx')

unique_class = df['class 1'].unique()
class_counts = df['class 1'].value_counts()


df['class 1'] = df['class 1'].replace({'sterk siltig': 'ks3',
                                       ' matig siltig': 'ks2',
                                       ' zwak siltig' : 'ks1',
                                       'matig siltig': 'ks2',
                                       ' sterk zandig': 'kz3',
                                       ' zwak zandig': 'kz1'})
unique_class2 = df['class 1'].unique()








#%%

columns_names = [
    "saturated unit weight kN/m3",
    "dry unit weight kN/m3",
    "water content (initial) %",
    "sigma 3 kPa",
    "sigma 1 kPa",    
    "friction angle",
    "q"
]


df.drop(columns=['folder name', 'page', 'class full', 'class 2'], inplace=True) 
df = df[df['saturated unit weight kN/m3'] != 48.5]

df = df.drop_duplicates()

df_selected = df


#%%

# Define the mapping of class1 values to new column values
class1_mapping = {
    'ks1': '10000000',
    'ks2': '01000000',
    'ks3': '00100000',
    'ks4': '00010000',
    'kz1': '00001000',
    'kz2': '00000100',
    'kz3': '00000010'
}
# Create the new column using the map method
df['class_encoded'] = df['class 1'].map(class1_mapping)
#%%

# Create a subplot grid
num_rows = 2  # You can adjust the number of rows and columns as needed
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))


# normal fir is black, lognormal is red and gamma is green
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
    
    # Summary statistics (modified)
    mean_val = np.mean(data)
    max_val = np.max(data)
    min_val = np.min(data)
    std_val = np.std(data)
    median_val = np.median(data)  # Calculate median
    summary_stats_str = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}\nStd: {std_val:.2f}"

    # Plot histogram
    ax.hist(data, bins=20, color='skyblue', edgecolor='black', density=True)
    ax.set_title(col_name)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Frequency')
    ax.grid(True)

    # Fit and plot distributions
    fit_and_plot_distribution(data, ax)


    # Add summary statistics text at the bottom
    ax.text(0.02, 0.02, summary_stats_str, transform=ax.transAxes, fontsize=10, va='bottom', bbox=dict(facecolor='white', alpha=0.5))

     # Hide empty subplots
    for i in range(len(columns_names), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
     
# Adjust layout
plt.tight_layout()
plt.show()
#%%

columns_names = [
    "saturated unit weight kN/m3",
    "dry unit weight kN/m3",
    "water content (initial) %",
    "sigma 3 kPa",
    "sigma 1 kPa",
    "friction angle",
    "q",
    "class 1"
]


df_selected = df[columns_names]

df_subset1 = df_selected.drop(columns=['class 1'])

#correction because of some found NaN values: 4 valus
# Replace inf and -inf with NaN in the subset
df_subset1.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with the mean of each column in the subset
df_subset1.fillna(df_subset1.mean(), inplace=True)

# Update the original DataFrame with the modified subset
df_selected.update(df_subset1)

# Create a color palette mapping unique classes to colors
unique_classes = df_selected['class 1'].unique()
color_palette = sns.color_palette("tab10", len(unique_classes)) 
class_to_color = {cls: color for cls, color in zip(unique_classes, color_palette)}

# Function to calculate R², plot with color-coded points, and add linear regression line
def annotate_r2(x, y, **kwargs):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value ** 2
    ax = plt.gca()
    ax.text(0.1, 0.9, f'$R^2={r2:.2f}$', transform=ax.transAxes, size=12, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8))
    sns.regplot(x=df_selected[x.name], y=df_selected[y.name], ax=ax, scatter=True, fit_reg=True, 
    line_kws={'color': 'red', 'linewidth': 1})

# Create a PairGrid and map the plots
pair_grid = sns.PairGrid(df_selected)
pair_grid.map_lower(annotate_r2)
pair_grid.map_diag(sns.histplot, kde=True)

# --- Calculate Correlation Matrix ---
# Select only numeric columns for correlation
numeric_columns = df_selected.select_dtypes(include=[np.number]).columns
corr_matrix = df_selected[numeric_columns].corr()
print(corr_matrix)

# --- Create the Correlation Heatmap ---
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 220, as_cmap=True)  
sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .8}, square=True)
plt.title('Correlation Matrix Heatmap')

# --- Create Legend for Scatterplot Colors ---
legend_elements = [Patch(facecolor=class_to_color[cls], label=cls) for cls in unique_classes]
plt.figure(figsize=(3, 3)) 
plt.legend(handles=legend_elements, title='Class%', loc='center')
plt.axis('off') 

# Create a separate figure for the legend
legend_elements = [Patch(facecolor=class_to_color[cls], label=cls) for cls in unique_classes]
plt.figure(figsize=(3, 3))  # Adjust figure size as needed
plt.legend(handles=legend_elements, title='Class%', loc='center')
plt.axis('off')  # Hide axes

# Adjust layout and display plots
plt.tight_layout()
plt.show()

#%%


from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


# Function to add R² value to the plots (handling NaNs)
def add_r2(ax, x, y, **kwargs):
    x = pd.to_numeric(x)
    y = pd.to_numeric(y)
    mask = ~np.isnan(x) & ~np.isnan(y) # ignore the nan values
    r, _ = pearsonr(x[mask], y[mask])
    r2 = r ** 2
    ax.annotate(f'$R^2 = {r2:.2f}$', xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=10, color='red')


# Create separate PairGrids and plots for each clay type
for cls in unique_classes:
    df_subset = df_selected[df_selected['class 1'] == cls]
    
    # Standardize the data to prevent plotting errors
    scaler = StandardScaler()
    df_subset_scaled = scaler.fit_transform(df_subset.select_dtypes(include='number'))  # Scale only numeric columns
    df_subset_scaled = pd.DataFrame(df_subset_scaled, columns=df_subset.select_dtypes(include='number').columns)  # Convert back to DataFrame
    
    pair_grid = sns.PairGrid(df_subset_scaled)
    pair_grid.map_diag(sns.histplot, kde=True)
    
    # Mapping the scatterplot with regression line and R² value
    for i, j in zip(*np.tril_indices_from(pair_grid.axes, -1)):
        x_col = df_subset_scaled.columns[j]
        y_col = df_subset_scaled.columns[i]
        sns.regplot(
            ax=pair_grid.axes[i, j],
            x=x_col,
            y=y_col,
            data=df_subset_scaled,
            scatter=True,
            fit_reg=True,
            line_kws={'color': 'red'}
        )
        add_r2(pair_grid.axes[i, j], df_subset_scaled[x_col], df_subset_scaled[y_col])
    
    plt.suptitle(f'Correlation Map for Clay Type: {cls}', y=1.02)  
    plt.show()

    # --- Calculate Correlation Matrix ---
    corr_matrix = df_subset_scaled.corr()
    print(corr_matrix)

    # --- Create the Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8}, square=True)
    plt.title(f'Correlation Matrix Heatmap for Clay Type: {cls}')
    plt.show()

    
    # --- Display Legend ---
    plt.tight_layout()
    plt.show()
    
    
#%%
columns_names = [
        "saturated unit weight kN/m3",
        "dry unit weight kN/m3",
        "water content (initial) %",
        "sigma 3 kPa",
        "sigma 1 kPa",
        "friction angle",
        "q",
        
    ]
    
# Determine number of columns dynamically based on length of columns_names
num_cols = 4
num_rows = (len(columns_names) + num_cols - 1) // num_cols  # Calculate rows based on column count

for cls in unique_classes:
    df_subset = df_selected[df_selected['class 1'] == cls]
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(f"Histograms and Distributions for Clay Type: {cls}", y=1.02)

    # Define the distribution fitting function
    def fit_and_plot_distribution(df_subset, ax):
        # Fit a normal distribution
        mu, std = norm.fit(df_subset)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label='Normal fit')

        # Fit a log-normal distribution (handle non-positive values)
        df_subset_pos = df_subset[df_subset > 0]  # Filter out non-positive values
        if len(df_subset_pos) > 0:
            shape, loc, scale = lognorm.fit(df_subset_pos, floc=0)
            p = lognorm.pdf(x, shape, loc, scale)
            ax.plot(x, p, 'r', linewidth=2, label='LogNormal fit')
        
        # Fit a gamma distribution
        alpha, loc, beta = gamma.fit(df_subset)
        p = gamma.pdf(x, alpha, loc, beta)
        ax.plot(x, p, 'g', linewidth=2, label='Gamma fit')

        ax.legend(loc='upper right')  # Move legend to upper right

    # Plot histograms, fitted distributions, and summary statistics
    for i, col_name in enumerate(columns_names):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        df_subset_col = df_subset[col_name].dropna()
        
        # Summary statistics (modified)
        mean_val = np.mean(df_subset_col)
        max_val = np.max(df_subset_col)
        min_val = np.min(df_subset_col)
        std_val = np.std(df_subset_col)
        median_val = np.median(df_subset_col)  # Calculate median
        summary_stats_str = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}\nStd: {std_val:.2f}"

        # Plot histogram
        ax.hist(df_subset_col, bins=20, color='skyblue', edgecolor='black', density=True)
        ax.set_title(col_name)
        ax.set_xlabel(col_name)
        ax.set_ylabel('Frequency')
        ax.grid(True)

        # Fit and plot distributions
        fit_and_plot_distribution(df_subset_col, ax)

        # Add summary statistics text at the bottom
        ax.text(0.02, 0.02, summary_stats_str, transform=ax.transAxes, fontsize=10, va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    # Hide empty subplots
    for i in range(len(columns_names), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()