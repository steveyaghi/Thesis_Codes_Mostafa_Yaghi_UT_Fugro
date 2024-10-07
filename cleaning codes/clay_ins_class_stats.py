# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:52:59 2024

@author: m.yaghi
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as MplPolygon
from scipy.stats import norm, lognorm, gamma
import seaborn as sns
from scipy.stats import linregress
from matplotlib.patches import Patch

# Load the data
df = pd.read_csv(r'C:\Users\m.yaghi\OneDrive\Desktop\23_5_thesis files\after cleaning\clay_inspectionC.csv')

# Function to normalize soil content to 100%
def g(df):
    df['clay content 100%'] = (df['clay content'] / (df['sand content'] + df['silt content'] + df['clay content'])) * 100
    df['sand content 100%'] = (df['sand content'] / (df['sand content'] + df['silt content'] + df['clay content'])) * 100
    df['silt content 100%'] = (df['silt content'] / (df['sand content'] + df['silt content'] + df['clay content'])) * 100
    df['x'] = (100 - df['sand content 100%']) - df['clay content 100%'] * np.cos((1/3)*np.pi)
    df['y'] = (np.sin((1/3)*np.pi)) * df['clay content 100%']
    return df

df = g(df.copy())

# Function to classify soil based on content
def classify_soil(df):
    conditions = [
        (df['clay content 100%'] >= 50) & (df['clay content 100%'] <= 100) & (df['silt content 100%'] >= 0) & (df['silt content 100%'] <= 50) & (df['sand content 100%'] >= 0) & (df['sand content 100%'] <= 50),
        (df['clay content 100%'] >= 35) & (df['clay content 100%'] <= 50) & (df['silt content 100%'] >= 50) & (df['silt content 100%'] <= 65) & (df['sand content 100%'] >= 0) & (df['sand content 100%'] <= 67.5),
        (df['clay content 100%'] >= 25) & (df['clay content 100%'] <= 35) & (df['silt content 100%'] >= 65) & (df['silt content 100%'] <= 75) & (df['sand content 100%'] >= 0) & (df['sand content 100%'] <= 80),
        (df['clay content 100%'] >= 8) & (df['clay content 100%'] <= 25) & (df['silt content 100%'] >= 25) & (df['silt content 100%'] <= 75) & (df['sand content 100%'] >= 0) & (df['sand content 100%'] <= 50),
        
        (df['clay content 100%'] >= 17.5) & (df['clay content 100%'] <= 25) & (df['silt content 100%'] >= 25) & (df['silt content 100%'] <= 35) & (df['sand content 100%'] >= 50) & (df['sand content 100%'] <= 82.5),
        (df['clay content 100%'] >= 12) & (df['clay content 100%'] <= 17.5) & (df['silt content 100%'] >= 35) & (df['silt content 100%'] <= 40) & (df['sand content 100%'] >= 50) & (df['sand content 100%'] <= 87),
        (df['clay content 100%'] >= 8) & (df['clay content 100%'] <= 12) & (df['silt content 100%'] >= 40) & (df['silt content 100%'] <= 45) & (df['sand content 100%'] >= 50) & (df['sand content 100%'] <= 90)
            
    ]
    
    classifications = ['ks1', 'ks2', 'ks3', 'ks4', 'kz1', 'kz2', 'kz3']#, 'zk', 'zs1', 'zs2', 'zs3', 'zs4', 'lz1', 'lz2']
    df['classification'] = np.select(conditions, classifications, default='other')
    return df

df = classify_soil(df.copy())

df['void'] = ((df['liquid limit']/100) * 2.7)-0.15
df['dry unit weight'] = (2.7*9.81)/(1+df['void'])
df['water_content_0.75'] = df['liquid limit'] - 0.75*df['plasticity index']
df['water_content_0.60'] = df['liquid limit'] - 0.60*df['plasticity index']
df['water_content_0.85'] = df['liquid limit'] - 0.85*df['plasticity index']

# Define particle densities (g/cm³)
particle_densities = {
    'clay': 2.7,
    'sand': 2.65,
    'silt': 2.6
}

def calculate_dry_density(row, volume):
    """Calculates the dry density for a given row of soil component percentages."""

    total_mass_g = 0
    for component, density in particle_densities.items():
        component_percentage = row[f'{component} content 100%'] / 100
        component_volume_m3 = component_percentage * volume
        component_volume_cm3 = component_volume_m3 * 1e6  # Convert m³ to cm³
        component_mass_g = component_volume_cm3 * density
        total_mass_g += component_mass_g

    dry_density_kg_per_m3 = (total_mass_g / volume) / 1000  # Convert to kg/m³
    return dry_density_kg_per_m3

# Assuming your DataFrame has columns like 'clay content 100%', 'sand content 100%', and 'silt content 100%'
# and a column called 'Volume (m3)' with the volume of the sample in cubic meters
volume = 0.000944
df['dry_density (kg/m3)'] = df.apply(
    lambda row: calculate_dry_density(row, volume),  # Pass volume for each row
    axis=1
)







#plotting triangle
# Define the triangle vertices
triangle_vertices = [
    (0, 0),
    (50, 86.6),
    (100, 0)
]

# Define the lines data from the table as pairs of points
lines_data = [
    ((25, 43.30127019), (75, 43.30127019)),#ks1
    ((17.5, 30.31089), (82.5, 30.31089)),#ks2
    ((12.5, 21.65064), (87.5, 21.65064)),#ks3
    ((8.75, 15.15544), (41.25, 15.15544)),#kz1
    ((6, 10.3923), (44, 10.3923)),#kz2
    ((4, 6.928203), (46, 6.928203)),#kz3
    ((2.5, 4.330127), (15, 4.330127)),#zk
    ((7.5, 4.330127), (10, 0)),#zs1 (to the left)
    ((13.5, 6.928203), (17.5, 0)),#zs2 (to the left)
    ((28.5, 6.928203), (32.5, 0)),#zs3 (to the left)
    ((50, 0), (46, 6.928203)),#zs4 (to the left)
    ((46, 6.928203), (37.5, 21.65064)), #kz1,2,3, to the 
    ((46, 6.928203), (87.5, 21.65064)),#ks3  to the right and kz123 to the left
    ((85, 0), (75.29, 16.97))#lz3(to the left) en lz1 to the right
   
]

areas = {
    "Ks1": Polygon([(25, 43.30127019), (75, 43.30127019), (50, 86.6)]),
    "Ks2": Polygon([(25, 43.30127019),(17.5, 30.31089), (82.5, 30.31089), (75, 43.30127019) ]),
    "Ks3": Polygon([(17.5, 30.31089), (12.5, 21.65064), (87.5, 21.65064), (82.5, 30.31089)]),
    "Ks4": Polygon([(46, 6.928203), (37.5, 21.65064), (87.5, 21.65064)]),
    "Kz1": Polygon([(8.75, 15.15544), (41.25, 15.15544), (37.5, 21.65064),(12.5, 21.65064) ]),
    "Kz2": Polygon([(8.75, 15.15544), (6, 10.3923), (44, 10.3923), (41.25, 15.15544)]),
    "Kz3": Polygon([ (6, 10.3923), (4, 6.928203), (46, 6.928203), (44, 10.3923)]),
    "zk":  Polygon([(2.5, 4.330127), (15, 4.330127), (13.5, 6.928203), (4 , 6.928203)]),
    "Zs1": Polygon([(0, 0), (10, 0), (7.5, 4.330127), (2.5 , 4.330127)]),
    "Zs2": Polygon([(10, 0), (17.5, 0), (15, 4.330127), (7.5 , 4.330127)]),
    "Zs3": Polygon([(17.5, 0), (32.5, 0), (28.5, 6.928203), (13.5 , 6.928203)]),
    "Zs4": Polygon([(32.5 , 0), (50, 0), (46, 6.928203) , (28.5 , 6.928203)]),
    "Lz3": Polygon([(50, 0), (85, 0), (75.29, 16.97), (46 , 6.928203)]),
    "Lz1": Polygon([(85, 0), (100, 0), (87.5, 21.65064), (75.29 , 16.97)])
}



# Function to classify points based on their x, y coordinates
def classify_point(point, areas):
    for area_name, polygon in areas.items():
        if polygon.contains(point):
            return area_name
    return 'other'

# Classify each point in the dataframe
df['class%'] = df.apply(lambda row: classify_point(Point(row['x'], row['y']), areas), axis=1)
######### One-hot Encoding
# One-hot encode the 'class 1' column
df_encoded = pd.get_dummies(df['class%'])

# Concatenate the original DataFrame with the one-hot encoded DataFrame
df = pd.concat([df, df_encoded], axis=1)

# Convert one-hot columns to 0/1
for col in df_encoded.columns:
    df[col] = df[col].astype(int)

# Combine one-hot columns into a single column
df['class_1_encoded'] = df[df_encoded.columns].apply(lambda x: ''.join(x.astype(str)), axis=1)

# Drop the original one-hot encoded columns and the original class column
df.drop(df_encoded.columns, axis=1, inplace=True)




######

columns_names = [
    "clay content",
    "silt content",
    "sand content",
    "salt content",
    "organic content",
    "Mass loss",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    "classification",
    "class%",
    "void",
    "dry unit weight",
    
]



######## determining outliers
def determine_outlier_thresholds_std(dataframe, col_name):
    upper_boundary = dataframe[col_name].mean() + 3 * dataframe[col_name].std()
    lower_boundary = dataframe[col_name].mean() - 3 * dataframe[col_name].std()
    return lower_boundary, upper_boundary


for col_name in [ "clay content",
                 "silt content",
                 "sand content",
                 "salt content",
                 "organic content",
                 "Mass loss",
                 "liquid limit",
                 "plastic limit",
                 "plasticity index",
                 "clay content 100%",
                 "sand content 100%",
                 "silt content 100%",
]:
    
    lower_limit, upper_limit = determine_outlier_thresholds_std(df, col_name)
    print(f"Outlier_std thresholds for {col_name}: ({lower_limit}, {upper_limit})")


def determine_outlier_thresholds_iqr(dataframe, col_name, th1=0.05, th3=0.95):
    quartile1 = dataframe[col_name].quantile(th1)
    quartile3 = dataframe[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    upper_limit = quartile3 + 1.5 * iqr
    lower_limit = quartile1 - 1.5 * iqr
    return lower_limit, upper_limit

for col_name in [ "clay content",
                 "silt content",
                 "sand content",
                 "salt content",
                 "organic content",
                 "Mass loss",
                 "liquid limit",
                 "plastic limit",
                 "plasticity index",
                 "clay content 100%",
                 "sand content 100%",
                 "silt content 100%",
]:
    lower_limit, upper_limit = determine_outlier_thresholds_iqr(df, col_name)
    print(f"Outlier thresholds for {col_name}: ({lower_limit}, {upper_limit})")
    
    
########checking outliers

def check_outliers_iqr(dataframe, col_name):
    lower_limit, upper_limit = determine_outlier_thresholds_iqr(dataframe, col_name)
    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        return True
    else: 
        return False

def replace_with_thresholds_iqr(dataframe, cols, th1=0.25, th3=0.75, replace=False):
    from tabulate import tabulate
    
    data = []
    for col_name in cols:
        # Removed the 'Outcome' column check
        outliers_ = check_outliers_iqr(dataframe, col_name)
        count = None
        lower_limit, upper_limit = determine_outlier_thresholds_iqr(dataframe, col_name, th1, th3)
        if outliers_:
            count = dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)][col_name].count()
            if replace:
                dataframe.loc[(dataframe[col_name] > upper_limit), col_name] = upper_limit
                dataframe.loc[(dataframe[col_name] < lower_limit), col_name] = lower_limit  
        outliers_status = check_outliers_iqr(dataframe, col_name)
        data.append([outliers_, outliers_status, count, col_name, lower_limit, upper_limit])


    table = tabulate(data, headers=['Outliers (Previously)', 'Outliers', 'Count', 'Column', 'Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
    print("Removing Outliers using IQR")
    print(table)


columns_to_process = [ 
    "clay content",
    "silt content",
    "sand content",
    "salt content",
    "organic content",
    "Mass loss",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    
]

# Call the function with the list of columns
replace_with_thresholds_iqr(df, columns_to_process, replace=True)
#######
df_sized = df.drop(columns=['Unnamed: 0','class%','classification','class_1_encoded'])

#correction because of some found NaN values: 4 valus
# Replace inf and -inf with NaN in the subset
df_sized.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with the mean of each column in the subset
df_sized.fillna(df_sized.mean(), inplace=True)

# Update the original DataFrame with the modified subset
df.update(df_sized)
######### introducing the new categorization of activity
conditions_activity = [
    (df['activity'] < 0.75),
    (df['activity'] >= 0.75) & (df['activity'] < 1.25),
    (df['activity'] >= 1.25) & (df['activity'] < 2.0),
    (df['activity'] >= 2.0)
]

values_activity = ['Inactive', 'Normal', 'Active', 'Highly Active']



# Create the new column using `np.select` (corrected)
df['activity_class'] = np.select(conditions_activity, values_activity, default='Unknown')
df['activity_class'] = pd.Series(
    df['activity_class'],
    dtype="category"  
)


#########

columns_to_process = [ 
    "clay content",
    "silt content",
    "sand content",
    "salt content",
    "organic content",
    "Mass loss",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    
]



df.loc[df['activity'] == 0, 'activity'] = 0.1 # replaving 0 values of activity with 0.1


num_rows = 4  # You can adjust the number of rows and columns as needed
num_cols = 4
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))



# Plot histograms
for i, col in enumerate(columns_names):
    row = i // num_cols
    col = columns_names[i]  # Correctly access the column by its name
    axes[row, i % num_cols].hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
    axes[row, i % num_cols].set_title(col)
    axes[row, i % num_cols].set_xlabel(col)
    axes[row, i % num_cols].set_ylabel('Frequency')
    axes[row, i % num_cols].grid(True)

    # Add statistical summary
    summary = df[col].describe()
    summary_text = '\n'.join([f'{stat}: {value}' for stat, value in summary.items()])
    axes[row, i % num_cols].text(0.05, 0.95, summary_text, transform=axes[row, i % num_cols].transAxes, verticalalignment='bottom')

# Adjust layout
plt.tight_layout()

# Create the plot
plt.figure(figsize=(10, 10))

# Plot the polygons FIRST (zorder=1 to ensure they are behind)
for area_name, polygon in areas.items():
    mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True, edgecolor='blue', alpha=0.3, label=area_name, zorder=1)
    plt.gca().add_patch(mpl_poly) 

# Plot the triangle (zorder=2 to be above polygons)
triangle_x, triangle_y = zip(*triangle_vertices)
plt.plot(triangle_x + (triangle_x[0],), triangle_y + (triangle_y[0],), color='black', linewidth=2, zorder=2)  # Thicker lines

# Plot the lines within the triangle (zorder=2 as well)
for (x1, y1), (x2, y2) in lines_data:
    plt.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, zorder=5) 
  
    
# Set the color to light blue RGB
color = (0/255, 200/255, 0/255)



   # Plot points from the dataframe (selected points)
removed_scatter = plt.scatter(df['x'], df['y'], color='red', zorder=3, label='Removed Points')



# Add labels for the different spaces in the triangle
labels = {
    "Ks1": (50, 70),
    "Ks2": (50, 40),
    "Ks3": (50, 25),
    "Ks4": (50, 10),
    "Kz1": (15, 18),
    "Kz2": (15, 12),
    "Kz3": (15, 8),
    "Lz3": (60, 8),
    "Lz1": (90, 10),
    "zk" : (10, 5),
    "Zs1": (5, 2),
    "Zs2": (13, 2),
    "Zs3": (20, 2),
    "Zs4": (40, 2),
}

for label, position in labels.items():
    plt.text(position[0], position[1], label, fontsize=12, ha='center', zorder=4) 



class_counts = df['class%'].value_counts()

df = df[(df['class%'] != 'Lz1') & (df['class%'] != 'Lz3') &
        (df['class%'] != 'Zs3') & (df['class%'] != 'Zs4')]


 
    # Plot points from the dataframe(removed points)
selected_scatter = plt.scatter(df['x'], df['y'], color='green', zorder=3, label='Selected Points')  # zorder is set to ensure points are on top



# Configure plot
plt.title('Soil Classification Triangle')
plt.legend(handles=[removed_scatter, selected_scatter], title='Point Types', loc='upper left')  # Place legend in upper left (adjust 'loc' as needed)


# Show plot
plt.show(block=False)

########

columns_names = [
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    "organic content",
    "Mass loss",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "water_content_0.75",
    "water_content_0.60",
    "water_content_0.85"
]

df2 = df.copy()

df2.drop(columns=['classification', 'class%'], inplace=True) 

# Function to replace zero values with a small number
def replace_zeros_with_small_value(df2, columns, small_value=0.01):
    for column in columns:
        df2[column] = df2[column].replace(0, small_value)
    return df2

# Columns that might have zero values
columns_with_zeros = ['salt content', 'organic content']

# Replace zero values in the specified columns
df2 = replace_zeros_with_small_value(df2, columns_with_zeros)

####### ONLY HISTOGRAMS

# Create subplots, ensuring enough for all columns
num_rows = (len(columns_names) + 3) // 4  # Calculate rows based on column count
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
    ax = axes[row, i % num_cols] if num_rows > 1 else axes[i % num_cols]
    data = df2[col_name].dropna()
    
    # Summary statistics
    mean_val = np.mean(data)
    median_val = np.median(data)  
    max_val = np.max(data)
    min_val = np.min(data)
    std_val = np.std(data)
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


    
# Adjust layout
plt.tight_layout()
plt.show(block=False)


#%%
#####

columns_names = [
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    "organic content",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "class%",
    "water_content_0.75",
    "water_content_0.60",
    "water_content_0.85"
    

]


df_selected = df[columns_names]


df_subset1 = df_selected.drop(columns=['class%'])

#correction because of some found NaN values: 4 valus
# Replace inf and -inf with NaN in the subset
df_subset1.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with the mean of each column in the subset
df_subset1.fillna(df_subset1.mean(), inplace=True)

# Update the original DataFrame with the modified subset
df_selected.update(df_subset1)



#%%

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import linregress, pearsonr
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create color palettes and mappings for class% 
unique_classes = df_selected['class%'].unique()
color_palette_class = sns.color_palette("tab10", len(unique_classes))
class_to_color = {cls: color for cls, color in zip(unique_classes, color_palette_class)}

# Function to calculate R², plot with color-coded points (only class%), and add linear regression line
def annotate_r2(x, y, **kwargs):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value ** 2
    ax = plt.gca()

    # Add the linear regression equation
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    ax.text(0.1, 0.8, equation, transform=ax.transAxes, size=10, ha='left', va='center')

    # Add R^2 value
    ax.text(0.1, 0.9, f'$R^2={r2:.2f}$', transform=ax.transAxes, size=12, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8))

    # Color-coded scatterplot with only class% 
    sns.scatterplot(
        x=x,
        y=y,
        ax=ax,
        hue=df_selected['class%'],
        palette=[class_to_color[cls] for cls in unique_classes],
        s=50  # Removed style argument
    )
    
    # Linear regression line
    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter=False,
        fit_reg=True,
        line_kws={'color': 'red', 'linewidth': 1}
    )


# Create the PairGrid and plots
pair_grid = sns.PairGrid(df_selected)
pair_grid.map_lower(annotate_r2)
pair_grid.map_diag(sns.histplot, kde=True)

# --- Calculate Correlation Matrix ---
numeric_columns = df_selected.select_dtypes(include=[np.number]).columns
corr_matrix = df_selected[numeric_columns].corr()
print(corr_matrix)

# --- Create the Correlation Heatmap ---
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 220, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .8}, square=True)
plt.title('Correlation Matrix Heatmap')

# Create Legends for Scatterplot Colors (only class%)
legend_elements_class = [Patch(facecolor=class_to_color[cls], label=cls) for cls in unique_classes]

fig, ax = plt.subplots(figsize=(6, 3))
ax.legend(handles=legend_elements_class, title='Class%', loc='center', ncol=1) # Removed activity_class from legend
ax.axis('off')

plt.tight_layout()
plt.show(block=False)


# Function to add R² value to the plots (handling NaNs)
def add_r2(ax, x, y, **kwargs):
    x = pd.to_numeric(x)
    y = pd.to_numeric(y)
    mask = ~np.isnan(x) & ~np.isnan(y) # ignore the nan values
    r, _ = pearsonr(x[mask], y[mask])
    r2 = r ** 2
    ax.annotate(f'$R^2 = {r2:.2f}$', xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=10, color='red')

# Create separate PairGrids and plots for each class% only
for col_name, unique_values, color_mapping in [('class%', unique_classes, class_to_color)]: 
    for cls in unique_values:
        df_subset = df_selected[df_selected[col_name] == cls]
        
        # Standardize the data 
        scaler = StandardScaler()
        df_subset_scaled = scaler.fit_transform(df_subset.select_dtypes(include='number'))
        df_subset_scaled = pd.DataFrame(df_subset_scaled, columns=df_subset.select_dtypes(include='number').columns)
        
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
        
        plt.suptitle(f'Correlation Map for {col_name}: {cls}', y=1.02)
        plt.show(block=False)

        # --- Calculate Correlation Matrix ---
        corr_matrix = df_subset_scaled.corr()
        print(corr_matrix)

        # --- Create the Correlation Heatmap ---
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, center=0, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8}, square=True)
        plt.title(f'Correlation Matrix Heatmap for {col_name}: {cls}')
        plt.show(block=False)

#%%
###################################################################################################################
columns_names = [
    "clay content 100%",
    "sand content 100%",
    "silt content 100%",
    "organic content",
    "liquid limit",
    "plastic limit",
    "plasticity index",
    "water_content_0.75",
    "water_content_0.60",
    "water_content_0.85"
]

# Determine number of columns dynamically based on length of columns_names
num_cols = 4
num_rows = (len(columns_names) + num_cols - 1) // num_cols 

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
     
     # Summary statistics
     mean_val = np.mean(df_subset_col)
     median_val = np.median(data)  
     max_val = np.max(df_subset_col)
     min_val = np.min(df_subset_col)
     std_val = np.std(df_subset_col)
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
plt.show(block=False)

# ... (rest of the histogram code is the same as before, but with 
#      fig.suptitle updated for each column name: col_name) 
#%%

# Define the vertical lines
vertical_lines = {
    'x': [35, 50, 70],
    'y_ranges': [(0, 25), (0, 38), (0, 55)]
}

# Define the line A and line U functions
def line_a(x):
    return 0.73 * (x - 20)

def line_u(x):
    return 0.9 * (x - 8)

# Create the plot
plt.figure(figsize=(12, 10))

# Plot the vertical lines
for i, x in enumerate(vertical_lines['x']):
    y_min, y_max = vertical_lines['y_ranges'][i]
    plt.plot([x, x], [y_min, y_max], 'k-', linewidth=1.5)

# Generate x values for line A and line U
x_values = np.linspace(26.5, 100, 400)
y_values_a = line_a(x_values)

# Generate x values for line U from x = 5 to the point where y = 80
x_values_u = np.linspace(17, 80/0.9 + 8, 400)
y_values_u = line_u(x_values_u)

# Plot lines A and U
plt.plot(x_values, y_values_a, 'r-', label='A-lijn: Ip = 0.73 * (wL - 20)')
plt.plot(x_values_u, y_values_u, 'b--', label='U-lijn: Ip = 0.9 * (wL - 8)')

# Add horizontal lines
plt.plot([10, 31], [8, 8], 'k-', linewidth=1.5)
plt.plot([10, 26.5], [5, 5], 'k-', linewidth=1.5)

# Custom colors for scatter plot
custom_colors = ['red', 'lightblue', 'blue', 'green', 'yellow', 'purple', 'orange', 'gray']

# Plot the points from df_selected, colored by 'class%'
classes = df_selected['class%'].unique()

for i, class_label in enumerate(classes):
    class_data = df_selected[df_selected['class%'] == class_label]
    plt.scatter(class_data['liquid limit'], class_data['plasticity index'], 
                color=custom_colors[i], edgecolor='k', s=100, label=class_label)

# Add regions and labels for the classifications
plt.text(82, 55, 'CIV', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(82, 35, 'SIV', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(60, 40, 'CIH', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(60, 20, 'SIH', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(42.5, 25, 'CIM', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(42.5, 10, 'SIM', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(20, 7, 'CIL-SIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(30, 15, 'CIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
plt.text(30, 5, 'SIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))

# Set plot labels and title
plt.xlabel('Vloeigrens, wL')
plt.ylabel('Plasticiteitsindex, Ip')
plt.title('Plasticiteitsgrafiek')
plt.grid(True)
plt.legend()

# Set axis limits to match the example figure
plt.xlim(0, 100)
plt.ylim(0, 80)

# Show plot
plt.show(block=False)
#%%

# Define the vertical lines
vertical_lines = {
    'x': [35, 50, 70],
    'y_ranges': [(0, 25), (0, 38), (0, 55)]
}

# Define the line A and line U functions
def line_a(x):
    return 0.73 * (x - 20)

def line_u(x):
    return 0.9 * (x - 8)

# Function to create the base plot
def create_base_plot():
    plt.figure(figsize=(12, 10))

    # Plot the vertical lines
    for i, x in enumerate(vertical_lines['x']):
        y_min, y_max = vertical_lines['y_ranges'][i]
        plt.plot([x, x], [y_min, y_max], 'k-', linewidth=1.5)

    # Generate x values for line A and line U
    x_values = np.linspace(26.5, 100, 400)
    y_values_a = line_a(x_values)

    # Generate x values for line U from x = 5 to the point where y = 80
    x_values_u = np.linspace(17, 80/0.9 + 8, 400)
    y_values_u = line_u(x_values_u)

    # Plot lines A and U
    plt.plot(x_values, y_values_a, 'r-', label='A-lijn: Ip = 0.73 * (wL - 20)')
    plt.plot(x_values_u, y_values_u, 'b--', label='U-lijn: Ip = 0.9 * (wL - 8)')

    # Add horizontal lines
    plt.plot([10, 31], [8, 8], 'k-', linewidth=1.5)
    plt.plot([10, 26.5], [5, 5], 'k-', linewidth=1.5)

    # Add regions and labels for the classifications
    plt.text(82, 55, 'CIV', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(82, 35, 'SIV', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(60, 40, 'CIH', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(60, 20, 'SIH', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(42.5, 25, 'CIM', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(42.5, 10, 'SIM', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(20, 7, 'CIL-SIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(30, 15, 'CIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))
    plt.text(30, 5, 'SIL', verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=1))

    # Set plot labels and title
    plt.xlabel('Vloeigrens, wL')
    plt.ylabel('Plasticiteitsindex, Ip')
    plt.title('Plasticiteitsgrafiek')
    plt.grid(True)
    plt.legend()

    # Set axis limits to match the example figure
    plt.xlim(0, 100)
    plt.ylim(0, 80)

# Custom colors for scatter plot
custom_colors = ['red', 'blue', 'lightblue', 'green', 'yellow', 'purple', 'orange', 'black']

# Plot all points together
create_base_plot()

classes = df_selected['class%'].unique()
for i, class_label in enumerate(classes):
    class_data = df_selected[df_selected['class%'] == class_label]
    plt.scatter(class_data['liquid limit'], class_data['plasticity index'], 
                color=custom_colors[i], edgecolor='k', s=100, label=class_label)

# Show plot with all points
plt.show(block=False)

# Plot each class in a separate figure
for i, class_label in enumerate(classes):
    create_base_plot()
    class_data = df_selected[df_selected['class%'] == class_label]
    plt.scatter(class_data['liquid limit'], class_data['plasticity index'], 
                color=custom_colors[i], edgecolor='k', s=100, label=class_label)
    plt.title(f'Plasticiteitsgrafiek - {class_label}')
    plt.legend()
    plt.show()

#########################################################
