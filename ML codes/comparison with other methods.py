import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
path = r'C:\Users\m.yaghi\OneDrive\Desktop\23_5_thesis files\Machine learning phase'
COMARF = pd.read_excel(path + r'\COMARF.xlsx')
COMRF = pd.read_excel(path + r'\COMRF.xlsx')
COMARF_s1 = pd.read_excel(path + r'\COMARF_scenario1.xlsx')
COMARF_s2 = pd.read_excel(path + r'\COMARF_scenario2.xlsx')

#%%
# Specify the columns of interest
column_names = [
    "clay_content_100%",
    "sand_content_100%",
    "silt_content_100%",
    "liquid_limit",
    "plastic_limit",
    "plasticity_index",
    "water_content_0.85",
    "predicted_dry_density_proctor_0.85",
    "predicted_dry_unit_weight_0.85"
]

# Subset the dataframe to include only the specified columns
COMARF_subset = COMARF[column_names]

# Compute the correlation matrix
correlation_matrix = COMARF_subset.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap=sns.diverging_palette(220, 220, as_cmap=True), vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Function to compute linear equations and R^2 values
def compute_linear_regression(dataframe):
    columns = dataframe.columns
    n = len(columns)
    equations = []
    
    for i in range(n):
        for j in range(i + 1, n):
            x = dataframe[columns[i]].values.reshape(-1, 1)  # Ensure x is 2D
            y = dataframe[columns[j]].values
            
            # Fit the model
            model = LinearRegression().fit(x, y)
            predictions = model.predict(x)
            
            # Calculate R^2
            r2 = r2_score(y, predictions)
            # Handle model.coef_ which is a 1D array
            coef = model.coef_[0] if len(model.coef_) == 1 else model.coef_[0][0]
            equations.append(f'{columns[j]} = {model.intercept_:.2f} + {coef:.2f} * {columns[i]} (R² = {r2:.2f})')
    
    return equations

# Compute and print linear equations with R^2 values
equations = compute_linear_regression(COMARF_subset)
for eq in equations:
    print(eq)

#%%

# Load the data
path = r'C:\Users\mosty\OneDrive\Desktop\23_5_thesis files\Machine learning phase'
COMARF = pd.read_excel(path + r'\COMARF.xlsx')
COMRF = pd.read_excel(path + r'\COMRF.xlsx')

# Extract the necessary columns
liquid_limit_COMARF = COMARF['liquid_limit']
clay_content_COMARF = COMARF['clay_content_100%']

liquid_limit_COMRF = COMRF['liquid_limit']
clay_content_COMRF = COMRF['clay_content_100%']

liquid_limit_s1 = COMARF_s1['liquid_limit']
predicted_clay_content_s1 = COMARF_s1['Predicted_clay_content_100%']

# Create a scatter plot
plt.figure(figsize=(10, 6))

#plt.scatter(clay_content_COMARF, liquid_limit_COMARF, color='blue', label='COMARF Data', alpha=0.6)
plt.scatter(clay_content_COMRF, liquid_limit_COMRF, color='blue', label='COMRF Data', alpha=0.6)

# Plot the linear correlation line
clay_content_range = np.linspace(8, 75, 300)  # Synthetic intervals for clay_content
liquid_limit_predicted = 13.21 + 1.17 * clay_content_range
plt.plot(clay_content_range, liquid_limit_predicted, color='red', label='Linear Correlation LL = 13.21 + 1.17 * CC, R2 = 0.77')

# Add the second equation based on the rearranged formula for liquid_limit
plasticity_index_range = np.linspace(15, 100, 300)  # Synthetic intervals for plasticity index
liquid_limit_from_second_eq = (0.26 * clay_content_range + 10 + plasticity_index_range) / 0.69
plt.plot(clay_content_range, liquid_limit_from_second_eq, color='purple', label='Polidorli(2007), LL = (0.26 * CC + 10 + PI) / 0.69')

# Values for k1
k1_values = [0.67, 2.4]

# Plot the lines for each k1 value
for k1 in k1_values:
    liquid_limit_k1 = k1 * clay_content_range
    plt.plot(clay_content_range, liquid_limit_k1, color='blue', label=f'Polidorli(2007) LL = {k1} * Clay Content')
    
plt.scatter(predicted_clay_content_s1, liquid_limit_s1, color='green', label='Predicted Clay cotent from Scenario 1 Random Forest', alpha=0.6)


# Labeling the axes
plt.xlabel('Clay Content (%)')
plt.ylabel('Liquid Limit')
plt.title('Scatter Plot with Linear Correlations')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%

# Create a new figure
plt.figure(figsize=(10, 6))

# Original data: COMARF scatter plot
plt.scatter(COMRF['plastic_limit'], COMRF['predicted_dry_unit_weight_0.85'], color='blue', label='COMRF data 0.85', alpha=0.6)
#plt.scatter(COMARF['plastic_limit'], COMARF['predicted_dry_unit_weight_0.60'], color='yellow', label='Auto Encoded Data (COMARF) 0.60', alpha=0.6)
#plt.scatter(COMARF['plastic_limit'], COMARF['predicted_dry_unit_weight_0.75'], color='purple', label='Auto Encoded Data (COMARF) 0.75', alpha=0.6)
# Random Forest data: COMARF_scenario1 scatter plot
plt.scatter(COMARF_s1['plastic_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.85'], color='green', label='Predicted dry unit weight out of Scenario 1 0.85', alpha=0.6)
plt.scatter(COMARF_s1['plastic_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.60'], color='yellow', label='Predicted dry unit weight out of Scenario 1 0.60', alpha=0.6)
plt.scatter(COMARF_s1['plastic_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.75'], color='black', label='Predicted dry unit weight out of Scenario 1 0.75', alpha=0.6)
#plt.scatter(COMARF_s1['plastic_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.60'], color='pink', label='Predicted dry unit weight out of Scenario 1', alpha=0.6)
#plt.scatter(COMARF_s1['plastic_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.75'], color='cyan', label='Predicted dry unit weight out of Scenario 1', alpha=0.6)
# Plot the linear correlation: dry_unit_weight = 13.34 - 0.21 * plastic_limit
plastic_limit_range = np.linspace(5, 55, 300)  # Interval for Plastic Limit and Plasticity Index
dry_unit_weight_linear = 19.34 - 0.21 * plastic_limit_range
plt.plot(plastic_limit_range, dry_unit_weight_linear, color='red', label='Linear Correlation Dry_unit_weight = (13.34 - 0.21 * PL) R2=0.75')

# Plot Gurtag equation: dry_unit_weight = [33.85 * log(0.94 * PL) * 9.81] / 1000
OMC = 0.94 * plastic_limit_range/100

dry_unit_weight_gurtag = (-13.58 * np.log(OMC) + 33.83)
#plt.plot(plastic_limit_range, dry_unit_weight_gurtag, color='purple', label='Gurtag Equation')


# Labeling the axes
plt.xlabel('Atterberg Limits (Plastic Limit)')
plt.ylabel('Dry Unit Weight (kN/m³)')
plt.title('Dry Unit Weight vs. Atterberg Limits')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%


# Create a new figure
plt.figure(figsize=(10, 6))

# Original data: COMARF scatter plot
plt.scatter(COMRF['plasticity_index'], COMRF['predicted_dry_unit_weight_0.85'], color='blue', label='COMRF Data', alpha=0.6)

# Random Forest data: COMARF_scenario1 scatter plot
plt.scatter(COMARF_s1['Predicted_plasticity_index'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.85'], color='green', label='Predicted dry unit weight out of Scenario 1 0.85', alpha=0.6)
plt.scatter(COMARF_s1['Predicted_plasticity_index'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.60'], color='yellow', label='Predicted dry unit weight out of Scenario 1 0.60', alpha=0.6)
plt.scatter(COMARF_s1['Predicted_plasticity_index'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.75'], color='black', label='Predicted dry unit weight out of Scenario 1 0.75', alpha=0.6)
# Plot the linear correlation: dry_unit_weight = 13.34 - 0.21 * plastic_limit
plastic_index_range = np.linspace(5, 55, 300)  # Interval for Plastic Limit and Plasticity Index
dry_unit_weight_linear = 16.68 - 0.08 * plastic_index_range
plt.plot(plastic_limit_range, dry_unit_weight_linear, color='red', label='Linear Correlation, Dry_unit_weight = 16.68 - 0.08 * PI, R2=0.45)')


# Plot first Nagaraj equation: dry_unit_weight = 20.64 - 0.19 * plastic_index
dry_unit_weight_nagaraj1 = 20.64 - 0.19 * plastic_index_range
plt.plot(plastic_index_range, dry_unit_weight_nagaraj1, color='black', label='Nagaraj Equation, Dry_unit_weight = 20.64 - 0.19 * PI')

# Plot modified Nagaraj equation: dry_unit_weight = 20.35 - 0.17 * plasticity_index * (1 - clay_content / 100)
clay_content_range = np.linspace(5, 55, 300)  # Assuming same interval for clay_content
dry_unit_weight_nagaraj2 = 20.35 - 0.17 * plastic_index_range * (1 - clay_content_range / 100)
plt.plot(plastic_index_range, dry_unit_weight_nagaraj2, color='orange', label='Modified Nagaraj Equation')

# Labeling the axes
plt.xlabel('Atterberg Limits (Plasticity Index)')
plt.ylabel('Dry Unit Weight (kN/m³)')
plt.title('Dry Unit Weight vs. Atterberg Limits')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%

# Create a new figure
plt.figure(figsize=(10, 6))

# Original data: COMARF scatter plot
plt.scatter(COMRF['liquid_limit'], COMRF['predicted_dry_unit_weight_0.85'], color='blue', label='COMRF Data', alpha=0.6)

# Random Forest data: COMARF_scenario1 scatter plot
plt.scatter(COMARF_s1['liquid_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.85'], color='green', label='Predicted dry unit weight out of Scenario 1 0.85', alpha=0.6)
plt.scatter(COMARF_s1['liquid_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.75'], color='yellow', label='Predicted dry unit weight out of Scenario 1 0.60', alpha=0.6)
plt.scatter(COMARF_s1['liquid_limit'], COMARF_s1['Predicted_predicted_dry_unit_weight_0.60'], color='black', label='Predicted dry unit weight out of Scenario 1 0.75', alpha=0.6)

# Plot the linear correlation: dry_unit_weight = 13.34 - 0.21 * plastic_limit
liquid_limit_range = np.linspace(5, 80, 300)  # Interval for Plastic Limit and Plasticity Index
dry_unit_weight_linear = 18.21 - 0.08 * liquid_limit_range
plt.plot(liquid_limit_range, dry_unit_weight_linear, color='red', label='Linear Correlation, Dry_unit_weight = 18.21 - 0.08 * LL, R2=0.63')


# Constants
water_unit_weight = 9.81  # kN/m³ (specific weight of water)
specific_gravity_solids = 2.7  # Assumed specific gravity of solids

# Liquid limit range
liquid_limit_range = np.linspace(5, 80, 300)  # Adjust this range as necessary

# Step 1: Calculate void ratio
void_ratio = ((liquid_limit_range/100) * specific_gravity_solids) - 0.15

# Step 2: Calculate dry unit weight using the formula
dry_unit_weight_budhu = (specific_gravity_solids * water_unit_weight) / (1 + void_ratio)

plt.plot(liquid_limit_range, dry_unit_weight_budhu, color='black', label='corrected Budhu Equation')

# Labeling the axes
plt.xlabel('Atterberg Limits (liquid limit)')
plt.ylabel('Dry Unit Weight (kN/m³)')
plt.title('Dry Unit Weight vs. Atterberg Limits')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%



# Scatter plot friction angle vs plasticity index
plt.figure(figsize=(10, 6))
#plt.scatter(COMARF['plasticity_index'], COMARF['friction_angle'], color='black', label='COMARF Data', alpha=0.6)

# Generate a synthetic range for plasticity index (PI) and clay content
PI_range = np.linspace(5, 100, 300)
clay_content_range = np.linspace(5, 75, 300)

# Budhu equation: Friction Angle = sin^-1 (0.35 - 0.1 * ln(PI / 100))
friction_angle_budhu = np.degrees(np.arcsin(0.35 - 0.1 * np.log(COMARF['plasticity_index'] / 100)))
plt.scatter(COMARF['plasticity_index'], friction_angle_budhu, color='red', label='Budhu Equation Friction Angle (cs) = sin^-1 (0.35 - 0.1 * ln(PI / 100)')

# Amertunga 1 equation: Friction Angle = 43 - 10 * log(PI)
friction_angle_amertunga1 = 43 - 10 * np.log10(COMARF['plasticity_index'])
plt.scatter(COMARF['plasticity_index'], friction_angle_amertunga1, color='blue', label='Amertunga Equation Friction Angle (peak) = 43 - 10 * log(PI)')

# Third equation: sin(friction angle) = 0.247 + 0.409 * (PI / LL)
# Assume an average LL value for plotting purposes
LL_average = COMARF['liquid_limit']  # Use the actual average liquid limit from COMARF
friction_angle_third = np.degrees(np.arcsin(0.247 + 0.409 * (COMARF['plasticity_index'] / LL_average)))
plt.scatter(COMARF['plasticity_index'], friction_angle_third, color='green', label='Amertunga Equation sin(Friction angle) = 0.247 + 0.409 * (PI / LL)')

# Fourth equation: Friction Angle = -0.743 * Clay Content + 36
friction_angle_clay = -0.743 * clay_content_range + 36
#plt.plot(clay_content_range, friction_angle_clay, color='purple', label='Clay Content Equation')

# Fifth equation: sin(friction angle) = 0.8 - 0.094 * ln(PI)
friction_angle_fifth = np.degrees(np.arcsin(0.8 - 0.094 * np.log(COMARF['plasticity_index'])))
plt.scatter(COMARF['plasticity_index'], friction_angle_fifth, color='orange', label='Akayuli  Equation sin(friction angle) = 0.8 - 0.094 * ln(PI)')

# Labeling the axes
plt.xlabel('Plasticity Index')
plt.ylabel('Friction Angle (degrees)')
plt.title('Friction Angle vs Plasticity Index with Various Equations')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%

# Scatter plot friction angle vs plasticity index
plt.figure(figsize=(10, 6))
#plt.scatter(COMARF['plasticity_index'], COMARF['friction_angle'], color='black', label='COMARF Data', alpha=0.6)

# Generate a synthetic range for plasticity index (PI) and clay content
PI_range = np.linspace(5, 100, 300)
clay_content_range = np.linspace(5, 75, 300)

# Budhu equation: Friction Angle = sin^-1 (0.35 - 0.1 * ln(PI / 100))
friction_angle_budhu = np.degrees(np.arcsin(0.35 - 0.1 * np.log(PI_range / 100)))
plt.plot(PI_range, friction_angle_budhu, color='red', label='Budhu Equation')

# Amertunga 1 equation: Friction Angle = 43 - 10 * log(PI)
friction_angle_amertunga1 = 43 - 10 * np.log10(PI_range)
plt.plot(PI_range, friction_angle_amertunga1, color='blue', label='Amertunga Equation 1')

# Third equation: sin(friction angle) = 0.247 + 0.409 * (PI / LL)
# Assume an average LL value for plotting purposes
LL_average = np.mean(COMARF['liquid_limit'])  # Use the actual average liquid limit from COMARF
friction_angle_third = np.degrees(np.arcsin(0.247 + 0.409 * (PI_range / LL_average)))
plt.plot(PI_range, friction_angle_third, color='green', label='Amertunga Equation 2')

# Fourth equation: Friction Angle = -0.743 * Clay Content + 36
friction_angle_clay = -0.743 * clay_content_range + 36
#plt.plot(clay_content_range, friction_angle_clay, color='purple', label='Clay Content Equation')

# Fifth equation: sin(friction angle) = 0.8 - 0.094 * ln(PI)
friction_angle_fifth = np.degrees(np.arcsin(0.8 - 0.094 * np.log(PI_range)))
plt.plot(PI_range, friction_angle_fifth, color='orange', label='Akayuli  Equation')

# Labeling the axes
plt.xlabel('Plasticity Index')
plt.ylabel('Friction Angle (degrees)')
plt.title('Friction Angle vs Plasticity Index with Various Equations')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%




# Scatter plot friction angle vs plasticity index
plt.figure(figsize=(10, 6))
plt.scatter(COMARF_s2['clay_content_100%'], COMARF_s2['Predicted_predicted_dry_unit_weight_0.85'], color='green', label='predicted dry unit weight 0.85', alpha=1)
plt.scatter(COMARF_s2['clay_content_100%'], COMARF_s2['Predicted_predicted_dry_unit_weight_0.75'], color='yellow', label='predicted dry unit weight 0.75', alpha=1)
plt.scatter(COMARF_s2['clay_content_100%'], COMARF_s2['Predicted_predicted_dry_unit_weight_0.60'], color='black', label='predicted dry unit weight 0.60', alpha=1)

plt.scatter(COMARF_s2['clay_content_100%'], COMARF_s2['Actual_predicted_dry_unit_weight_0.85'], color='blue', label='combined dry unit weight', alpha=0.6)

linear_dry_unit_weight = -0.0738 * clay_content_range + 16.748
plt.plot(clay_content_range, linear_dry_unit_weight, color='orange', label='Linear equaion R2=0.62')

# Labeling the axes
plt.xlabel('Clay content')
plt.ylabel('Dry unit weight')
plt.title('Clay content vs Dry unit weight (combined and predicted)')

# Add a legend
plt.legend()

# Show the plot
plt.show()