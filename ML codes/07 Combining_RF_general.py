import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
CC = pd.read_excel('CCA.xlsx')
CI = pd.read_excel('CIA.xlsx')

# Drop unimportant columns from CI
#CI = CI.drop(columns=[
#    'Unnamed: 0', 'clay_content', 'silt_content', 'sand_content',
#    'x', 'y', 'mass_loss', 'void'
#])

# Filter and drop unimportant columns from CC
#CC = CC[CC['degree_of_compaction'] >= 97]
#CC = CC.drop(columns=['file_name', 'W_max', 'degree_of_compaction', 'dry_density_in_situ'])

# Align feature columns - select only 'water_content'
X_CC = CC[['water_content']]
y_CC = CC[['dry_density_proctor', 'dry_unit_weight']]

# Train-test split for CC
X_train, X_test, y_train, y_test = train_test_split(X_CC, y_CC, test_size=0.2, random_state=42)

# Scale the features (only for X since Random Forest does not require scaling for y)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Define the Random Forest model with the specified parameters
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=50,
    max_features='sqrt',
    max_samples=0.85,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)

# Train the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

for i, param in enumerate(['dry_density_proctor', 'dry_unit_weight']):
    print(f'R² for {param}: {r2[i]:.4f}')
    print(f'MSE for {param}: {mse[i]:.4f}')
    print(f'MAE for {param}: {mae[i]:.4f}\n')

# Residuals plot for each predicted parameter
plt.figure(figsize=(14, 10))
for i, param in enumerate(['dry_density_proctor', 'dry_unit_weight']):
    residuals = y_test.iloc[:, i] - y_pred[:, i]
    plt.subplot(2, 2, i + 1)
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals for {param}')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Define water content scenarios
water_content_scenarios = ['water_content_0.75', 'water_content_0.60', 'water_content_0.85']

# Loop through each water content scenario
for scenario in water_content_scenarios:
    # Rename the water content column temporarily
    CI_temp = CI.rename(columns={scenario: 'water_content'}).copy()
    
    # Select the appropriate water content column (now renamed to 'water_content')
    X_CI = CI_temp[['water_content']]
    
    # Scale the CI features based on the scaler fitted to the CC features
    X_CI_scaled = scaler_X.transform(X_CI)

    # Make predictions for the CI dataset
    predictions = rf_model.predict(X_CI_scaled)
    
    # Assign predictions to new columns with scenario suffix
    CI[f'predicted_dry_density_proctor_{scenario[-4:]}'] = predictions[:, 0]
    CI[f'predicted_dry_unit_weight_{scenario[-4:]}'] = predictions[:, 1]
    
    # Save the combined dataframe for each scenario to a separate Excel file
    CI.to_excel(f'COMBINED_AutoEncoded_RF_{scenario}.xlsx', index=False)

    print(f"Combined dataframe saved to COMBINED_AutoEncoded_RF_{scenario}.xlsx")
