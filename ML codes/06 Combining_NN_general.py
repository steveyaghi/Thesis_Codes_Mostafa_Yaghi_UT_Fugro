import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
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

# Split CC into features (X) and target (y)
X_CC = CC.drop(columns=['dry_density_proctor', 'dry_unit_weight'])
y_CC = CC[['dry_density_proctor', 'dry_unit_weight']]

# Train-test split for CC
X_train, X_test, y_train, y_test = train_test_split(X_CC, y_CC, test_size=0.2, random_state=42)

# Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Build the neural network model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(X_train_scaled.shape[1],)))
for _ in range(8):
    model.add(layers.Dense(250, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.55))
    model.add(layers.BatchNormalization())

model.add(layers.Dense(y_train_scaled.shape[1]))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')

# Learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=1200, batch_size=120,
                    validation_data=(X_test_scaled, y_test_scaled), verbose=0,
                    callbacks=[early_stopping, lr_scheduler])

# Evaluate the model on the test set
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled)

# Calculate R2, MSE, and MAE
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

for i, param in enumerate(['dry_density_proctor', 'dry_unit_weight']):
    print(f'R² for {param}: {r2[i]:.4f}')
    print(f'MSE for {param}: {mse[i]:.4f}')
    print(f'MAE for {param}: {mae[i]:.4f}\n')

# Residuals plot for each predicted parameter
predicted_parameters = ['dry_density_proctor', 'dry_unit_weight']

plt.figure(figsize=(14, 10))
for i, param in enumerate(predicted_parameters):
    plt.subplot(2, 2, i + 1)
    residuals = y_test[:, i] - y_pred[:, i]
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
    predictions_scaled = model.predict(X_CI_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # Assign predictions to new columns with scenario suffix
    CI[f'predicted_dry_density_proctor_{scenario[-4:]}'] = predictions[:, 0]
    CI[f'predicted_dry_unit_weight_{scenario[-4:]}'] = predictions[:, 1]
    
    # Save the combined dataframe for each scenario to a separate Excel file
    CI.to_excel(f'COMBINED_AutoEncoded_NT_{scenario}.xlsx', index=False)

    print(f"Combined dataframe saved to COMBINED_Auto_Encoded_NT_{scenario}.xlsx")
