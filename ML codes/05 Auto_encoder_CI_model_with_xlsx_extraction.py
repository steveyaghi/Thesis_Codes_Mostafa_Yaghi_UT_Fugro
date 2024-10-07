import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import math

# Load the data
CC = pd.read_excel('clay_compaction_ML.xlsx')
CI = pd.read_excel('Clay_ins_ML.xlsx')
CT = pd.read_excel('Clay_Tri_F.xlsx')

# Drop unimportant columns
CI = CI.drop(columns=[
    'Unnamed: 0', 'clay_content', 'silt_content', 'sand_content',
    'x', 'y', 'mass_loss', 'void', 'dry_unit_weight', 'class_1_encoded', 'organic_content', 'salt_content'
])

CC = CC[CC['degree_of_compaction'] >= 97]
CC = CC.drop(columns=['file_name', 'W_max', 'degree_of_compaction'])

# One-hot encode the 'class%' column in CI
CI_encoded = pd.get_dummies(CI['class%'])
CI_encoded = CI_encoded.astype(int)

# Combine encoded columns into one, keeping 0s and 1s
CI_encoded['encoded_class'] = CI_encoded.apply(lambda row: ''.join(row.astype(str)), axis=1)

# Join the encoded DataFrame with the original CI DataFrame
CI = CI.join(CI_encoded['encoded_class'])

CI_encoded_activity = pd.get_dummies(CI['activity_class'])
CI_encoded_activity = CI_encoded_activity.astype(int)

# Combine encoded columns into one, keeping 0s and 1s
CI_encoded_activity['encoded_activity'] = CI_encoded_activity.apply(lambda row: ''.join(row.astype(str)), axis=1)

# Join the encoded DataFrame with the original CI DataFrame
CI = CI.join(CI_encoded_activity['encoded_activity'])

# Keep encoded columns for later use
encoded_columns = CI[['encoded_class', 'encoded_activity']]

# Separate encoded columns before feeding into autoencoder
CI_data = CI.drop(columns=['class%', 'activity_class', 'encoded_class', 'encoded_activity', 'activity'])

def remove_outliers_std(df, threshold=3):
    """
    Remove outliers from all numeric columns in a DataFrame based on the standard deviation method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    threshold (float): The number of standard deviations to use for defining outliers. Default is 3.

    Returns:
    pd.DataFrame: A DataFrame with outliers replaced by NaN.
    dict: A dictionary with counts of outliers removed for each column.
    """
    outliers_removed = {}
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            mean = df[col].mean()
            std_dev = df[col].std()
            lower_bound = mean - threshold * std_dev
            upper_bound = mean + threshold * std_dev
            
            # Count the number of outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_removed[col] = outliers.shape[0]
            
            # Replace outliers with NaN
            df[col] = df[col].where((df[col] >= lower_bound) & (df[col] <= upper_bound))
    
    return df, outliers_removed

# Apply the function to your entire DataFrame
CI_cleaned, outliers_info = remove_outliers_std(CI_data)
CI_data['sand_content_100%'] = CI_data['sand_content_100%'].fillna(CI_data['sand_content_100%'].mean())

print("Number of outliers removed per column:")
for col, count in outliers_info.items():
    print(f"{col}: {count} outliers removed")

# Define the best parameters explicitly
best_params = {
    'encoding_dim': 3,
    'activation': 'relu',
    'optimizer': 'adam',
    'epochs': 1197,
    'batch_size': 32  
}

# Print best parameters
print("Best parameters:", best_params)

# Replace NaNs in the CI_data with column means
CI_data = CI_data.fillna(CI_data.mean())

# Standardize the data
scaler = StandardScaler()
CI_data_scaled = scaler.fit_transform(CI_data)

# Define input dimension
input_dim = CI_data.shape[1]

# Define Autoencoder Model with Best Parameters
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(best_params['encoding_dim'], activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=best_params['optimizer'], loss='mse')

# Define learning rate schedules
def constant_lr(epoch):
    return 0.001  # Example: constant learning rate

def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr

def exponential_decay(epoch):
    initial_lr = 0.001
    k = 0.1
    lr = initial_lr * math.exp(-k * epoch)
    return lr

# Choose a learning rate schedule
lr_schedule = LearningRateScheduler(constant_lr)  # Or step_decay, exponential_decay

# Train the autoencoder with the learning rate schedule
history = autoencoder.fit(CI_data_scaled, CI_data_scaled, epochs=best_params['epochs'],
                          batch_size=best_params['batch_size'],
                          validation_split=0.2,  # Use 20% of the data for validation
                          callbacks=[lr_schedule],
                          verbose=0)

# Predict with the trained autoencoder
reconstructed_data_scaled = autoencoder.predict(CI_data_scaled)

# Reverse the scaling for the reconstructed data
reconstructed_data = scaler.inverse_transform(reconstructed_data_scaled)

# Check for NaNs in reconstructed data
print("Checking for NaNs in reconstructed data:", np.isnan(reconstructed_data).sum())

# Create DataFrame for original and reconstructed values
original_df = CI_data.reset_index(drop=True)  # Ensure the index is reset
reconstructed_df = pd.DataFrame(reconstructed_data, columns=CI_data.columns).add_suffix('_reconstructed')

# Combine encoded columns with the original and reconstructed data
encoded_full = CI[['encoded_class', 'encoded_activity']].reset_index(drop=True)
results_df = pd.concat([encoded_full, original_df, reconstructed_df], axis=1)

# Check the shape of the results DataFrame
print("Results DataFrame shape:", results_df.shape)

# Save to Excel
results_df.to_excel('CI_AUTO_ENCODED_FULL.xlsx', index=False)

# Initialize dictionaries to store the MSE and MAE for each feature
mse_scores = {}
mae_scores = {}

# Calculate MSE and MAE for each feature
for col in CI_data.columns:
    reconstructed_col = col + '_reconstructed'
    if reconstructed_col in reconstructed_df.columns:
        original = CI_data[col].values
        reconstructed = reconstructed_df[reconstructed_col].values

        # Ensure no NaNs are present in the data
        if np.any(np.isnan(original)) or np.any(np.isnan(reconstructed)):
            print(f"NaNs detected in the data for column {col}. Filling NaNs with 0.")
            original = np.nan_to_num(original)
            reconstructed = np.nan_to_num(reconstructed)

        # Compute MSE and MAE for this column
        mse_value = mean_squared_error(original, reconstructed)
        mae_value = mean_absolute_error(original, reconstructed)

        # Store the results in the dictionaries
        mse_scores[col] = mse_value
        mae_scores[col] = mae_value

        # Print the results
        print(f'{col} - Mean Squared Error (MSE): {mse_value:.4f}')
        print(f'{col} - Mean Absolute Error (MAE): {mae_value:.4f}')

# Calculate R-squared scores
r2_scores = {}
for col in CI_data.columns:
    reconstructed_col = col + '_reconstructed'
    if reconstructed_col in reconstructed_df.columns:
        try:
            r2 = r2_score(CI_data[col], reconstructed_df[reconstructed_col])
            r2_scores[col] = r2
            print(f'R-squared for {col}: {r2:.3f}')
        except ValueError as e:
            print(f"Error calculating R-squared for {col}: {e}")

# Plotting
subplots_per_figure = 3
num_figures = len(CI_data.columns) // subplots_per_figure + (1 if len(CI_data.columns) % subplots_per_figure > 0 else 0)

for fig_num in range(num_figures):
    start_index = fig_num * subplots_per_figure
    end_index = min((fig_num + 1) * subplots_per_figure, len(CI_data.columns))

    fig, axes = plt.subplots(nrows=1, ncols=end_index - start_index, figsize=(15, 5))

    # Ensure axes is always treated as a list for consistency
    if end_index - start_index == 1:
        axes = [axes]

    for i, col in enumerate(CI_data.columns[start_index:end_index]):
        reconstructed_col = col + '_reconstructed'
        if reconstructed_col in reconstructed_df.columns:
            axes[i].scatter(CI_data[col], reconstructed_df[reconstructed_col], alpha=0.5)
            axes[i].set_title(f'{col} - R-squared: {r2_scores.get(col, "N/A"):.3f}')
            axes[i].set_xlabel('Original')
            axes[i].set_ylabel('Reconstructed')

    plt.tight_layout()
    plt.show()

# Plotting Residuals
subplots_per_figure = 3
num_figures = len(CI_data.columns) // subplots_per_figure + (1 if len(CI_data.columns) % subplots_per_figure > 0 else 0)

for fig_num in range(num_figures):
    start_index = fig_num * subplots_per_figure
    end_index = min((fig_num + 1) * subplots_per_figure, len(CI_data.columns))

    fig, axes = plt.subplots(nrows=1, ncols=end_index - start_index, figsize=(15, 5))

    # Ensure axes is always treated as a list for consistency
    if end_index - start_index == 1:
        axes = [axes]

    for i, col in enumerate(CI_data.columns[start_index:end_index]):
        reconstructed_col = col + '_reconstructed'
        if reconstructed_col in reconstructed_df.columns:
            original = CI_data[col].values
            reconstructed = reconstructed_df[reconstructed_col].values
            residuals = original - reconstructed

            # Plot residuals
            axes[i].scatter(original, residuals, alpha=0.5)
            axes[i].axhline(y=0, color='r', linestyle='--')  # Add a line at y=0
            axes[i].set_title(f'{col} - Residuals')
            axes[i].set_xlabel('Original')
            axes[i].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()
