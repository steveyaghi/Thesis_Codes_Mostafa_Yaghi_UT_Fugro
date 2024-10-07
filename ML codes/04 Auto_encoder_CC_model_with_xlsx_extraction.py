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

# Preprocessing CC Data
CC = CC[CC['degree_of_compaction'] >= 97]
CC = CC.drop(columns=['file_name', 'W_max', 'degree_of_compaction','dry_density_in_situ'])

# Replace NaNs with column means for CC
CC_data = CC.fillna(CC.mean())

# Standardize the CC data
scaler_CC = StandardScaler()
CC_data_scaled = scaler_CC.fit_transform(CC_data)

# Define input dimension for CC
input_dim_CC = CC_data.shape[1]

# Define Autoencoder Model with Best Parameters for CC
input_layer_CC = Input(shape=(input_dim_CC,))
encoded_CC = Dense(128, activation='relu')(input_layer_CC)
encoded_CC = BatchNormalization()(encoded_CC)
encoded_CC = Dropout(0.2)(encoded_CC)
encoded_CC = Dense(3, activation='relu')(encoded_CC)  # encoding_dim=3
decoded_CC = Dense(128, activation='relu')(encoded_CC)
decoded_CC = Dense(input_dim_CC, activation='linear')(decoded_CC)
autoencoder_CC = Model(input_layer_CC, decoded_CC)
autoencoder_CC.compile(optimizer='adam', loss='mse')

# Learning rate schedule (use a constant for simplicity)
def constant_lr(epoch):
    return 0.001

lr_schedule = LearningRateScheduler(constant_lr)

# Train the autoencoder with CC data
history_CC = autoencoder_CC.fit(CC_data_scaled, CC_data_scaled, 
                                epochs=1197,  # epochs=1197
                                batch_size=32, 
                                validation_split=0.2,  # 20% validation data
                                callbacks=[lr_schedule], 
                                verbose=0)

# Predict with the trained autoencoder for CC data
reconstructed_data_scaled_CC = autoencoder_CC.predict(CC_data_scaled)

# Reverse the scaling for the reconstructed CC data
reconstructed_data_CC = scaler_CC.inverse_transform(reconstructed_data_scaled_CC)

# Check for NaNs in reconstructed data
print("Checking for NaNs in reconstructed CC data:", np.isnan(reconstructed_data_CC).sum())

# Create DataFrame for original and reconstructed CC values
original_df_CC = CC_data.reset_index(drop=True)  # Ensure the index is reset
reconstructed_df_CC = pd.DataFrame(reconstructed_data_CC, columns=CC_data.columns).add_suffix('_reconstructed')

# Combine original and reconstructed CC data
results_df_CC = pd.concat([original_df_CC, reconstructed_df_CC], axis=1)

# Check the shape of the results DataFrame for CC
print("Results DataFrame shape (CC):", results_df_CC.shape)

# Save to Excel
results_df_CC.to_excel('CC_AUTO_ENCODED_FULL.xlsx', index=False)

# Initialize dictionaries to store the MSE and MAE for each feature
mse_scores_CC = {}
mae_scores_CC = {}

# Calculate MSE and MAE for each feature in CC data
for col in CC_data.columns:
    reconstructed_col = col + '_reconstructed'
    if reconstructed_col in reconstructed_df_CC.columns:
        original = CC_data[col].values
        reconstructed = reconstructed_df_CC[reconstructed_col].values

        # Ensure no NaNs are present in the data
        if np.any(np.isnan(original)) or np.any(np.isnan(reconstructed)):
            print(f"NaNs detected in the data for column {col}. Filling NaNs with 0.")
            original = np.nan_to_num(original)
            reconstructed = np.nan_to_num(reconstructed)

        # Compute MSE and MAE for this column
        mse_value = mean_squared_error(original, reconstructed)
        mae_value = mean_absolute_error(original, reconstructed)

        # Store the results in the dictionaries
        mse_scores_CC[col] = mse_value
        mae_scores_CC[col] = mae_value

        # Print the results
        print(f'{col} - Mean Squared Error (MSE): {mse_value:.4f}')
        print(f'{col} - Mean Absolute Error (MAE): {mae_value:.4f}')

# Calculate R-squared scores for CC data
r2_scores_CC = {}
for col in CC_data.columns:
    reconstructed_col = col + '_reconstructed'
    if reconstructed_col in reconstructed_df_CC.columns:
        try:
            r2 = r2_score(CC_data[col], reconstructed_df_CC[reconstructed_col])
            r2_scores_CC[col] = r2
            print(f'R-squared for {col}: {r2:.3f}')
        except ValueError as e:
            print(f"Error calculating R-squared for {col}: {e}")

# Plotting for CC data
subplots_per_figure = 3
num_figures = len(CC_data.columns) // subplots_per_figure + (1 if len(CC_data.columns) % subplots_per_figure > 0 else 0)

for fig_num in range(num_figures):
    start_index = fig_num * subplots_per_figure
    end_index = min((fig_num + 1) * subplots_per_figure, len(CC_data.columns))

    fig, axes = plt.subplots(nrows=1, ncols=end_index - start_index, figsize=(15, 5))

    # Ensure axes is always treated as a list for consistency
    if end_index - start_index == 1:
        axes = [axes]

    for i, col in enumerate(CC_data.columns[start_index:end_index]):
        reconstructed_col = col + '_reconstructed'
        if reconstructed_col in reconstructed_df_CC.columns:
            axes[i].scatter(CC_data[col], reconstructed_df_CC[reconstructed_col], alpha=0.5)
            axes[i].set_title(f'{col} - R-squared: {r2_scores_CC.get(col, "N/A"):.3f}')
            axes[i].set_xlabel('Original')
            axes[i].set_ylabel('Reconstructed')

    plt.tight_layout()
    plt.show()

# Plotting Residuals for CC data
for fig_num in range(num_figures):
    start_index = fig_num * subplots_per_figure
    end_index = min((fig_num + 1) * subplots_per_figure, len(CC_data.columns))

    fig, axes = plt.subplots(nrows=1, ncols=end_index - start_index, figsize=(15, 5))

    # Ensure axes is always treated as a list for consistency
    if end_index - start_index == 1:
        axes = [axes]

    for i, col in enumerate(CC_data.columns[start_index:end_index]):
        reconstructed_col = col + '_reconstructed'
        if reconstructed_col in reconstructed_df_CC.columns:
            original = CC_data[col].values
            reconstructed = reconstructed_df_CC[reconstructed_col].values
            residuals = original - reconstructed

            # Plot residuals
            axes[i].scatter(original, residuals, alpha=0.5)
            axes[i].axhline(y=0, color='r', linestyle='--')  # Add a line at y=0
            axes[i].set_title(f'{col} - Residuals')
            axes[i].set_xlabel('Original')
            axes[i].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()
