import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import seaborn as sns
from sklearn.metrics import r2_score
import optuna
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from openpyxl import load_workbook, Workbook
import inspect
import os  # Import os module for file handling

#load the data



CC = pd.read_excel('clay_compaction_ML.xlsx')
CI = pd.read_excel('Clay_ins_ML.xlsx')

CT = pd.read_excel('Clay_Tri_F.xlsx')


#dropping unimportant columns for the performance of the model
# Load the data
CC = pd.read_excel('clay_compaction_ML.xlsx')
CI = pd.read_excel('Clay_ins_ML.xlsx')
CT = pd.read_excel('Clay_Tri_F.xlsx')

# Drop unimportant columns
CI = CI.drop(columns=[
    'Unnamed: 0', 'clay_content', 'silt_content', 'sand_content',
    'x', 'y', 'mass_loss', 'void', 'dry_unit_weight','class_1_encoded','organic_content','salt_content'
])

CC = CC[CC['degree_of_compaction'] >= 97]
CC = CC.drop(columns=['file_name', 'W_max', 'degree_of_compaction', 'mass'])

#CI['dry_density_in_situ'] = (CI['dry_unit_weight'] * 1000)/9.81


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
CI = CI[CI['class%'] != 'other']
unique_classes_CI = CI['class%'].unique()
# Separate encoded columns before feeding into autoencoder
encoded_columns = CI[['encoded_class', 'encoded_activity']]
CI = CI.drop(columns=['class%', 'activity_class', 'encoded_class', 'encoded_activity'])


#CC = CC.drop(columns=['file_name',
#                      'W_max',
#                      'degree_of_compaction',
#                      'mass'])





unique_classes_CT = CT['class 1'].unique()




CT['q'] = CT['q'].fillna(CT['sigma_1'] - CT['sigma_3'])
# On-hot encoding for the clay classes


#applying the encoding of Ins files to the triaxial data

# Define the mapping of class1 values to new column values
class1_mapping = {
    'ks1': '1000000',
    'ks2': '0100000',
    'ks3': '0010000',
    'ks4': '0001000',
    'kz1': '0000100',
    'kz2': '0000010',
    'kz3': '0000001'
}

# Create the new column using the map method
CT['class_encoded'] = CT['class 1'].map(class1_mapping)
CT.drop('class 1', axis=1, inplace=True)


#%%


# Load your dataset here (assuming CI is already loaded)

# 2. Split Data
CI_train, CI_test = train_test_split(CI, test_size=0.2, random_state=42)  # First split into training and test sets
CI_train, CI_val = train_test_split(CI_train, test_size=0.2, random_state=42)  # Then split the training set into training and validation sets

input_dim = CI.shape[1]  # Number of features in the dataset 

# Check for non-numeric values in CI
for col in CI.columns:
    if not pd.api.types.is_numeric_dtype(CI[col]):
        print(f"Non-numeric values found in column: {col}")
        print(CI[col].unique())

# Check for non-numeric values in CI_test   
for col in CI_test.columns:
    if not pd.api.types.is_numeric_dtype(CI_test[col]):
        print(f"Non-numeric values found in column: {col}")
        print(CI_test[col].unique())

# 3. Optuna Objective Function for Autoencoder
def objective(trial):
    encoding_dim = trial.suggest_int('encoding_dim', 1, input_dim // 5) 
    activation = trial.suggest_categorical('activation', ['relu'])
    optimizer = trial.suggest_categorical('optimizer', ['adam'])
    epochs = trial.suggest_int('epochs', 800, 1200)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation=activation)(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    history = autoencoder.fit(CI_train, CI_train, epochs=epochs, batch_size=batch_size, 
                              validation_data=(CI_val, CI_val), verbose=0)

    reconstructed_data = autoencoder.predict(CI_test)
    reconstruction_loss = np.mean(np.square(reconstructed_data - CI_test))
    
    for epoch, loss in enumerate(history.history['loss']):
        trial.set_user_attr(f'train_loss_epoch_{epoch + 1}', loss)
    for epoch, val_loss in enumerate(history.history['val_loss']):
        trial.set_user_attr(f'val_loss_epoch_{epoch + 1}', val_loss)
    return reconstruction_loss

study_name = f"Autoencoder_CI_new" 
storage = optuna.storages.RDBStorage('sqlite:///db.sqlite3')

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except KeyError:
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage)
else:
    study_name = "AUTOADD_" + study_name
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=storage)

study.optimize(objective, n_trials=100)  # Updated to 100 trials

# Retrieve Best Parameters
best_params = study.best_params
print("Best parameters:", best_params)

# Define Autoencoder Model with Best Parameters
input_layer = Input(shape=(input_dim,))
encoded = Dense(best_params['encoding_dim'], activation=best_params['activation'])(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=best_params['optimizer'], loss='mse')

# Train the autoencoder with best parameters
history = autoencoder.fit(CI_train, CI_train, epochs=best_params['epochs'], 
                          batch_size=best_params['batch_size'], 
                          validation_data=(CI_val, CI_val), verbose=0)

# Predict with the trained autoencoder
reconstructed_data = autoencoder.predict(CI_test)

# Visualize Results
reconstructed_df = pd.DataFrame(reconstructed_data, columns=CI_test.columns)
reconstructed_df['Original'] = CI_test.values.tolist()

subplots_per_figure = 3
num_figures = len(CI.columns) // subplots_per_figure + (len(CI.columns) % subplots_per_figure > 0)

for fig_num in range(num_figures):
    start_index = fig_num * subplots_per_figure
    end_index = min((fig_num + 1) * subplots_per_figure, len(CI.columns))

    fig, axes = plt.subplots(nrows=1, ncols=end_index - start_index, figsize=(15, 5))

    if end_index - start_index == 1:
        axes = [axes]  # Make axes a list so it can be indexed

    for i, col in enumerate(CI.columns[start_index:end_index]):
        r2 = r2_score(reconstructed_df['Original'].apply(lambda x: x[i + start_index]), reconstructed_df[col])

        axes[i].scatter(reconstructed_df['Original'].apply(lambda x: x[i + start_index]), reconstructed_df[col], alpha=0.5)
        axes[i].set_title(f'{col} - R-squared: {r2:.3f}')
        axes[i].set_xlabel('Original')
        axes[i].set_ylabel('Reconstructed')
 
    plt.tight_layout()
    plt.show()

# Define the Excel file path
excel_file = 'Predicted_Tri_data_autoEncoders_B.xlsx'

# Check if the Excel file exists; if not, create a new one
if not os.path.isfile(excel_file):
    wb = Workbook()
    wb.save(excel_file)

# Save predicted data to Excel with a dynamically named sheet
with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
    # Get study name for the sheet
    sheet_name = study_name.replace('AUTOADD_', '')  # Remove 'AUTOADD_' prefix

    # Create a DataFrame with the reconstructed data
    reconstructed_df = pd.DataFrame(reconstructed_data, columns=CI_test.columns)
    reconstructed_df['Original'] = CI_test.values.tolist()

    # Write DataFrame to Excel
    reconstructed_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Predicted data saved to {excel_file} under sheet '{sheet_name}'")