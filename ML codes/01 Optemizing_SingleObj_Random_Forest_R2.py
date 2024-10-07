# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:54:23 2024

@author: mosty
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import inspect
from optuna.visualization import plot_contour, plot_edf, plot_parallel_coordinate 
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import openpyxl
from openpyxl import load_workbook



#load the data



CC = pd.read_excel('clay_compaction_ML.xlsx')
CI = pd.read_excel('Clay_ins_ML.xlsx')
CIA = pd.read_excel('Autoencoded_CI.xlsx')
CT = pd.read_excel('Clay_Tri_F.xlsx')


#dropping unimportant columns for the performance of the model
CI = CI.drop(columns=['Unnamed: 0',
                      'clay_content',
                      'silt_content',
                      'sand_content',
                      'activity',
                      'activity_class',
                      'x',
                      'y',
                      'salt_content',
                      'mass_loss',
                      'void',
                      'organic_content',])

CC = CC.drop(columns=['file_name',
                      'W_max',
                      'degree_of_compaction',
                      'mass'])



unique_classes_CI = CI['class%'].unique()

unique_classes_CT = CT['class 1'].unique()


CI = CI[CI['class%'] != 'other']

CT['q'] = CT['q'].fillna(CT['sigma_1'] - CT['sigma_3'])
# On-hot encoding for the clay classes


# One-hot encode the 'class 1' column
CI_encoded = pd.get_dummies(CI['class%'])

# Concatenate the original DataFrame with the one-hot encoded DataFrame
CI = pd.concat([CI, CI_encoded], axis=1)

# Convert one-hot columns to 0/1
for col in CI_encoded.columns:
    CI[col] = CI[col].astype(int)

# Combine one-hot columns into a single column
CI['class_1_encoded'] = CI[CI_encoded.columns].apply(lambda x: ''.join(x.astype(str)), axis=1)

# Drop the original one-hot encoded columns and the original class column
CI.drop(CI_encoded.columns, axis=1, inplace=True)
CI.drop('class%', axis=1, inplace=True)



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

# %%


def objective(trial, data_frame, dependent_var_columns, independent_var_columns):
    # Hyperparameters for Random Forest
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    max_samples = trial.suggest_float("max_samples", 0.1, 1.0)

    # Check and handle NaN values
    if data_frame.isnull().values.any():
        data_frame = data_frame.fillna(method='ffill').fillna(method='bfill') # Forward fill then backward fill
    # Covert Dataframe to numpy array
    data_array = data_frame.values
    #Extract dependent and independent variables using the column names
    independent_vars = data_frame[independent_var_columns].values
    dependent_vars = data_frame[dependent_var_columns].values
    #Identify encoded columns (replace with original values after scaling)
    encoded_columns = [i for i, col in enumerate(independent_var_columns) if 'encoded' in col]
    #exclude encoded columns from scaling
    independent_vars_to_scale = np.delete(independent_vars, encoded_columns, axis=1)
    dependent_vars_to_scale = dependent_vars
    #standardize the non-encoded columns
    scaler_independent = StandardScaler()
    scaler_dependent = StandardScaler()
    independent_vars_scaled = scaler_independent.fit_transform(independent_vars_to_scale)
    dependent_vars_scaled = scaler_dependent.fit_transform(dependent_vars_to_scale)

    for idx in encoded_columns:
        independent_vars_scaled = np.insert(independent_vars_scaled, idx, independent_vars[:, idx], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(independent_vars_scaled, dependent_vars_scaled, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        random_state=42,  # Add a random state for reproducibility
    )
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model (similar to before, with the same scaler)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_dependent.inverse_transform(y_pred_scaled)
    y_test_original = scaler_dependent.inverse_transform(y_test)


    y_pred_scaled = model.predict(X_test)
    if np.isnan(y_pred_scaled).any():
        raise ValueError("Model predictions contain NaN values.")
    
    y_pred = scaler_dependent.inverse_transform(y_pred_scaled)
    y_test_original = scaler_dependent.inverse_transform(y_test)

    r2_scores = [r2_score(y_test_original[:, i], y_pred[:, i]) for i in range(y_test_original.shape[1])]
    return np.mean(r2_scores), r2_scores  # Return both the mean R2 and the list of R2 scores
    

def optimize_random_forest(data_frame, dependent_var_columns, independent_var_columns, n_trials=50):
    for var_name, var_value in inspect.currentframe().f_back.f_locals.items():
        if var_value is data_frame:
            data_name = var_name
            break
    
    

    study_name = f"RandomForest_Single objective, CC+R2, {n_trials}-trials_{data_name}-USING_{','.join(independent_var_columns)}-PREDICTING_{','.join(dependent_var_columns)}"
    storage = optuna.storages.RDBStorage('sqlite:///db.sqlite3')

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage)
    else:
        study_name = "AUTOADD_" + study_name
        study = optuna.create_study(study_name=study_name, direction='maximize', storage=storage)
    
    all_r2_scores = []

    def objective_with_logging(trial):
        mean_r2, r2_scores = objective(trial, data_frame, dependent_var_columns, independent_var_columns)
        all_r2_scores.append(r2_scores)
        return mean_r2
    
    study.optimize(objective_with_logging, n_trials=n_trials)

    average_r2 = np.mean(all_r2_scores)
    print(f"Average R2 over all trials: {average_r2:.4f}")

    plot_optimization_history(study)
    plot_param_importances(study)

    print("Best hyperparameters:", study.best_params)
    return study, all_r2_scores

def plot_individual_r2_scores(r2_scores, dependent_var_columns):
    mean_r2_scores = np.mean(r2_scores, axis=0)
    plt.bar(dependent_var_columns, mean_r2_scores)
    plt.xlabel('Dependent Variable')
    plt.ylabel('Mean R2 Score')
    plt.title('Mean R2 Score for each Dependent Variable')
    plt.show()

def visualize_optuna_results(study, training_losses=None, validation_losses=None):
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_contour(study, params=["n_estimators", "max_depth"])  # Correct parameters
    plot_edf(study)
    plot_parallel_coordinate(study, params=["n_estimators", "max_depth", "max_features"])  # Correct parameters

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_contour(study, params=["n_estimators", "max_depth"])  # Correct parameters
    optuna.visualization.plot_edf(study)
    optuna.visualization.plot_parallel_coordinate(study, params=["n_estimators", "max_depth", "max_features"])  # Correct parameters

def plot_individual_r2_scores(r2_scores, dependent_var_columns):
    mean_r2_scores = np.mean(r2_scores, axis=0)
    plt.bar(dependent_var_columns, mean_r2_scores)
    plt.xlabel('Dependent Variable')
    plt.ylabel('Mean R2 Score')
    plt.title('Mean R2 Score for each Dependent Variable')
    plt.show()    
    
  
#%%
# # Update the function call for CI
#independent_var_columns = ['clay_content_100%', 'sand_content_100%', 'silt_content_100%']
#independent_var_columns = ['liquid_limit', 'plastic_limit', 'plasticity_index']
#independent_var_columns = ['class_1_encoded','clay_content_100%']
independent_var_columns = ['dry_unit_weight']
dependent_var_columns = [col for col in CC.columns if col not in independent_var_columns]

study, all_r2_scores = optimize_random_forest(CC, dependent_var_columns,
                                                                      independent_var_columns,
                                                                        n_trials=100)
visualize_optuna_results(study)
plot_individual_r2_scores(all_r2_scores, dependent_var_columns)
# %%
#independent_var_columns = ['dry unit weight kN/m3']
#dependent_var_columns = [col for col in CT.columns if col not in independent_var_columns]

# Optimize hyperparameters
#study, all_r2_scores = optimize_random_forest(CT, dependent_var_columns,
#                                                                      independent_var_columns,
#                                                                        n_trials=100)
#visualize_optuna_results(study)
#plot_individual_r2_scores(all_r2_scores, dependent_var_columns)