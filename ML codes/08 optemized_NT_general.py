from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Users\mosty\OneDrive\Desktop\23_5_thesis files\Machine learning phase'

# Load the data
COMNN = pd.read_excel(path + r'\COMBINED_NT.xlsx')
COMRF = pd.read_excel(path + r'\COMBINED_RF.xlsx')
COMANN = pd.read_excel(path + r'\COMBINED_Auto_encoded_NT.xlsx')
COMARF = pd.read_excel(path + r'\COMBINED_Auto_encoded_RF.xlsx')

columns_to_drop = ['salt_content', 'organic_content', 'activity', 'class%', 'class_1_encoded', 'activity_class', 'dry_unit_weight']
columns_to_dropA = ['encoded_class', 'encoded_activity']
COMNN = COMNN.drop(columns=columns_to_drop)
COMRF = COMRF.drop(columns=columns_to_drop)
COMANN = COMANN.drop(columns=columns_to_dropA)
COMARF = COMARF.drop(columns=columns_to_dropA)

def clip_and_replace(df, density_columns, unit_weight_columns, df_name, density_min=1094, density_max=1919, weight_min=10.90, weight_max=18.73):
    replaced_count = {col: 0 for col in density_columns + unit_weight_columns}

    df[density_columns] = df[density_columns].clip(lower=density_min)
    for col in density_columns:
        replaced_count[col] += (df[col] > density_max).sum()
        df[col] = df[col].apply(lambda x: density_max if x > density_max else x)
    
    df[unit_weight_columns] = df[unit_weight_columns].clip(lower=weight_min)
    for col in unit_weight_columns:
        replaced_count[col] += (df[col] > weight_max).sum()
        df[col] = df[col].apply(lambda x: weight_max if x > weight_max else x)

    print(f"Results for DataFrame '{df_name}':")
    for col, count in replaced_count.items():
        print(f"Column '{col}': {count} values replaced")

    return df

dry_density_to_clip = ['predicted_dry_density_proctor_0.60',
                       'predicted_dry_density_proctor_0.75',
                       'predicted_dry_density_proctor_0.85']

dry_unitweight_to_clip = ['predicted_dry_unit_weight_0.60',
                          'predicted_dry_unit_weight_0.75',
                          'predicted_dry_unit_weight_0.85']

COMNN = clip_and_replace(COMNN, dry_density_to_clip, dry_unitweight_to_clip, 'COMNN')
COMRF = clip_and_replace(COMRF, dry_density_to_clip, dry_unitweight_to_clip, 'COMRF')
COMANN = clip_and_replace(COMANN, dry_density_to_clip, dry_unitweight_to_clip, 'COMANN')
COMARF = clip_and_replace(COMARF, dry_density_to_clip, dry_unitweight_to_clip, 'COMARF')
#%%
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
import matplotlib.pyplot as plt
# Define the triangle vertices, areas, and lines data (previously defined)
triangle_vertices = [(0, 0), (50, 86.6), (100, 0)]

lines_data = [
    ((25, 43.30127019), (75, 43.30127019)),  # ks1
    ((17.5, 30.31089), (82.5, 30.31089)),  # ks2
    ((12.5, 21.65064), (87.5, 21.65064)),  # ks3
    ((8.75, 15.15544), (41.25, 15.15544)),  # kz1
    ((6, 10.3923), (44, 10.3923)),  # kz2
    ((4, 6.928203), (46, 6.928203)),  # kz3
    ((2.5, 4.330127), (15, 4.330127)),  # zk
    ((7.5, 4.330127), (10, 0)),  # zs1 (to the left)
    ((13.5, 6.928203), (17.5, 0)),  # zs2 (to the left)
    ((28.5, 6.928203), (32.5, 0)),  # zs3 (to the left)
    ((50, 0), (46, 6.928203)),  # zs4 (to the left)
    ((46, 6.928203), (37.5, 21.65064)),  # kz1,2,3
    ((46, 6.928203), (87.5, 21.65064)),  # ks3 to the right
    ((85, 0), (75.29, 16.97))  # lz3 (to the left) and lz1 to the right
]

areas = {
    "Ks1": Polygon([(25, 43.30127019), (75, 43.30127019), (50, 86.6)]),
    "Ks2": Polygon([(25, 43.30127019), (17.5, 30.31089), (82.5, 30.31089), (75, 43.30127019)]),
    "Ks3": Polygon([(17.5, 30.31089), (12.5, 21.65064), (87.5, 21.65064), (82.5, 30.31089)]),
    "Ks4": Polygon([(46, 6.928203), (37.5, 21.65064), (87.5, 21.65064)]),
    "Kz1": Polygon([(8.75, 15.15544), (41.25, 15.15544), (37.5, 21.65064), (12.5, 21.65064)]),
    "Kz2": Polygon([(8.75, 15.15544), (6, 10.3923), (44, 10.3923), (41.25, 15.15544)]),
    "Kz3": Polygon([(6, 10.3923), (4, 6.928203), (46, 6.928203), (44, 10.3923)]),
    "zk": Polygon([(2.5, 4.330127), (15, 4.330127), (13.5, 6.928203), (4, 6.928203)]),
    "Zs1": Polygon([(0, 0), (10, 0), (7.5, 4.330127), (2.5, 4.330127)]),
    "Zs2": Polygon([(10, 0), (17.5, 0), (15, 4.330127), (7.5, 4.330127)]),
    "Zs3": Polygon([(17.5, 0), (32.5, 0), (28.5, 6.928203), (13.5, 6.928203)]),
    "Zs4": Polygon([(32.5, 0), (50, 0), (46, 6.928203), (28.5, 6.928203)]),
    "Lz3": Polygon([(50, 0), (85, 0), (75.29, 16.97), (46, 6.928203)]),
    "Lz1": Polygon([(85, 0), (100, 0), (87.5, 21.65064), (75.29, 16.97)])
}

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
    "zk": (10, 5),
    "Zs1": (5, 2),
    "Zs2": (13, 2),
    "Zs3": (20, 2),
    "Zs4": (40, 2),
}

def classify_point(point, areas):
    """Classifies a point based on its x, y coordinates."""
    for area_name, polygon in areas.items():
        if polygon.contains(point):
            return area_name
    return 'other'

def process_dataframe(df):
    """Processes a DataFrame by calculating x, y coordinates and classifying points into areas."""
    # Calculate x and y
    df['x'] = (100 - df['sand_content_100%']) - df['clay_content_100%'] * np.cos((1/3) * np.pi)
    df['y'] = np.sin((1/3) * np.pi) * df['clay_content_100%']
    
    # Classify points
    df['classification'] = df.apply(lambda row: classify_point(Point(row['x'], row['y']), areas), axis=1)
    
    return df

def plot_triangle(df, title):
    """Plots the triangle with classified points, polygons, and labels"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the polygons FIRST (zorder=1 to ensure they are behind)
    for area_name, polygon in areas.items():
        mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True, edgecolor='blue', alpha=0.3, label=area_name, zorder=1)
        ax.add_patch(mpl_poly)

    # Plot the triangle (zorder=2 to be above polygons)
    triangle_x, triangle_y = zip(*triangle_vertices)
    ax.plot(triangle_x + (triangle_x[0],), triangle_y + (triangle_y[0],), color='black', linewidth=2, zorder=2)

    # Plot the lines within the triangle (zorder=5)
    for (x1, y1), (x2, y2) in lines_data:
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, zorder=5)

    # Filter out points in specific areas for "removed" and "selected" points
    removed_points = df[df['classification'].isin(['Lz1', 'Lz3', 'Zs3', 'Zs4'])]

    selected_points = df[~df['classification'].isin(['Lz1', 'Lz3', 'Zs3', 'Zs4'])]

    # Plot removed points (zorder=3)
    removed_scatter = ax.scatter(removed_points['x'], removed_points['y'], color='red', zorder=3, label='Removed Points')

    # Plot selected points (zorder=3)
    selected_scatter = ax.scatter(selected_points['x'], selected_points['y'], color='green', zorder=3, label='Selected Points')

    # Add labels for the different spaces in the triangle (zorder=4)
    for label, position in labels.items():
        ax.text(position[0], position[1], label, fontsize=12, ha='center', zorder=4)

    # Configure plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 90)
    ax.set_title(title)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    # Add a legend
    ax.legend(handles=[removed_scatter, selected_scatter], title='Point Types', loc='upper left')

    # Show plot
    plt.grid(True)
    plt.show()

# Example usage:
COMNN = process_dataframe(COMNN)
plot_triangle(COMNN, "COMNN Triangle Plot")

COMRF = process_dataframe(COMRF)
plot_triangle(COMRF, "COMRF Triangle Plot")

COMANN = process_dataframe(COMANN)
plot_triangle(COMANN, "COMANN Triangle Plot")

COMARF = process_dataframe(COMARF)
plot_triangle(COMARF, "COMARF Triangle Plot")

#%% encoding the Triangle classifications

# Define the function for encoding class 1 column in a dataframe
def encode_class_column(df):
    # Define the mapping of class1 values to new column values
    class1_mapping = {
        'Ks1':  '10000000',
        'Ks2':  '01000000',
        'Ks3':  '00100000',
        'Ks4':  '00010000',
        'Lz3':  '00010000',
        'Kz1':  '00001000',
        'Kz2':  '00000100',
        'Kz3':  '00000010',
        'other':'00000001'    
    }

    # Encode the 'class 1' column based on the mapping
    df['class_encoded'] = df['classification'].map(class1_mapping)
    return df

# Apply the function to the four dataframes
COMNN = encode_class_column(COMNN)
COMRF = encode_class_column(COMRF)
COMANN = encode_class_column(COMANN)
COMARF = encode_class_column(COMARF)



class_to_drop = ['classification']
COMNN = COMNN.drop(columns=class_to_drop)
COMRF = COMRF.drop(columns=class_to_drop)
COMANN = COMANN.drop(columns=class_to_drop)
COMARF = COMARF.drop(columns=class_to_drop)


#%% New classifications method
def create_classification_plot_with_new_classes(df, title="Plasticity Graph", save_filename=None):
    # Define the polygons for the new classifications
    classification_polygons = {
        "CIV": Polygon([(69.5, 37), (100, 58), (100, 80), (97,80), (69.5, 55)]),
        "SIV": Polygon([(69.5, 0), (100, 0), (100, 58), (69.5, 37)]),
        "CIH": Polygon([(49.5, 22), (70, 37), (70, 55), (49.5, 37)]),
        "SIH": Polygon([(50, 0), (70, 0), (70, 37), (50, 23)]),
        "CIM": Polygon([(34.5, 10.5), (50, 22.5), (50, 38), (34.5, 25)]),
        "SIM": Polygon([(35, 0), (50, 0), (50, 22), (35, 11)]),
        "CIL": Polygon([(33, 8), (35, 12), (35, 25), (18, 8), (33, 8)]),
        "SIL": Polygon([(25, 0), (35, 0), (35, 12), (25, 8)]),
        "CIL-SIL": Polygon([(10, 5), (25, 5), (25, 8), (10, 8)])
    }

    # Define colors for the polygons using RGB tuples
    polygon_colors = {
        "CIV": (1.0, 0.5, 0.5),  # lightcoral
        "SIV": (0.678, 0.847, 0.902),  # lightblue
        "CIH": (0.564, 0.933, 0.564),  # lightgreen
        "SIH": (0.800, 0.520, 0.800),  # lightpurple
        "CIM": (1.0, 0.647, 0.0),  # lightorange
        "SIM": (1.0, 1.0, 0.8),  # lightyellow
        "CIL": (0.88, 1.0, 1.0),  # lightcyan
        "SIL": (1.0, 0.75, 1.0),  # lightmagenta
        "CIL-SIL": (0.82, 0.41, 0.12)  # lightbrown
    }

    # Define the vertical lines and the functions for lines A and U
    vertical_lines = {
        'x': [35, 50, 70],
        'y_ranges': [(0, 25), (0, 38), (0, 55)]
    }

    def line_a(x):
        return 0.73 * (x - 20)

    def line_u(x):
        return 0.9 * (x - 8)

    # Create the base plot function
    def create_base_plot():
        plt.figure(figsize=(12, 10))

        # Plot the vertical lines
        for i, x in enumerate(vertical_lines['x']):
            y_min, y_max = vertical_lines['y_ranges'][i]
            plt.plot([x, x], [y_min, y_max], 'k-', linewidth=1.5)

        # Generate x values for line A and line U
        x_values = np.linspace(26.5, 100, 400)
        y_values_a = line_a(x_values)

        x_values_u = np.linspace(17, 80/0.9 + 8, 400)
        y_values_u = line_u(x_values_u)

        # Plot lines A and U
        plt.plot(x_values, y_values_a, 'r-', label='A-lijn: Ip = 0.73 * (wL - 20)')
        plt.plot(x_values_u, y_values_u, 'b--', label='U-lijn: Ip = 0.9 * (wL - 8)')

        # Add horizontal lines
        plt.plot([10, 31], [8, 8], 'k-', linewidth=1.5)
        plt.plot([10, 26.5], [5, 5], 'k-', linewidth=1.5)

        # Set plot labels and title
        plt.xlabel('Liquid limit, LL')
        plt.ylabel('Plasticity index, PI')
        plt.title(title)  # Use the title parameter here
        plt.grid(True)
        plt.legend()

        # Set axis limits
        plt.xlim(0, 100)
        plt.ylim(0, 80)

    # Calculate the centroid of each polygon for labeling
    def get_polygon_centroid(polygon):
        x, y = polygon.exterior.xy
        return np.mean(x), np.mean(y)

    # Plot all points together
    create_base_plot()

    # Plot the classification polygons with their respective colors and labels
    for class_name, polygon in classification_polygons.items():
        mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True, edgecolor='black', facecolor=polygon_colors.get(class_name, (0.5, 0.5, 0.5)), alpha=0.5, label=class_name)
        plt.gca().add_patch(mpl_poly)
        centroid_x, centroid_y = get_polygon_centroid(polygon)
        plt.text(centroid_x, centroid_y, class_name, fontsize=12, ha='center', va='center', color='blue', weight='bold')

    # Assign points to the polygon they belong to (new classifications)
    def classify_point_new_class(point, polygons):
        for class_name, polygon in polygons.items():
            if polygon.contains(point):
                return class_name
        return 'Other'

    # Apply the classification to the dataframe
    df['new_classification'] = df.apply(lambda row: classify_point_new_class(Point(row['liquid_limit'], row['plasticity_index']), classification_polygons), axis=1)

    # Custom colors for scatter plot (based on new classification)
    class_colors = {
        "CIV": (1.0, 0.0, 0.0),  # red
        "SIV": (0.0, 0.0, 1.0),  # blue
        "CIH": (0.0, 1.0, 0.0),  # green
        "SIH": (0.5, 0.0, 0.5),  # purple
        "CIM": (1.0, 0.5, 0.0),  # orange
        "SIM": (1.0, 1.0, 0.0),  # yellow
        "CIL": (0.0, 1.0, 1.0),  # cyan
        "SIL": (1.0, 0.0, 1.0),  # magenta
        "CIL-SIL": (0.6, 0.3, 0.1),  # brown
        "Other": (0.0, 0.0, 0.0)  # black
    }

    # Plot points colored by new classification
    for class_name, color in class_colors.items():
        class_data = df[df['new_classification'] == class_name]
        plt.scatter(class_data['liquid_limit'], class_data['plasticity_index'], color=color, edgecolor='k', s=100, label=class_name)

    # Save the plot with a filename based on the title parameter
    if save_filename is None:
        save_filename = f"{title.replace(' ', '_')}_classification_new"
        
    plt.legend(title="Classifications")
    plt.savefig(f"{save_filename}.png")
    plt.show(block=False)
    
    # Return the dataframe with the new classification column
    return df

# Example Usage:
COMNN = create_classification_plot_with_new_classes(COMNN, title="Plasticity gragh COMNN")
COMRF = create_classification_plot_with_new_classes(COMRF, title="Plasticity gragh COMRF")
COMANN = create_classification_plot_with_new_classes(COMANN, title="Plasticity gragh COMANN")
COMARF = create_classification_plot_with_new_classes(COMARF, title="Plasticity gragh COMARF")
#%%

# Define the function for encoding class 1 column in a dataframe
def new_encode_class_column(df):
    # Define the mapping of class1 values to new column values
    class1_mapping = {
        'CIV':    '1000000000',
        'SIV':    '0100000000',
        'CIH':    '0010000000',
        'SIH':    '0001000000',
        'CIM':    '0000100000',
        'SIM':    '0000010000',
        'CIL':    '0000001000',
        'SIL':    '0000000100',
        'CIL-SIL':'0000000010',
        'Other' : '0000000001'
    }

    # Encode the 'class 1' column based on the mapping
    df['new_class_encoded'] = df['new_classification'].map(class1_mapping)
    return df

# Apply the function to the four dataframes
COMNN = new_encode_class_column(COMNN)
COMRF = new_encode_class_column(COMRF)
COMANN = new_encode_class_column(COMANN)
COMARF = new_encode_class_column(COMARF)



new_class_to_drop = ['new_classification']
COMNN = COMNN.drop(columns=new_class_to_drop)
COMRF = COMRF.drop(columns=new_class_to_drop)
COMANN = COMANN.drop(columns=new_class_to_drop)
COMARF = COMARF.drop(columns=new_class_to_drop)

#%% dropping x and y
COMNN = COMNN.drop(columns=['x','y'])
COMRF = COMRF.drop(columns=['x','y'])
COMANN = COMANN.drop(columns=['x','y'])
COMARF = COMARF.drop(columns=['x','y'])

#%% calculatin unit weight before predictions

def calculate_unit_weights(df):
    
    # Calculate unit weights for each scenario
    df['unit_weight_0.60'] = df['predicted_dry_unit_weight_0.60'] * (1 + df['water_content_0.60']/100)
    df['unit_weight_0.75'] = df['predicted_dry_unit_weight_0.75'] * (1 + df['water_content_0.75']/100)
    df['unit_weight_0.85'] = df['predicted_dry_unit_weight_0.85'] * (1 + df['water_content_0.85']/100)
    
    return df

# Assuming COMNN, COMRF, COMANN, COMARF are your DataFrames
COMNN = calculate_unit_weights(COMNN)
COMRF = calculate_unit_weights(COMRF)
COMANN = calculate_unit_weights(COMANN)
COMARF = calculate_unit_weights(COMARF)


#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def train_neural_network(X, y, df_name, scenario_num, num_folds=5):
    if X.empty or y.empty:
        raise ValueError(f"Independent or dependent variables are empty for {df_name}. Check the DataFrame and columns.")
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_fold_index = -1
    best_mean_r2 = -float('inf')
    best_model = None
    best_scaler_X = None
    best_scaler_y = None
    best_y_pred = None
    best_y_test_original = None
    best_r2_scores = {}

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Model architecture
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
        
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
        #the following is to modify the network for the categorised data
# =============================================================================
#         for _ in range(8):
#             model.add(layers.Dense(250, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
#             model.add(layers.Dropout(0.55))
#             model.add(layers.BatchNormalization())
#         
#         # Last layer - using sigmoid activation for multi-label classification
#         model.add(layers.Dense(y_train_scaled.shape[1], activation='sigmoid'))
#         
#         # Compile model
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy')
#         
#         # Callbacks
#         lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
#         
# =============================================================================
        # Train model
        history = model.fit(X_train_scaled, y_train_scaled, epochs=1200, batch_size=120,
                            validation_data=(X_test_scaled, scaler_y.transform(y_test)),
                            verbose=0, callbacks=[early_stopping, lr_scheduler])
        
        # Predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = y_test.values
        
        # Calculate R2 scores
        r2_scores = {y.columns[i]: r2_score(y_test_original[:, i], y_pred[:, i]) for i in range(y_test_original.shape[1])}
        mean_r2 = np.mean(list(r2_scores.values()))
        
        # Save the best fold
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_fold_index = fold + 1
            best_model = model
            best_scaler_X = scaler_X
            best_scaler_y = scaler_y
            best_y_pred = y_pred
            best_y_test_original = y_test_original
            best_r2_scores = r2_scores

    # Save metrics
    mae_scores = [mean_absolute_error(best_y_test_original[:, i], best_y_pred[:, i]) for i in range(best_y_test_original.shape[1])]
    mse_scores = [mean_squared_error(best_y_test_original[:, i], best_y_pred[:, i]) for i in range(best_y_test_original.shape[1])]

    results = {
        'Fold': best_fold_index,
        'Mean R2': best_mean_r2,
        'MAE': mae_scores,
        'MSE': mse_scores
    }
    results_df = pd.DataFrame(results, index=y.columns)
    
    r2_df = pd.DataFrame.from_dict(best_r2_scores, orient='index', columns=['R2 Score'])

    # Combine original y_test data and predictions into a DataFrame for saving
    combined_df = pd.DataFrame(best_y_test_original, columns=[f"Actual_{col}" for col in y.columns])
    combined_df = combined_df.join(pd.DataFrame(best_y_pred, columns=[f"Predicted_{col}" for col in y.columns]))

    # Plot training/validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {df_name} - Scenario {scenario_num} (Best Fold)')
    plt.legend()
    plt.show()

    # Save the best model
    model_save_path = f'{df_name}_scenario_{scenario_num}_best_model.h5'
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")

    return best_model, best_scaler_X, best_scaler_y, results_df, r2_df, combined_df

# Function to apply scenarios to multiple dataframes
def run_scenarios_on_dataframes(dataframes, independent_var_scenarios):
    all_results = {}
    
    # Iterate over each DataFrame
    for df_name, df in dataframes.items():
        
        # Iterate over each scenario
        for i, scenario in enumerate(independent_var_scenarios):
            # Define the dependent variables by excluding the current scenario's columns
            dependent_var_columns = [col for col in df.columns if col not in scenario]
            
            if len(dependent_var_columns) == 0:
                raise ValueError(f"No dependent variables identified in {df_name} for scenario {scenario}. Please check column names and scenarios.")
            
            print(f"Processing {df_name} with scenario {scenario}...")

            # Define independent and dependent variables for the scenario
            X = df[scenario]
            y = df[dependent_var_columns]
            
            # Train the neural network
            model, scaler_X, scaler_y, results_df, r2_df, combined_df = train_neural_network(X, y, df_name, i+1)
            
            # Save results for this scenario in the dictionary
            all_results[f'{df_name}_scenario{i+1}_results'] = results_df
            all_results[f'{df_name}_scenario{i+1}_r2_scores'] = r2_df
            all_results[f'{df_name}_scenario{i+1}_predictions'] = combined_df

    # Return all results (to be written into Excel or further processed)
    return all_results

# Example DataFrames
dataframes = {
    'COMNN': COMNN,
    'COMRF': COMRF,
    'COMANN': COMANN,
    'COMARF': COMARF
}

# Example independent variable scenarios
independent_var_scenarios = [
    ['liquid_limit', 'plastic_limit'],
    ['clay_content_100%', 'sand_content_100%', 'silt_content_100%'],
    ['class_encoded'],
    ['new_class_encoded']
]

# Run the scenarios and train the models
results = run_scenarios_on_dataframes(dataframes, independent_var_scenarios)

# Save the results to Excel
with pd.ExcelWriter('results_NN.xlsx') as writer:
    for sheet_name, df in results.items():
        df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel sheet names must be <= 31 characters
print("Excel file 'results_NN.xlsx' has been saved.")


