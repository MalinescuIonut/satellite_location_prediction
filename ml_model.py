# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.ensemble import RandomForestRegressor  # ML model
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For model evaluation
import joblib  # Import joblib for saving the model


# Function to create and save 3D plots for individual satellite trajectories
def plot_individual_trajectories(test_data, predictions, actual_data, sat_ids):
    for sat_id in sat_ids:
        # Filter data for the specific satellite
        sat_test_data = test_data[test_data['sat_id'] == sat_id]
        sat_actual_data = actual_data[test_data['sat_id'] == sat_id]
        sat_predictions = predictions[test_data['sat_id'] == sat_id]

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot actual trajectory in blue
        ax.plot(sat_actual_data[:, 0], sat_actual_data[:, 1], sat_actual_data[:, 2],
                label=f'Actual Trajectory (sat_id: {sat_id})', color='blue', linewidth=2)

        # Plot predicted trajectory in red (dashed)
        ax.plot(sat_predictions[:, 0], sat_predictions[:, 1], sat_predictions[:, 2],
                label=f'Predicted Trajectory (sat_id: {sat_id})', color='red', linestyle='--')

        # Configure plot appearance
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'Satellite Trajectory (sat_id: {sat_id})')
        ax.legend()

        # Save and display the plot
        plt.savefig(f'satellite_trajectory_sat_id_{sat_id}.png')
        plt.show()

# Function to create 3D visualization of satellite trajectory around Earth
def render_trajectory_around_earth(sat_actual_data, sat_predictions, sat_id):
    # Initialize 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create sphere representing Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))  # Earth's radius in km
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)

    # Plot actual trajectory in green
    ax.plot(sat_actual_data[:, 0], sat_actual_data[:, 1], sat_actual_data[:, 2],
            label=f'Actual Trajectory (sat_id: {sat_id})', color='green', linewidth=2)

    # Plot predicted trajectory in red (dashed)
    ax.plot(sat_predictions[:, 0], sat_predictions[:, 1], sat_predictions[:, 2],
            label=f'Predicted Trajectory (sat_id: {sat_id})', color='red', linestyle='--')

    # Configure plot appearance
    ax.set_xlabel('X Coordinate (km)')
    ax.set_ylabel('Y Coordinate (km)')
    ax.set_zlabel('Z Coordinate (km)')
    ax.set_title(f'Satellite Trajectory Around Earth (sat_id: {sat_id})')
    ax.legend()

    # Save and display the plot
    plt.savefig(f'satellite_around_earth_sat_id_{sat_id}.png')
    plt.show()

# Function to save trajectory data to CSV file
def save_trajectory_data_as_csv(test_data, predictions, actual_data, sat_ids, filename="satellite_trajectory_data.csv"):
    output_data = []

    # Iterate through each satellite
    for sat_id in sat_ids:
        # Filter data for the specific satellite
        sat_test_data = test_data[test_data['sat_id'] == sat_id]
        sat_actual_data = actual_data[test_data['sat_id'] == sat_id]
        sat_predictions = predictions[test_data['sat_id'] == sat_id]

        # Combine actual and predicted coordinates
        for i in range(len(sat_actual_data)):
            output_data.append({
                "sat_id": sat_id,
                "x_actual": sat_actual_data[i, 0],
                "y_actual": sat_actual_data[i, 1],
                "z_actual": sat_actual_data[i, 2],
                "x_predicted": sat_predictions[i, 0],
                "y_predicted": sat_predictions[i, 1],
                "z_predicted": sat_predictions[i, 2]
            })

    # Convert to DataFrame and save
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# 1. DATA PREPROCESSING
# Function to preprocess data by converting epoch to datetime
def preprocess_data(df):
    df['epoch'] = pd.to_datetime(df['epoch'], errors='coerce')
    return df

# Function to scale features using MinMaxScaler
def scale_features(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    return scaler.transform(X)

# 2. MODEL DEFINITION
# Function to initialize and train a Random Forest model
def train_model(X_train, Y_train, params=None):
    if params:
        model = RandomForestRegressor(**params)
    else:
        model = RandomForestRegressor(n_estimators=100)  # Default 100 trees
    model.fit(X_train, Y_train)
    return model

# 3. MODEL EVALUATION
# Function to calculate various model evaluation metrics
def evaluate_model(model, X, Y_true):
    predictions = model.predict(X)
    mae = mean_absolute_error(Y_true, predictions)  # Mean Absolute Error
    mse = mean_squared_error(Y_true, predictions)   # Mean Squared Error
    rmse = np.sqrt(mse)                            # Root Mean Squared Error
    r2 = r2_score(Y_true, predictions)             # R-squared score
    return mae, mse, rmse, r2, predictions

# 1.1 LOAD DATA
# Load input datasets
train = pd.read_csv('jan_train.csv')
test = pd.read_csv('jan_test.csv')
key = pd.read_csv('answer_key.csv')

# 1.2 PREPROCESS DATA
# Preprocess training and test data
train = preprocess_data(train)
test = preprocess_data(test)

# 1.3 FEATURE ENGINEERING
# Calculate distance and velocity
train['distance'] = np.sqrt(train['x_sim']**2 + train['y_sim']**2 + train['z_sim']**2)
train['velocity'] = np.sqrt(train['Vx_sim']**2 + train['Vy_sim']**2 + train['Vz_sim']**2)

test['distance'] = np.sqrt(test['x_sim']**2 + test['y_sim']**2 + test['z_sim']**2)
test['velocity'] = np.sqrt(test['Vx_sim']**2 + test['Vy_sim']**2 + test['Vz_sim']**2)

# 1.4 FEATURE SELECTION
# Define features and target variables
X = train[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance', 'velocity']]
Y = train[['x', 'y', 'z']]

# 1.5 FEATURE SCALING
# Scale features using MinMaxScaler
X_scaled, scaler = scale_features(X)

# 1.6 DATA SPLITTING
# Split data into training, validation, and test sets
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5, random_state=42)

# 2.1 MODEL TRAINING
# Train Random Forest model
regressor = train_model(X_train, Y_train)

# Save the trained model
joblib.dump(regressor, 'random_forest_model.pkl')

# 3.1 MODEL VALIDATION
# Evaluate model on validation set
mae, mse, rmse, r2, _ = evaluate_model(regressor, X_val, Y_val)
print(f"Validation Metrics:\nMAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# 4. MODEL PREDICTION
# Prepare test data for predictions
test_features = test[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance', 'velocity']]
test_scaled = scale_features(test_features, scaler=scaler)

# Generate predictions for test data
test_predictions = regressor.predict(test_scaled)

# Get actual trajectory from answer key
actual_trajectory = key[['x', 'y', 'z']].values

# 5. VISUALIZATION AND RESULTS
# Create and save 3D plot comparing actual vs predicted trajectories
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot actual trajectory
ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2],
        label='Actual Trajectory', color='blue', linewidth=2)

# Plot predicted trajectory
ax.plot(test_predictions[:, 0], test_predictions[:, 1], test_predictions[:, 2],
        label='Predicted Trajectory', color='red', linestyle='--')

# Configure plot appearance
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Satellite Trajectories: Actual vs Predicted')
ax.legend()

# Save and display the plot
plt.savefig('satellite_trajectory_comparison.png')
plt.show()

# Create scatter plots comparing actual vs predicted values for each coordinate
axes = ['x', 'y', 'z']
for i, axis in enumerate(axes):
    plt.figure(figsize=(10, 6))
    # Plot scatter points
    plt.scatter(actual_trajectory[:, i], test_predictions[:, i], alpha=0.5, color='green')
    # Plot perfect prediction line
    plt.plot([actual_trajectory[:, i].min(), actual_trajectory[:, i].max()],
             [actual_trajectory[:, i].min(), actual_trajectory[:, i].max()],
             color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel(f'Actual {axis.upper()}')
    plt.ylabel(f'Predicted {axis.upper()}')
    plt.title(f'Actual vs Predicted for {axis.upper()} Coordinate')
    plt.legend()
    plt.grid()
    plt.savefig(f'satellite_{axis}_comparison.png')
    plt.show()

# Example usage with specific satellite IDs
sat_ids = [1, 33, 51]  # Example satellite IDs

# Prepare data for visualization
test_data_with_sat_ids = test[['sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance', 'velocity']]
actual_data_with_sat_ids = key[['x', 'y', 'z']].values

# Generate visualizations for each satellite
for sat_id in sat_ids:
    plot_individual_trajectories(test_data_with_sat_ids, test_predictions, actual_data_with_sat_ids, [sat_id])
    render_trajectory_around_earth(actual_data_with_sat_ids, test_predictions, sat_id)

# Save trajectory data to CSV - satellite_trajectory_data.csv
save_trajectory_data_as_csv(test_data_with_sat_ids, test_predictions, actual_data_with_sat_ids, sat_ids)
