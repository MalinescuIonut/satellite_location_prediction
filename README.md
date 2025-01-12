# Satellite Trajectory Estimation Using Machine Learning: A Comparative Analysis of Predicted and Actual Orbits

Keywords: Satellite Trajectory Prediction, Machine Learning in Space Applications, AI-Based Orbit Modeling, Geospatial Data Processing, Telecommunications Satellites, Supervised Learning, Random Forest Regressor
 

## I.	Introduction

	The accurate prediction of satellite trajectories is of utmost importance in modern aerospace operations, satellite network management, and space exploration. Trajectory forecasting ensures collision avoidance, optimizes fuel usage, and facilitates satellite-based communication and Earth observation missions. As the number of active satellites in orbit is exponentially growing, applications providing efficient and reliable prediction methods are increasingly essential.

	On the other hand, Artificial Intelligence (AI) and machine learning are some of the biggest technological breakthroughs in mankind’s recent history. Their unceasing evolution uncovers capabilities that were never thought to be possible. It is in the nature of these technologies to be fluid and that is precisely why they can be adapted, regarding any tasks they are used in. By leveraging historical and simulated satellite data, machine learning models can capture intricate nonlinear patterns and predict positions and velocities with high precision.

	This paper presents a machine learning approach for satellite trajectory prediction using a Random Forest Regressor. The primary objective of this paper refers to accurately forecasting satellite positions and velocities based on simulation data. This paper will serve as a proof of concept (POC) of the effectiveness in predicting a probable trajectory of space objects (satellites, space debris) provided the existence of historical data regarding these objects.
	
## II. Theoretical Fundamentals

	Machine learning is a subdivision of AI (artificial intelligence) that refers to the computer science field which studies the effect experience has on a data processing system. Machine learning provides the user with an approach for solving problems, without the need for the user to fully understand the inner connections of the algorithm, which may be helpful for when it may be too difficult to describe the way of transforming a certain input, to a desired output. For this to be possible, the algorithm is tuned by some variables called Parameters, the value of which is chosen by the model [1].

## A. The Machine Learning Process

	The process of devising a successful ML model can be described in five steps: data gathering, data pre-processing, model training, data testing and further improvements. 

	1. Data Gathering – the quality of the final model is a direct consequence of the amount of input data; this data is aggregated inside objects called feature vectors. 
	2. Data Pre-Processing – before feeding the gathered feature vector to the model some things should be taken into consideration such as filtering unnecessary data, which has no use in the training of the model (filtering noise, filling missing spots etc); normalization of the input data is also especially important for feature vectors containing multiple types of metrics, which might be represented on different scales. This is because ML algorithms can achieve better performance when working with features of similar scales, thus improving the training speed and accuracy. 
	3. Model Training – as a first step, a model should be chosen to fit the type of data gathered in the feature vector. There are numerous types of models designed for specific tasks, which makes choosing one a critical assignment, for example Linear Regression Models are good for predicting continuous outcomes or Neural Networks work best with large amounts of data, etc. After choosing a model, a part of the input data will be fed into it, as previously said. 
	4. Model Testing – using the rest of the input data, which has not been used for the training of the model, tests are performed to determine its accuracy. Using data unknown to the trained model is important to avoid bias during testing. 
	5. Improving the Model – this can be achieved by increasing the dataset or by tuning the models’ parameters [2].
 
![image](https://github.com/user-attachments/assets/db8dd2a5-1a6d-40b9-9252-222736536b6c)

## B. The Utilized Dataset

	The dataset utilized for the training of the ML model was procured from kaggle.com and is called: “satellite position data IDAO 2020” [4]. This dataset contains synthetic data simulating the positions and velocities of 600 satellites in Earth's orbit. It includes: a training dataset (jan_train.csv) including the following: features simulated positions (x_sim, y_sim, z_sim) and velocities (Vx_sim, Vy_sim, Vz_sim) of satellites, along with their corresponding true positions (x, y, z) and velocities (Vx, Vy, Vz). This dataset serves as the primary data for training machine learning models. The test dataset (jan_test.csv) includes simulated positions and velocities without the true target values. Models trained on the training set predict these values. Answer Key (answer_key.csv): Provides the true positions and velocities for the test dataset, enabling the evaluation of model performance. 

	To track individual satellites, each space object has one distinct ID marker and a SAT ID marker. Also included are the answer keys enabling performance evaluation after the training of the model. This resource particularly suits regression analysis, with applications in enhancing orbital path prediction and satellite collision prevention systems.

## C. The Training of the Model – Random Forest Regressor 

	For predicting the continuous values of satellite positions and velocities, a Random Forest Regressor was employed. A regressor is a type of machine learning model designed to predict continuous numerical values based on the provided input features. Unlike classifiers, which predict discrete categories, regressors are designed to estimate real-valued outputs, which in the case of this paper refers to the position of satellites. Regressors are commonly used in tasks involving prediction of numerical data trends, patterns, or relationships [5].

	The Random Forest algorithm builds an ensemble of decision trees, each trained on a randomly sampled subset of the training data using the technique called bootstrap sampling. To further enhance diversity among the trees, the model considers only a random subset of features at each split within a tree. In this setup, the Random Forest Regressor predicts the satellite trajectories by averaging the outputs of all individual decision trees, ensuring stability and reducing overfitting. Hyperparameter tuning (number of trees and tree depth) further optimized the model's performance, resulting in high predictive accuracy for the test dataset [6].

# III. Proposed Method

	The proposed method for this project involves using a machine learning approach with Random Forest Regression to predict satellite trajectories based on telemetry data. The model is trained on satellite position and velocity features. The dataset is split into training, validation, and test sets, and performance is evaluated using multiple metrics such as MAE, MSE, RMSE, and R². The predicted trajectories are compared to actual data through 3D visualizations and key point analysis, including perigee and apogee. Having presented the methodology of work, the next step will be to present the actual implementation of the actual Satellite Trajectory Estimation model.




# IV. Implementation

	The implementation of the actual trajectory estimation application consists of five main components:

	1.Data Preprocessing : 
	
	The first step in the implementation is data preprocessing, which prepares the raw dataset for model training and evaluation. This involves converting timestamps ("epochs") into human-readable (HR) datetime formats using the “pd.to_datetime” function. Following this, the code performs feature engineering, where new features are created from the existing date, in this case, the distance and velocity of the satellite are calculated using the satellite's position and velocity components. The distance is derived by calculating the Euclidean distance from the origin, and the velocity is computed by taking the square root of the squared velocity components in the x, y, and z directions. These additional features are important because they provide more context for the model to understand the dynamics of the satellite’s movement. 

# Calculate distance and velocity
train['distance'] = np.sqrt(train['x_sim']**2 + train['y_sim']**2 + train['z_sim']**2)
train['velocity'] = np.sqrt(train['Vx_sim']**2 + train['Vy_sim']**2 + train['Vz_sim']**2)
test['distance'] = np.sqrt(test['x_sim']**2 + test['y_sim']**2 + test['z_sim']**2)
test['velocity'] = np.sqrt(test['Vx_sim']**2 + test['Vy_sim']**2 + test['Vz_sim']**2)	

	The data is then scaled using the MinMaxScaler from scikit-learn. Scaling the features is crucial for the performance of many machine learning models. 

	2. Model Definition
	
	After processing the data, the next step is to define and train the ML model. For this, a Random Forest Regressor is used. The code implements the RandomForestRegressor() class from scikit-learn. The model is trained using the training data (X_train, Y_train), where X_train contains features such as: position, velocity and distance from the earth and Y_train represents the target variables (the satellite's actual positions in x, y, and z coordinates). By training the model on this data, the Random Forest learns the relationship between the satellite's features and its trajectory.

# 1.4 FEATURE SELECTION 
X = train[['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim', 'distance', 'velocity']] 
Y = train[['x', 'y', 'z']]
	
	The model, as mentioned before, is defined using the RandomForestRegressor() class from scikit-learn, which creates an ensemble of decision trees. Each tree in the forest is trained on a random subset of the data, and the final prediction is obtained by averaging the predictions from all individual trees. This approach reduces the variance and overfitting that might occur with a single decision tree. The key parameters utilized were:
•	n_estimators (default = 100) - number of trees in the forest. 
•	max_depth: - depth of each tree, which is set to none, meaning that each tree can grow until reaching pure leaves (a leaf is pure when all the samples in that leaf have the same (or nearly the same) output value)

3. Model Evaluation

	After training the model, its performance is evaluated using appropriate metrics which is done on a validation set (X_val, Y_val) that the model has not seen during training. This uncertainty helps assess how well the model generalizes to new, unseen data. The evaluation of the model is done utilizing the following metrics:
•	Mean Absolute Error (MAE) - measures the average magnitude of the errors in the predictions, without considering their direction. It provides a measure of how far off the model’s predictions are from the actual values. 
•	Mean Squared Error (MSE) is a metric that squares the errors, which penalizes larger errors more heavily. This can help identify if the model is producing large deviations in any particular predictions. 
•	Root Mean Squared Error (RMSE) is the square root of MSE, providing the error in the same units as the output (satellite position), meaning that it returns the error in meters.
•	R-squared (R²) is a statistical measure that explains the proportion of variance in the target variable that is explained by the model. A value closer to 1 indicates a better fit. 


4. Model Prediction

	After being trained and evaluated, the model is used to make predictions on the test dataset. The test data is preprocessed similarly to the training data (including scaling) to ensure consistency. The model then predicts the satellite's positions in space, which are compared to the actual positions provided in the answer key. 
	
	To visualize these predictions, the code generates 3D plots that compare the actual and predicted trajectories. The 3D plots help in visually assessing how closely the predicted trajectory matches the actual trajectory in three-dimensional space. The actual trajectories are plotted in continuous lines, while the predicted trajectories are shown in dashed  lines to highlight the difference. These visualizations serve as a valuable tool to intuitively understand the model's performance, particularly for tasks that involve spatial data like satellite positioning.

	For a more in-deep analysis, the model also generates individual visualizations for specific satellites. By selecting specific satellite IDs, the code produces 3D plots showing the trajectory of each satellite, both actual and predicted. This plot helps in understanding how well the model performs for different cases, namely satellites, as different space objects may have different motion patterns due to orbital mechanics. In addition, the model visualizes the satellite’s trajectory around Earth. 

5. Saving Results

	After visualization, the model is designed to save the predicted and actual trajectory data to a .CSV file (satellite_trajectory_data.csv). This file contains the x, y, and z coordinates for both the actual and predicted positions of the satellites, allowing for further analysis and reporting. These files will be further utilized within a MATLAB script for visualization purposes while also marking key points such as perigee, apogee, and start/end positions.

