# Bike Sharing Demand Prediction

## Overview

This project aims to predict the demand for bike sharing using machine learning. The dataset used contains features such as date, time, weather conditions, and the number of bike rentals. A regression model is built to predict the 'count' of total rental bikes including both casual and registered.

## Code Structure

1. **Data Loading and Preprocessing:**
   - The code begins by importing necessary libraries like pandas, NumPy, TensorFlow, Matplotlib, and scikit-learn.
   - The bike sharing dataset ('train.csv') is loaded using pandas.
   - Missing values are handled, and the 'datetime' column is dropped.
   - Features (X) and the target variable (y - 'count') are separated.
   - Data scaling is performed using RobustScaler to handle outliers.
   - The data is split into training and testing sets using `train_test_split`.

2. **Model Building:**
   - A sequential model is created using Keras with the following layers:
     - Input layer with the shape based on the number of features.
     - Dense layers with ReLU activation and L1/L2 regularization.
     - Batch Normalization layers for improved training stability.
     - Dropout layers to prevent overfitting.
     - Output layer with a single neuron for regression.

3. **Model Compilation and Training:**
   - The model is compiled using the Adam optimizer and mean squared error loss function.
   - Early stopping is implemented to prevent overfitting and save the best model weights.
   - The model is trained on the training data with specified epochs and batch size.

4. **Model Evaluation:**
   - Predictions are made on the test data.
   - Evaluation metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared are calculated to assess model performance.

## Logic and Algorithms

- The project uses a supervised learning approach for regression.
- A neural network is built to learn the relationship between features and bike sharing demand.
- The model learns by minimizing the difference between predicted and actual bike rental counts.
- Regularization techniques and dropout are employed to prevent overfitting and improve generalization.
- Early stopping ensures that training stops when the model's performance on validation data plateaus.

## Technologies

- Python
- Pandas
- NumPy
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Seaborn
  
![image](https://github.com/user-attachments/assets/e9b8232a-89d0-42f9-959c-605c3eb72389)

## How to Run

1. Make sure you have the necessary libraries installed. You can install them using `pip install pandas numpy tensorflow scikit-learn matplotlib seaborn`.
2. Upload the 'train.csv' dataset to your Colab environment.
3. Run the code cells sequentially.
4. Observe the evaluation metrics to assess the model's performance.
