Project Report: AnomaData - Automated Anomaly Detection for Predictive Maintenance
1. Problem Statement
Many industries face the challenge of system failures in their equipment. This project addresses the need for predictive maintenance solutions by leveraging data analytics to identify anomalies, providing early indications of potential failures.
2. Data Overview
Dataset: Time series data with 18000+ rows collected over several days.
Target Variable: Binary labels in column 'y,' where 1 denotes an anomaly.
Predictors: Other columns containing features for analysis.
3.Tasks/Activities List
	Collect the time series data from the CSV file linked here.
	Exploratory Data Analysis (EDA) - Show the Data quality check, treat the missing values, outliers etc. if any.
	Get the correct data type for date.
	Feature Engineering and feature selection.
	Train/Test Split - Apply a sampling distribution to find the best split
	Choose the metrics for the model evaluation 
	Model Selection, Training, Predicting and Assessment
	Hyper parameter Tuning/Model Improvement
	Model deployment plan.











3.1 Import Libraries and Load Dataset-It Consist of libraries installation & Data fetching
Details are as follow:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv("C:/Users/admin/Downloads/data.csv")




3.2 Exploratory Data Analysis (EDA)- It Consist of data quality checks, handling of missing values, and explored data distribution.

# Exploratory Data Analysis (EDA)
# Show Data quality check, treat missing values, outliers, etc.
print(data.info())
print(data.describe())


3.3 Data Cleaning- It Consist of managing the missing values and standardize the data

# Drop missing values
data = data.dropna()





3.4 Feature Engineering- It consist of data type changing and extraction of addition features
# For simplicity, we'll just drop the 'date' column for now
data = data.drop('date', axis=1)
# Convert date column to datetime type
Anoma_data ['date'] = pd.to_datetime(Anoma_data ['date'])


3.5 Train/Test Split- It consist of application of sampling distribution to find the best split

# Train/Test Split
X = data.drop('y', axis=1)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3.6 Model Selection, Training, and Assessment- It Consist of evaluated model performance on unseen data
model = RandomForestClassifier(random_state=42)

3.7 Hyperparameter Tuning- It Consist of Hyper parameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

3.8 Model Validation

# Model Training
best_model.fit(X_train, y_train)


# Model Prediction
y_pred = best_model.predict(X_test)
# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))


3.9 Model Deployment Plan


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', best_model)
])

pipeline.fit(X_train, y_train)


# Save the model and pipeline
from joblib import dump
dump(pipeline, 'anomaly_detection_model.joblib')
