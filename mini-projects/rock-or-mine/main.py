import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Correct raw URL of the CSV file
dataset_url = "https://raw.githubusercontent.com/MoraisMNS/Rock-vs-mine-Prediction/master/sonar_data.csv"

# Load the dataset into a DataFrame
sonar_data = pd.read_csv(dataset_url)

# Display the first 3 rows and shape of the dataset
print(sonar_data.head(3))
print(sonar_data.shape)

# Separate features and target variable
# Assuming the target variable is the last column
X = sonar_data.iloc[:, :-1]  # All columns except the last one
Y = sonar_data.iloc[:, -1]   # The last column

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Create a logistic regression model with a custom max_iter
model = LogisticRegression(max_iter=2000)
model.fit(X_train, Y_train)

# Make predictions on the training and test sets
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate accuracy scores
train_accuracy_score = accuracy_score(Y_train, train_predictions)
test_accuracy_score = accuracy_score(Y_test, test_predictions)

print(f"Accuracy on training data: {train_accuracy_score}")
print(f"Accuracy on test data: {test_accuracy_score}")

x_pr = (0.0095,0.0308,0.0539,0.0411,0.0613,0.1039,0.1016,0.1394,0.2592,0.3745,0.4229,0.4499,0.5404,0.4303,0.3333,0.3496,0.3426,0.2851,0.4062,0.6833,0.7650,0.6670,0.5703,0.5995,0.6484,0.8614,0.9819,0.9380,0.8435,0.6074,0.5403,0.6890,0.5977,0.3244,0.0516,0.3157,0.3590,0.3881,0.5716,0.4314,0.3051,0.4393,0.4302,0.4831,0.5084,0.1952,0.1539,0.2037,0.1054,0.0251,0.0357,0.0181,0.0019,0.0102,0.0133,0.0040,0.0042,0.0030,0.0031,0.0033)
# Convert x_pr to DataFrame with column names
x_pr_df = pd.DataFrame([x_pr], columns=X.columns)
y_pred = model.predict(x_pr_df)

print(f"Prediction for the new sample: {y_pred[0]}")