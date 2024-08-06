import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# URL of the CSV file in the GitHub repository
url = 'https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv'

# Load the dataset into a DataFrame
data = pd.read_csv(url)

# Display the shape and value counts of the dataset
print(data.shape)
print(data['Outcome'].value_counts())

# 0 -> Non-diabetics
# 1 -> Diabetics

# Separate features (X) and target variable (Y)
X = data.drop(columns='Outcome')  # All columns except 'Outcome'
Y = data['Outcome']  # The 'Outcome' column

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Create and train an SVM classifier
clf = svm.SVC()
clf.fit(X_train, Y_train)

# Make predictions on the training and test sets
Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"Accuracy on training data: {train_accuracy:.2f}")
print(f"Accuracy on test data: {test_accuracy:.2f}")


# Input data for prediction
x_ = (12,120,63,22,108,2804,0.3,42)

# Convert to NumPy array and reshape
x_array = np.array(x_)
x_array = x_array.reshape(1, -1)

# Standardize the input data
x_scaled = scaler.transform(x_array)

# Make prediction
prediction = clf.predict(x_scaled)

print(f"Prediction for the input data: {prediction[0]}")