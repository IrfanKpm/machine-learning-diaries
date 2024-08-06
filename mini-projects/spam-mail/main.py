import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Correct URL for the raw CSV file
url = "https://raw.githubusercontent.com/IrfanKpm/Machine-Learning-Notes1/main/datasets/mail_data.csv"

# Load the dataset into a DataFrame
data = pd.read_csv(url)

# Replace NaN values with an empty string ''
data = data.fillna('')

# Rename the 'Category' column to 'label'
data = data.rename(columns={'Category': 'label'})

# Display the first 4 rows of the dataset
print(data.head(4))

# Print the shape of the DataFrame
print(data.shape)

# Map 'spam' to 0 and 'ham' to 1
data.loc[data.label == 'spam', 'label'] = 0
data.loc[data.label == 'ham', 'label'] = 1

print(data.head(4))

# Separate features (X) and target variable (Y)
X = data['Message']
Y = data['label']

# Split the data into training and testing sets with a random state of 3
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print()
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Feature extraction using TF-IDF Vectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data, transform the testing data
x_train_features = feature_extraction.fit_transform(X_train)
x_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Print first 5 examples of transformed training data
print("First 5 examples of x_train_features:\n", x_train_features.toarray()[:5])

# Train the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(x_train_features, Y_train)

# Predict on the training data
train_predictions = model.predict(x_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f'\nTraining Accuracy: {train_accuracy}')

# Predict on the testing data
test_predictions = model.predict(x_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f'Testing Accuracy: {test_accuracy}')

# Create a new message for prediction
#new_message = ["Congratulations! You've won a free lottery ticket. Click here to claim your prize."]
new_message = input("\n\n\nEnter message here > ")
new_message = [new_message]

# Transform the new message using the fitted TF-IDF vectorizer
new_message_features = feature_extraction.transform(new_message)

# Predict the category of the new message
new_message_prediction = model.predict(new_message_features)[0]

# Print the prediction result
if new_message_prediction == 0:
    print("\n\nIt's a spam message.")
else:
    print("\n\nIt's a ham message.")
