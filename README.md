# Machine-Learning-Notes-1

This repository contains my notes and code examples from the book **"Introduction to Machine Learning with Python"** by Andreas C. Müller and Sarah Guido.

![book-cover](https://m.media-amazon.com/images/I/51wF9ONArKL.jpg)

---

# Introduction to Machine Learning and Deep Learning Terms

## Machine Learning Types

Machine learning is a field of artificial intelligence that involves training algorithms to make predictions or decisions based on data. The two main types of machine learning are supervised learning and unsupervised learning.

### Supervised Learning

![supervised](https://media.geeksforgeeks.org/wp-content/uploads/20231121154747/Supervised-learning.png)

Supervised learning is a category of machine learning algorithms that learn from input/output pairs. In this approach, the algorithm is provided with a set of labeled data where each input is paired with the correct output. The goal is for the algorithm to learn a mapping from inputs to outputs so that it can accurately predict the output for new, unseen inputs. By analyzing the patterns and relationships in the training data, the algorithm can generalize this knowledge to make predictions on new data.

**Example:** A supervised learning model designed to recognize handwritten digits would be trained on images of handwritten digits and their corresponding labels. Once trained, the model can identify the correct digit in any new handwritten image.

#### Problem Classification

Supervised learning problems can be broadly categorized into two types: classification and regression.

![class-regr](https://static.javatpoint.com/tutorial/machine-learning/images/regression-vs-classification-in-machine-learning.png)

##### Classification Problems

A classification problem is a type of supervised learning task where the goal is to predict a discrete label or category for a given input. The input data can be anything from images to text to numerical values, and the output is a label that classifies the input into one of several predefined categories.

- **Binary Classification:** Binary classification is a specific type of classification where the output has two possible classes.

  **Example:** Classifying emails as "spam" or "not spam" is a binary classification problem.

- **Multi-class Classification:** Multi-class classification involves predicting a single output label from more than two distinct classes.

  **Example:** Classifying images of animals into categories such as "cat," "dog," "horse," and "bird" is a multi-class classification problem.

- **Multi-label Classification:** In multi-label classification, multiple labels can be assigned to each instance.

  **Example:** Assigning multiple tags to a news article based on its content, such as "politics," "sports," and "technology," is a multi-label classification problem.

##### Regression Problems

A regression problem is another type of supervised learning task where the goal is to predict a continuous value for a given input. This involves predicting numerical values rather than discrete labels.

![reg-types](https://static.javatpoint.com/tutorial/machine-learning/images/types-of-regression.png)

**Example:** Predicting the price of a house based on features such as the number of bedrooms, square footage, and location is a regression problem.

### Unsupervised Learning

![unsupervised](https://media.geeksforgeeks.org/wp-content/uploads/20231124111325/Unsupervised-learning.png)

Unsupervised learning involves training algorithms on data without labeled responses. The goal is for the algorithm to identify patterns, structures, or relationships within the data without any guidance on what the outputs should be. This type of learning is useful for exploratory data analysis and for discovering hidden structures in the data.

**Example:** An unsupervised learning algorithm might analyze a dataset of customer purchase histories to group customers with similar buying behaviors into clusters.

#### Clustering

![clus](https://databasetown.com/wp-content/uploads/2023/05/Unsupervised-Learning.jpg)

Clustering is a common unsupervised learning task where the goal is to group a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups.

**Example:** K-means clustering can be used to segment customers into different groups based on their purchasing patterns.

#### Dimensionality Reduction

Dimensionality reduction techniques are used in unsupervised learning to reduce the number of random variables under consideration, by obtaining a set of principal variables. This can be useful for visualizing high-dimensional data or for reducing noise.

**Example:** Principal Component Analysis (PCA) is a technique used to reduce the dimensionality of a dataset while retaining most of the variability in the data.

---

## Features and Labels

### Features

Features are the individual measurable properties or characteristics of the data used by the machine learning algorithm to make predictions. In a dataset, features are the columns, and each feature represents a specific attribute of the data.

**Example:** In a dataset of house prices, features might include the number of bedrooms, square footage, and location.

### Labels

In the context of supervised learning, labels are the correct output values provided with the input data during the training phase. Each label corresponds to an input example and indicates the desired output that the model should predict.

**Example:** In an image classification task, if the input is an image of a cat, the label would be "cat."

---

## Training and Validation

### Training Data

Training data is the dataset used to train a machine learning model. It consists of input examples paired with their corresponding labels. The model learns from this data by finding patterns and relationships between the inputs and outputs.

**Example:** In a supervised learning task for image recognition, the training data would include a large number of images and their corresponding labels, such as "cat," "dog," "car," etc.

### Validation Data

Validation data is a subset of the data set aside from the training data, used to evaluate the performance of the machine learning model during training. It helps in tuning the model's hyperparameters and preventing overfitting, which is when the model performs well on the training data but poorly on new, unseen data.

**Example:** After training an image classification model on the training data, the validation data would be used to check how well the model generalizes to new images.

---

## Model Evaluation

![model-val](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png)

### Overfitting

Overfitting occurs when a machine learning model learns the training data too well, capturing noise and details that do not generalize to new, unseen data. An overfitted model performs exceptionally well on the training data but poorly on the validation or test data.

**Example:** A model that memorizes the training examples rather than learning the underlying patterns is overfitting.

### Underfitting

Underfitting happens when a machine learning model is too simple to capture the underlying patterns in the data, resulting in poor performance on both the training and validation data. Underfitting can occur when the model is not complex enough or when there is insufficient training data.

**Example:** Using a linear model to fit a complex, non-linear dataset may result in underfitting.

---

## Neural Networks and Deep Learning

### Neural Network

![nn](https://miro.medium.com/v2/resize:fit:828/format:webp/1*ZXAOUqmlyECgfVa81Sr6Ew.png)

A neural network is a computational model inspired by the human brain's structure and function. It consists of interconnected layers of nodes (neurons), where each node represents a mathematical function. Neural networks are used in deep learning to model complex patterns in data. They have multiple layers, including input layers, hidden layers, and output layers.

**Example:** A neural network designed for image recognition might have several hidden layers to capture different levels of abstraction in the image data, leading to accurate recognition of objects within the images.

### Activation Functions

Activation functions are mathematical functions used in neural networks to introduce non-linearity into the model. This non-linearity allows the neural network to learn complex patterns and relationships in the data.

![active-fucn](https://miro.medium.com/v2/resize:fit:1200/1*ZafDv3VUm60Eh10OeJu1vw.png)

**Common Activation Functions:**
- **Sigmoid Function:** Squashes the input to a range between 0 and 1.
- **Tanh Function:** Squashes the input to a range between -1 and 1.
- **ReLU (Rectified Linear Unit):** Outputs the input directly if it is positive; otherwise, it outputs zero.

