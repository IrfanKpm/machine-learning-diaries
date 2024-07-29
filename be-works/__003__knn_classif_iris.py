import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Create a synthetic dataset
X, y = make_classification(n_samples=140, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, n_classes=3, random_state=42)

# Create a DataFrame from the dataset
df_synthetic = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df_synthetic['Class'] = y

# Display the first 10 rows of the synthetic dataset
print(df_synthetic.head(10))

# Features and target variable
X = df_synthetic[['Feature 1', 'Feature 2']].values  # Convert to numpy array
y = df_synthetic['Class'].values  # Convert to numpy array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Define a colormap for the plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Iterate over different values for n_neighbors
for n_neighbors, ax in zip([1, 3, 5, 7], axes.flatten()):
    # Create and fit the KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    # Plot the training data
    scatter_train = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                              cmap=cmap_bold, edgecolor='k', s=50, label='Training data')

    # Plot the test data
    scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                             cmap=cmap_bold, edgecolor='k', marker='s', s=100, label='Test data')

    # Set the title with train and test scores
    ax.set_title(
        "{} neighbor(s)\n train accuracy: {:.2f} test accuracy: {:.2f}".format(
            n_neighbors, clf.score(X_train, y_train),
            clf.score(X_test, y_test)
        )
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Add legend
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()
