
    # Scikit learn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = np.loadtxt('./ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]

# Reshape data
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create linear regression model
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

# Plotting
plt.figure(figsize=(12, 5))  # Set figure size

# First plot: scatter plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.plot(X, y, 'ro', ms=8, mec='k')
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000')
plt.title('Scatter Plot')

# Second plot: regression line
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(X, y,color='b')
plt.plot(X, y_pred, color='k')
plt.xlabel('Population of City in 10,000')
plt.title('Regression Line')

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

print(f"coef > {lr.coef_}")
print(f"intercept > {lr.intercept_}")


print(f"score > {lr.score(X,y)}")
print(f"MSE > {mean_squared_error(y,y_pred)}")