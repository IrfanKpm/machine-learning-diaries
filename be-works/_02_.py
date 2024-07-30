
   # Feature Scaling in Scikit learn

import numpy as np
import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Generating random data
y = np.random.random(300)
random.seed(300)
x = np.arange(300)

print(x.shape)
print(y.shape)

# Standardization function
def Standardization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std

# Applying manual standardization
Y = Standardization(y)

# Setting up the subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5))

# Plot before standardization
ax1.set_title('Before Standardization')
sns.kdeplot(y, ax=ax1)
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')

# Plot after manual standardization
ax2.set_title('Manual Standardization')
sns.kdeplot(Y, ax=ax2)
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')

# Applying Scikit-learn StandardScaler
scalar = StandardScaler()
y = y.reshape(-1, 1)
Y_std = scalar.fit_transform(y).flatten()

# Plot after Scikit-learn standardization
ax3.set_title('Scikit-learn Standardization')
sns.kdeplot(Y_std, ax=ax3)
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')

plt.tight_layout()
plt.show()

# Normalization
minmaxScaler = MinMaxScaler()
Y_minmax = minmaxScaler.fit_transform(y).flatten()

# Plot for normalization
plt.figure(figsize=(8, 5))
sns.kdeplot(Y_minmax, fill=True)
plt.title('Normalization')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
