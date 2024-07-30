
   # Linear Regression Multiple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# URL of the CSV file
url = 'https://raw.githubusercontent.com/braintek2021/Machine-Learning-Malayalam/main/Multi%20variable%20linear%20regression/Advertising.csv'

# Read the CSV file into a DataFrame
advert = pd.read_csv(url)

# Define features and target variable
X = advert[['TV', 'radio', 'newspaper']]
y = advert['sales']

#sns.pairplot(advert,x_vars=['TV','radio','newspaper'],y_vars='sales',height=7,aspect=0.7)
#plt.show()

# Fit the Linear Regression model
lr = LinearRegression()
lr.fit(X, y)


# Plotting the regression lines
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.regplot(x='TV', y='sales', data=advert, ax=axes[0], line_kws={"color": "red"})
axes[0].set_title('Sales vs TV')

sns.regplot(x='radio', y='sales', data=advert, ax=axes[1], line_kws={"color": "red"})
axes[1].set_title('Sales vs Radio')

sns.regplot(x='newspaper', y='sales', data=advert, ax=axes[2], line_kws={"color": "red"})
axes[2].set_title('Sales vs Newspaper')

plt.tight_layout()
plt.show()


# Print intercept and coefficients
print('Intercept:', lr.intercept_)
print('Coefficients:', lr.coef_)

sns.heatmap(advert.corr(),annot=True)

plt.show()

y_pred = lr.predict(X)

print(f"score : {lr.score(X,y)}")
print(f"MSE : {mean_absolute_error(y,y_pred)}")