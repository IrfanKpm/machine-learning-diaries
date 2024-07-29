

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_house_prices = pd.DataFrame({
    'Square Footage': [1500, 1800, 1600, 2000, 1200, 2500, 1700, 2200, 1300, 2400, 2100, 1900, 2800, 1400, 2300,
                        1750, 1900, 1650, 2100, 1250, 2600, 1800, 2300, 1400, 2500, 2200, 2000, 2700, 1500],
    'Bedrooms': [3, 4, 3, 4, 2, 5, 3, 4, 2, 4, 4, 3, 5, 2, 4, 
                 3, 4, 3, 4, 2, 5, 3, 4, 2, 5, 4, 3, 5, 2],
    'Price': [300, 350, 320, 400, 250, 500, 340, 450, 270, 470, 430, 360, 550, 290, 460,
              330, 370, 340, 430, 260, 520, 360, 460, 280, 490, 440, 410, 550, 300]
})

# Car Prices
df_car_prices = pd.DataFrame({
    'Age': [1, 2, 3, 1, 4, 2, 3, 5, 6, 1, 2, 4, 3, 5, 7,
            2, 3, 4, 1, 6, 3, 2, 5, 4, 6, 7, 3, 4, 5],
    'Mileage': [10, 20, 30, 15, 40, 25, 35, 50, 55, 12, 18, 45, 28, 48, 60,
                20, 30, 40, 15, 55, 30, 25, 50, 40, 55, 60, 35, 45, 50],
    'Price': [30, 25, 20, 28, 18, 22, 21, 15, 14, 29, 26, 17, 23, 16, 13,
              25, 20, 18, 30, 15, 22, 28, 16, 17, 19, 13, 24, 27, 18]
})

# Salary Prediction
df_salary = pd.DataFrame({
    'Years of Experience': [1, 3, 5, 7, 10, 2, 4, 6, 8, 12, 9, 11, 3, 6, 7,
                             4, 8, 5, 9, 11, 3, 6, 7, 10, 12, 5, 7, 8, 10],
    'Education Level': [0, 1, 1, 2, 2, 0, 1, 1, 2, 3, 2, 3, 1, 2, 2,
                         1, 2, 1, 3, 3, 1, 2, 3, 0, 1, 2, 3, 1, 2],
    'Salary': [30, 50, 60, 80, 90, 35, 55, 65, 85, 100, 75, 95, 52, 70, 78,
               55, 80, 60, 85, 95, 50, 70, 90, 40, 65, 75, 85, 100, 60]
})

# Create DataFrames for Classification datasets

# Customer Churn
df_customer_churn = pd.DataFrame({
    'Age': [25, 45, 35, 50, 29, 40, 55, 30, 47, 31, 41, 33, 52, 37, 46,
            28, 42, 56, 31, 49, 39, 26, 41, 54, 30, 46, 35, 50, 28],
    'Monthly Spending': [50, 60, 55, 70, 45, 65, 75, 50, 80, 52, 68, 48, 78, 60, 72,
                         55, 65, 75, 52, 70, 60, 50, 55, 65, 70, 75, 60, 80, 50],
    'Churn': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
})

# Iris Classification
df_iris = pd.DataFrame({
    'Sepal Length': [5.1, 7.0, 6.3, 5.5, 5.0, 6.7, 4.9, 6.5, 5.7, 5.8, 5.4, 6.4, 6.8, 5.9, 6.0,
                      5.2, 6.1, 5.8, 6.3, 5.0, 6.9, 5.6, 5.7, 6.2, 5.4, 6.1, 5.7, 5.9, 6.5],
    'Sepal Width': [3.5, 3.2, 3.3, 2.6, 3.4, 3.1, 3.0, 2.8, 2.5, 2.7, 3.9, 2.9, 3.2, 3.0, 2.9,
                    3.6, 2.9, 3.0, 2.8, 3.4, 3.1, 2.7, 2.5, 2.9, 3.0, 2.8, 2.7, 3.1, 3.0],
    'Petal Length': [1.4, 4.7, 6.0, 4.4, 1.5, 5.6, 1.4, 4.6, 5.0, 4.1, 1.7, 4.3, 5.9, 5.1, 4.5,
                      1.5, 4.4, 5.2, 4.7, 1.6, 5.8, 4.0, 5.1, 5.8, 4.6, 5.0, 4.3, 5.5, 5.6],
    'Petal Width': [0.2, 1.4, 2.5, 1.2, 0.2, 2.4, 0.2, 1.5, 2.0, 1.0, 0.4, 1.3, 2.3, 1.8, 1.5,
                    0.2, 1.2, 2.0, 1.5, 0.2, 1.4, 1.0, 1.2, 1.6, 1.3, 1.4, 1.5, 2.0, 1.4],
    'Species': [0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 1, 2, 2, 1,
                0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2]
})

# Email Spam Detection
df_spam_detection = pd.DataFrame({
    'Email Length': [200, 350, 150, 400, 180, 500, 160, 420, 220, 480, 140, 390, 200, 370, 210,
                      220, 400, 180, 350, 500, 150, 420, 200, 300, 450, 170, 420, 220, 480],
    'Number of Links': [2, 5, 1, 7, 3, 8, 2, 6, 4, 9, 1, 5, 3, 4, 2,
                        4, 7, 3, 6, 8, 2, 5, 7, 1, 6, 3, 4, 2, 6],
    'Spam': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
             0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
})


# Plotting
fig, axs = plt.subplots(2, 3, figsize=(12, 7))

# Regression Plots
sns.scatterplot(data=df_house_prices, x='Square Footage', y='Price', hue='Bedrooms', ax=axs[0, 0])
axs[0, 0].set_title('House Prices')

sns.scatterplot(data=df_car_prices, x='Mileage', y='Price', hue='Age', ax=axs[0, 1])
axs[0, 1].set_title('Car Prices')

sns.scatterplot(data=df_salary, x='Years of Experience', y='Salary', hue='Education Level', ax=axs[0, 2])
axs[0, 2].set_title('Salary Prediction')

# Classification Plots
sns.scatterplot(data=df_customer_churn, x='Age', y='Monthly Spending', hue='Churn', ax=axs[1, 0])
axs[1, 0].set_title('Customer Churn')

sns.scatterplot(data=df_iris, x='Sepal Length', y='Petal Length', hue='Species', ax=axs[1, 1])
axs[1, 1].set_title('Iris Classification')

sns.scatterplot(data=df_spam_detection, x='Email Length', y='Number of Links', hue='Spam', ax=axs[1, 2])
axs[1, 2].set_title('Email Spam Detection')

plt.tight_layout()
plt.show()


