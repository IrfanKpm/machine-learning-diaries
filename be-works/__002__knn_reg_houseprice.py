
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df_house_prices = pd.DataFrame({
    'Square Footage': [1500, 1800, 1600, 2000, 1200, 2500, 1700, 2200, 1300, 2400, 2100, 1900, 2800, 1400, 2300,
                        1750, 1900, 1650, 2100, 1250, 2600, 1800, 2300, 1400, 2500, 2200, 2000, 2700, 1500],
    'Bedrooms': [3, 4, 3, 4, 2, 5, 3, 4, 2, 4, 4, 3, 5, 2, 4, 
                 3, 4, 3, 4, 2, 5, 3, 4, 2, 5, 4, 3, 5, 2],
    'Price': [300, 350, 320, 400, 250, 500, 340, 450, 270, 470, 430, 360, 550, 290, 460,
              330, 370, 340, 430, 260, 520, 360, 460, 280, 490, 440, 410, 550, 300]
})
# Prepare data
X = df_house_prices[['Square Footage', 'Bedrooms']]
y = df_house_prices.Price

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array of axes for easy iteration
axs = axs.flatten()

# Loop over different numbers of neighbors
for idx, n in enumerate([1, 3, 5, 7]):
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[n > {n}] Mean Squared Error: {mse}")

    # Plotting
    axs[idx].scatter(y_test, y_pred, color='blue')
    axs[idx].set_xlabel('Actual Prices')
    axs[idx].set_ylabel('Predicted Prices')
    axs[idx].set_title(f'Actual vs Predicted Prices (n={n})')
    axs[idx].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

# Adjust layout
plt.tight_layout()
plt.show()
