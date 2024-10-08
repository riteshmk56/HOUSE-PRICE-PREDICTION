import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
data = pd.read_csv('data.csv')  # Change this to the actual path of your dataset

# Define features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'yr_built', 'sqft_lot']]  # Adjust based on your features
y = data['price']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib")
