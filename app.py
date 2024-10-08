from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        sqft_living = float(request.form['sqft_living'])
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        yr_built = int(request.form['yr_built'])
        sqft_lot = float(request.form['sqft_lot'])

        # Prepare features for prediction
        features = np.array([[sqft_living, bedrooms, bathrooms, yr_built, sqft_lot]])
        
        # Make prediction
        prediction = model.predict(features)

        # Return the result
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction[0]:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text='Error: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)
