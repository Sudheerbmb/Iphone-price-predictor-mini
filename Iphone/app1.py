from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


app = Flask(__name__)

# Load data and train model
csv_path = 'diabetes.csv'  # Update with the correct path to your dataset
data = pd.read_csv(csv_path)

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

model = LinearRegression()
model.fit(X, y)




predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)
print(mse)
print(mae)
print(r2)



