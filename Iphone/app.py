from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load data and train model
csv_path = 'diabetes.csv'  # Update with the correct path to your dataset
data = pd.read_csv(csv_path)

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

model = LinearRegression()
model.fit(X, y)

@app.route('/tee')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = [float(request.form[field]) for field in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    prediction = model.predict([inputs])
    output = "Diabetic" if prediction[0] >= 0.5 else "Not Diabetic"
    return render_template('index1.html', prediction_text=f'The person is {output}')

if __name__ == "__main__":
    app.run(debug=True)
