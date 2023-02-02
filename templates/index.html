# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the logistic regression model
filename = 'classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Load the standardise data
filename2 = 'scaler.pkl'
scaler = pickle.load(open(filename2, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    # return render_template('index.html')
    return render_template('landingPage.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = request.form['cholesterol']
        gluc = request.form['gluc']
        smoke = request.form['smoke']
        alco = request.form['alco']
        active = request.form['active']

        if gender == 'Female':
            gender = 1
        else:
            gender = 2

        if cholesterol == 'Normal':
            cholesterol = 1
        elif cholesterol == 'Above normal':
            cholesterol = 2
        else:
            cholesterol = 3

        if gluc == 'Normal':
            gluc = 1
        elif gluc == 'Above Normal':
            gluc = 2
        else:
            gluc = 3

        if smoke == 'Yes':
            smoke = 1
        else:
            smoke = 0

        if alco == 'Yes':
            alco = 1
        else:
            alco = 0

        if active == 'Yes':
            active = 1
        else:
            active = 0

        data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

        data_s = scaler.transform(data)

        my_prediction = classifier.predict(data_s)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
