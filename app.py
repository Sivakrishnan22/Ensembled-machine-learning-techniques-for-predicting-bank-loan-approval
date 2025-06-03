from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    input_data = [
        float(form_data['gender'] == 'Male'),
        float(form_data['married'] == 'Yes'),
        float(3 if form_data['dependents'] == '3+' else int(form_data['dependents'])),
        float(form_data['education'] == 'Graduate'),
        float(form_data['self_employed'] == 'Yes'),
        float(form_data['applicant_income']),
        float(form_data['coapplicant_income']),
        float(form_data['loan_amount']),
        float(form_data['loan_term']),
        float(form_data['credit_history']),
        float({'Urban': 2, 'Semiurban': 1, 'Rural': 0}[form_data['property_area']])
    ]
    scaled_data = scaler.transform([input_data])
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    result = "Approved" if prediction == 1 else "Rejected"
    return render_template('result.html', result=result, probability=f"{probability*100:.1f}%")

if __name__ == '__main__':
    app.run(debug=True)
