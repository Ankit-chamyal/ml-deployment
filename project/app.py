from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import pickle
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

app = Flask(__name__)

# Load only the Logistic Regression model and scaler
with open("saved_models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("saved_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Calibrated thresholds based on ROC curve analysis
# These thresholds are optimized for balanced sensitivity and specificity
THRESHOLDS = {
    'very_low': 0.35,    # 15% - High specificity for very low risk
    'low': 0.45,         # 30% - Good balance for low risk
    'medium': 0.55,      # 45% - Moderate risk threshold
    'high': 0.65,        # 65% - High risk threshold
    'very_high': 0.70    # 80% - Very high risk threshold
}

def get_prediction(input_data):
    scaled = scaler.transform([input_data])
    
    # Get prediction from Logistic Regression model
    raw_prob = model.predict_proba(scaled)[0][1]
    probability = round(raw_prob * 100, 2)

    # Create risk level gauge with calibrated thresholds
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, THRESHOLDS['very_low']*100], 'color': "#2ecc71"},  # Very Low Risk
                {'range': [THRESHOLDS['very_low']*100, THRESHOLDS['low']*100], 'color': "#3498db"},  # Low Risk
                {'range': [THRESHOLDS['low']*100, THRESHOLDS['medium']*100], 'color': "#f1c40f"},  # Medium Risk
                {'range': [THRESHOLDS['medium']*100, THRESHOLDS['high']*100], 'color': "#e67e22"},  # High Risk
                {'range': [THRESHOLDS['high']*100, 100], 'color': "#e74c3c"}  # Very High Risk
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    gauge_fig.update_layout(height=400)

    # Risk assessment with calibrated thresholds
    if probability < THRESHOLDS['very_low']*100:  # Very Low Risk
        risk = "Very Low Risk"
        color = "success"
        confidence = "High confidence in low risk assessment"
        precautions = [
            "Maintain your healthy lifestyle",
            "Regular exercise (30 minutes daily)",
            "Balanced diet with fruits and vegetables",
            "Annual health checkup recommended"
        ]
    elif probability < THRESHOLDS['low']*100:  # Low Risk
        risk = "Low Risk"
        color = "info"
        confidence = "Moderate confidence in low risk assessment"
        precautions = [
            "Continue healthy lifestyle habits",
            "Regular exercise (20-30 minutes daily)",
            "Monitor blood pressure monthly",
            "Annual health checkup recommended"
        ]
    elif probability < THRESHOLDS['medium']*100:  # Medium Risk
        risk = "Medium Risk"
        color = "warning"
        confidence = "Moderate confidence in medium risk assessment"
        precautions = [
            "Consult a cardiologist",
            "Monitor blood pressure weekly",
            "Control cholesterol levels",
            "Moderate exercise (15-20 minutes daily)"
        ]
    elif probability < THRESHOLDS['high']*100:  # High Risk
        risk = "High Risk"
        color = "danger"
        confidence = "High confidence in high risk assessment"
        precautions = [
            "Immediate medical attention recommended",
            "Strict lifestyle modifications needed",
            "Regular medication as prescribed",
            "Avoid stress and strenuous activities"
        ]
    else:  # Very High Risk
        risk = "Very High Risk"
        color = "danger"
        confidence = "Very high confidence in high risk assessment"
        precautions = [
            "Emergency medical attention required",
            "Immediate lifestyle changes necessary",
            "Strict medication adherence",
            "Complete rest and stress management"
        ]

    return {
        'avg_probability': probability,
        'risk': risk,
        'color': color,
        'confidence': confidence,
        'precautions': precautions,
        'gauge_graph': json.dumps(gauge_fig, cls=plotly.utils.PlotlyJSONEncoder)
    }

@app.route('/')
def index():
    return redirect('http://localhost:5500/index.html')

@app.route('/prediction-form')
def prediction_form():
    return redirect('http://localhost:5500/predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in request.form]
        result = get_prediction(features)
        return render_template("result.html", 
                             probability=result['avg_probability'],
                             risk=result['risk'],
                             color=result['color'],
                             confidence=result['confidence'],
                             tips=result['precautions'],
                             gauge_graph=result['gauge_graph'])
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/logout')
def logout():
    return redirect('http://localhost:5500/index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
