import numpy as np
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
with open('knn_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Load the scaler
scaler = MinMaxScaler()

# Define crop dictionary
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
    'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

predicted_crop = {v: k for k, v in crop_dict.items()}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Prepare input data
        input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        prediction = best_model.predict(input_values)
        crop_name = predicted_crop.get(prediction[0], "Unknown Crop")

        return render_template('index.html', prediction_text=f'The best crop to cultivate is {crop_name}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
