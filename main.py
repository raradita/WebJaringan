from flask import Flask, render_template, request, jsonify
from joblib import load  # Jika Anda menggunakan joblib untuk menyimpan model
import joblib
import numpy as np
# import pandas as pd
import lightgbm
import sys

print(sys.executable)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    

# Load the machine learning model
model_path = "./models/best_model.joblib"
model = joblib.load(model_path)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            print('Debug Info: No file part')
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            print('Debug Info: No selected file')
            return jsonify({'error': 'No selected file'})

        # Read the CSV file
        df = pd.read_csv(file)

        # Assuming the features are all columns except the target variable
        features = df.drop(columns=['target_variable'])

        # Print debug information about the features
        print('Debug Info: Features shape -', features.shape)
        print('Debug Info: Features columns -', features.columns)

        # Make predictions
        predictions = model.predict(features)

        # Add predictions to the DataFrame
        df['prediction'] = predictions.tolist()

        # Print debug information about predictions
        print('Debug Info: Predictions -', df['prediction'].tolist())

        # Return the DataFrame as JSON
        return df.to_json(orient='records')

    except Exception as e:
        print('Debug Info: Exception -', str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



