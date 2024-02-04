from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

print(sys.executable)

app = Flask(__name__)

# Load the machine learning model
scaler = joblib.load("./models/scaler_model.joblib")
rfe = joblib.load("./models/rfe_model.joblib")
model = joblib.load("./models/best_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

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

        # Fungsi mengubah jenis serangan pada label kedalam kategori anomali
        def change_labels(data):
          data.labels.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'anomali',inplace=True)
          data.labels.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'anomali',inplace=True)
          data.labels.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'anomali',inplace=True)
          data.labels.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'anomali',inplace=True)
        
        change_labels(df)

        # Label Encoder
        # 'labels'
        le_labels = LabelEncoder()
        df['labels'] = le_labels.fit_transform(df['labels'])
        # 'protocol_type'
        le_protocol_type = LabelEncoder()
        df['protocol_type'] = le_protocol_type.fit_transform(df['protocol_type'])
        # 'service'
        le_service = LabelEncoder()
        df['service'] = le_service.fit_transform(df['service'])
        # 'flag'
        le_flag = LabelEncoder()
        df['flag'] = le_flag.fit_transform(df['flag'])

        # Buat DataFrame baru tanpa kolom 'labels'
        df_new = df.drop(['labels'], axis=1)
        
        # Make predictions
        new_data_scaled = scaler.transform(df_new)
        new_data_rfe = rfe.transform(new_data_scaled)
        prediction = model.predict(new_data_rfe)

        # Misalnya, new_data_rfe adalah hasil transformasi menggunakan RFE pada data yang telah discaler
        df_new_rfe = pd.DataFrame(new_data_rfe, columns=df_new.columns[rfe.support_])
        # Menambahkan kolom 'labels' asli ke dalam DataFrame df_new_rfe
        df_new_rfe['labels'] = df['labels']
        # Menambahkan hasil prediksi ke dalam DataFrame df_new_rfe
        df_new_rfe['prediksi'] = prediction

        # Membuat fungsi untuk konversi nilai prediksi menjadi string
        def convert_to_class(prediction):
            return 'anomali' if prediction == 0 else 'normal'
        df_new_rfe['prediksi'] = [convert_to_class(pred) for pred in prediction]

        # Mengembalikan nilai ke bentuk awal
        df_new_rfe['labels'] = le_labels.inverse_transform(df['labels'])
        df_new_rfe['protocol_type'] = le_protocol_type.inverse_transform(df['protocol_type'])
        df_new_rfe['flag'] = le_flag.inverse_transform(df['flag'])

        # Mengonversi DataFrame ke dalam bentuk array
        array_data = df_new_rfe.to_numpy()

        # List dengan nama kolom
        result_list = [dict(zip(df_new_rfe.columns, row)) for row in array_data]

        # Response
        return jsonify(result_list)

    except Exception as e:
        print('Debug Info: Exception -', str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
