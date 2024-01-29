from flask import Flask, render_template, request
from joblib import load  # Jika Anda menggunakan joblib untuk menyimpan model
import numpy as np
import sys

print(sys.executable)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
