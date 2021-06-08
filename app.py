import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import model

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        f = request.files['recording'].read()
        print('file uploaded successfully')
        healthy, ill  = model.predict(f)

        return jsonify(healthy=str(healthy), ill=str(ill))
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)