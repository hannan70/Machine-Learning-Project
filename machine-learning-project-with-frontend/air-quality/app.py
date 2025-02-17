from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('air_quality_with_random_forest_classification.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)