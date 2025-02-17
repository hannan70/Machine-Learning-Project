
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open("student_study_performance.pkl", 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def get_predict():
    study_hours = [int(x) for x in request.form.values()]
    features = [np.array(study_hours)]
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    formated_price = f"{int(prediction[0])}%"
    return render_template("index.html", prediction=formated_price)

if __name__ == '__main__':
    app.run(debug=True)

