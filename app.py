from flask import Flask, request, jsonify
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# Loading models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Server is running"})

@app.route("/predict", methods=['POST'])
def predict():
    data = request.form
    Year = data['Year']
    average_rain_fall_mm_per_year = data['average_rain_fall_mm_per_year']
    pesticides_tonnes = data['pesticides_tonnes']
    avg_temp = data['avg_temp']
    Area = data['Area']
    Item = data['Item']

    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = dtr.predict(transformed_features).reshape(1, -1)

    return jsonify({"prediction": float(prediction[0][0])})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
