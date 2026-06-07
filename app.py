import os
import joblib
import numpy as np
from flask import Flask, request, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "rationshop_model.pkl"))
le_district = joblib.load(os.path.join(BASE_DIR, "le_district.pkl"))

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ration Shop Predictor</title>

<style>
body{margin:0;min-height:100vh;font-family:Arial;background:#0f172a;color:white;display:flex;justify-content:center;align-items:center}
.card{background:rgba(255,255,255,0.08);padding:25px;border-radius:15px;width:420px}
input{width:100%;padding:8px;margin:5px 0;border-radius:8px;border:none}
button{width:100%;padding:10px;background:linear-gradient(45deg,cyan,purple);border:none;color:white;border-radius:10px;cursor:pointer}
.result{margin-top:15px}
</style>

</head>
<body>

<div class="card">
<h2>Ration Shop Demand Predictor</h2>

<form action="/predict" method="POST">

<input name="District" placeholder="District Code" required>
<input name="Taluks" placeholder="Taluks" required>
<input name="Shops" placeholder="Shops" required>
<input name="Cards" placeholder="Cards" required>

<button type="submit">Predict</button>

</form>

{% if prediction %}
<div class="result">
<p><b>District:</b> {{ district_name }}</p>
<p><b>Predicted Customers:</b> {{ prediction }}</p>
</div>
{% endif %}

</div>
</body>
</html>
"""

def preprocess(features):
    return np.array(features).reshape(1, -1)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        district = float(request.form["District"])
        taluks = float(request.form["Taluks"])
        shops = float(request.form["Shops"])
        cards = float(request.form["Cards"])

        features = [district, taluks, shops, cards]

        prediction = model.predict(preprocess(features))[0]

        district_name = le_district.inverse_transform([int(district)])[0]

        return render_template_string(
            HTML,
            prediction=round(float(prediction), 2),
            district_name=district_name
        )

    except Exception as e:
        return str(e)

