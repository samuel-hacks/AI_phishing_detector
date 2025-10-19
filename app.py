from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib 
import pandas as pd
import re

app = Flask(__name__)

CORS(app)

print("Loading the model...")

model_filename = "phishing_model.joblib"
model = joblib.load(model_filename)

print("Model loaded successfully.")

def get_url_length(url): return len(str(url))
def has_at_symbol(url): return 1 if "@" in str(url) else 0
def has_ip_address(url):
    match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])', str(url))
    return 1 if match else 0
def get_dot_count(url): return str(url).count(".")

@app.route("/")

def home():
    return render_template("web.html")

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()

    if "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url_to_check = data["url"]

    url_df = pd.DataFrame({"url": [url_to_check]})

    url_df['url_length'] = url_df['url'].apply(get_url_length)
    url_df['has_at'] = url_df['url'].apply(has_at_symbol)
    url_df['has_ip'] = url_df['url'].apply(has_ip_address)
    url_df['dot_count'] = url_df['url'].apply(get_dot_count)

    feature_columns = ["url_length", "has_at", "has_ip", "dot_count"]
    features = url_df[feature_columns]

    print(f"DEBUG: Features sent to model: {features.to_dict(orient="records")}")

    prediction = model.predict(features)

    result = "bad" if prediction[0] == 1 else "good"

    return jsonify({"url": url_to_check, "prediction": result})

if __name__ == "__main__":
    app.run(host= "0.0.0.0", port = 5000, debug = True)