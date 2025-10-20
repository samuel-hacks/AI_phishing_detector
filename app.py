from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib 
import pandas as pd
import re

from urllib.parse import urlparse

import os
import time

app = Flask(__name__)

CORS(app)

print("Loading the model...")

model_filename = 'phishing_model.joblib'
model = joblib.load(model_filename)

print("Model loaded successfully.")

API_KEY = "AIzaSyAnC9Hwi6YQwL6G8bxb1YEKdTQxlv5Vygo"
GSB_CACHE_FILE = "gsb_cache.json"

def load_gsb_cache():
    if os.path.exists(GSB_CACHE_FILE):
        with open(GSB_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_gsb_cache(cache):
    with open(GSB_CACHE_FILE, "w") as f:
        json.dump(cache, f)

gsb_cache = load_gsb_cache()

def get_url_length(url): 
    return len(str(url))

def has_at_symbol(url): 
    return 1 if "@" in str(url) else 0

def has_ip_address(url):
    match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])', str(url))
    return 1 if match else 0

def get_dot_count(url): 
    return str(url).count(".")

def count_suspicious_keywords(url):
    keywords = ['login', 'verify', 'account', 'security', 'update', 'signin', 'banking', 'confirm']
    count = 0
    for keyword in keywords:
        if keyword in str(url).lower():
            count += 1
    return count

def count_special_chars(url):
    special_chars = ['-', '/', '?', '=', '.']
    count = 0
    for char in special_chars:
        count += str(url).count(char)
    return count

def has_hyphen_in_domain(url):
    try:
        domain = urlparse(str(url)).netloc
        return 1 if '-' in domain else 0
    except:
        return 0

def get_entropy(text):
    text = str(text)
    if not text: return 0
    p, lns = Counter(text), float(len(text))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def count_numeric_chars(url):
    return sum(c.isdigit() for c in str(url))

def vowel_consonant_ratio(url):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    v_count = sum(1 for char in str(url) if char in vowels)
    c_count = sum(1 for char in str(url) if char in consonants)
    if c_count == 0: return 0
    return v_count / c_count
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
    url_df['suspicious_keywords'] = url_df['url'].apply(count_suspicious_keywords)
    url_df['special_chars'] = url_df['url'].apply(count_special_chars)
    url_df['hyphen_in_domain'] = url_df['url'].apply(has_hyphen_in_domain)

    df['entropy'] = df['url'].apply(get_entropy)
    df['numeric_chars'] = df['url'].apply(count_numeric_chars)
    df['vowel_consonant_ratio'] = df['url'].apply(vowel_consonant_ratio)

    feature_columns = ["url_length", "has_at", "has_ip", "dot_count", 'suspicious_keywords', 'special_chars', 'hyphen_in_domain', 'entropy', 'numeric_chars', 'vowel_consonant_ratio']
    features = url_df[feature_columns]

    print(f"DEBUG: Features sent to model: {features.to_dict(orient="records")}")

    prediction = model.predict(features)

    result = "bad" if prediction[0] == 1 else "good"

    return jsonify({"url": url_to_check, "prediction": result})

if __name__ == "__main__":
    app.run(host= "0.0.0.0", port = 5000, debug = True)