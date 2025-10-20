import pandas as pd
import re

from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

import math
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

import os
import time

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

print("Step 1: Loading and preparing data...")

csv_file_name = "phishing_site_urls.csv"

try:
    df = pd.read_csv(csv_file_name)
    df = df.rename(columns = {"URL": "url", "Label": "label"})

    df.dropna(subset=["url"], inplace = True)

    df['url'] = df['url'].astype(str)

    print("Dataset loaded and cleaned successfully!")

except FileNotFoundError:
    print(f"Error: Dataset '{csv_file_name}' not found.")
    print("Please run prepare_data.py first to download it.")
    exit()

def get_url_length(url):
    return len(str(url))

def has_at_symbol(url):
    return 1 if "@" in str(url) else 0

def has_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])',
        str(url)
    )
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

    if not text:
        return 0
    
    p, lns = Counter(text), float(len(text))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def count_numeric_chars(url):
    return sum(c.isdigit() for c in str(url))

def vowel_consonant_ratio(url):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    v_count = sum(1 for char in str(url) if char in vowels)
    c_count = sum(1 for char in str(url) if char in consonants)
    if c_count == 0:
        return 0
    return v_count / c_count

df["url_length"] = df['url'].apply(get_url_length)
df['has_at'] = df['url'].apply(has_at_symbol)
df['has_ip'] = df['url'].apply(has_ip_address)
df['dot_count'] = df['url'].apply(get_dot_count)
df['suspicious_keywords'] = df['url'].apply(count_suspicious_keywords)
df['special_chars'] = df['url'].apply(count_special_chars)
df['hyphen_in_domain'] = df['url'].apply(has_hyphen_in_domain)

df['entropy'] = df['url'].apply(get_entropy)
df['numeric_chars'] = df['url'].apply(count_numeric_chars)
df['vowel_consonant_ratio'] = df['url'].apply(vowel_consonant_ratio)

feature_columns = ['url_length', 'has_at', 'has_ip', 'dot_count', 'suspicious_keywords', 'special_chars', 'hyphen_in_domain',
                    'entropy', 'numeric_chars', 'vowel_consonant_ratio']

X = df[feature_columns]
df['label_numeric'] = df['label'].apply(lambda label:1 if label == 'bad' else 0)
y = df['label_numeric']

print("Data preparation complete.")
print("\n" + "-" * 25 + "\n")

print("Step 2: Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print("Data split complete.")
print("\n" + "-" * 25 + "\n")

print("Step 3: Training the Stack Classifier...")

base_models = [
    ("xgb", xgb.XGBClassifier(use_label_encoder = False, eval_metric = "logloss", random_state = 42)),
    ("rf", RandomForestClassifier(n_estimators = 50, random_state = 42)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100, ), max_iter = 300, random_state = 42)) 
    ]
meta_model = LogisticRegression()

stacked_model = StackingClassifier(estimators = base_models, final_estimator = meta_model, cv = 3)

stacked_model.fit(X_train, y_train)

print("Model training complete.")
print("\n" + "-"*25 + "\n")

print("Step 4: Evaluating the stacked model...")

predictions = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Stacked Model Accuracy: {accuracy * 100:.2f}%")

print("\n" + "-"*25 + "\n")

print("Step 5: Saving the model...")

model_filename = 'phishing_model.joblib'
joblib.dump(stacked_model, model_filename)
print(f"Model saved to {model_filename}")

print("\n" + "=" * 50)