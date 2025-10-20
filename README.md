AI-Powered Phishing URL Detector

Overview

This project is an end-to-end web application that uses a machine learning model to detect malicious and phishing URLs. The system is built with a Python backend using Flask to serve a trained XGBoost model, and a minimal HTML/JavaScript frontend for user interaction.

The core of the project is an iteratively developed machine learning model, which has been progressively enhanced to achieve high accuracy through advanced feature engineering and the integration of external APIs.

Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-learn, XGBoost, Pandas

Frontend: HTML, CSS, JavaScript

API Integration: Google Web Risk API

Setup and Installation

Clone the repository:

git clone [https://github.com/samuel-hacks/AI_phishing_detector.git](https://github.com/samuel-hacks/AI_phishing_detector.git)
cd AI_phishing_detector


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate


Install the required dependencies:

pip install -r requirements.txt


(Note: You may need to create a requirements.txt file based on the libraries used: pandas, scikit-learn, xgboost, flask, flask-cors, requests, python-whois).

Add your Google Web Risk API key to app.py and model.py.

Model Development Journey

The model was developed through an iterative process, with each stage introducing more sophisticated features to improve predictive accuracy.

Stage 1: Baseline Model - Logistic Regression

A baseline was established using a simple Logistic Regression model with four basic features extracted from the URL string.

Features: URL Length, Presence of '@' Symbol, Presence of IP Address, Dot Count.

Estimated Accuracy: This initial setup provided a foundational accuracy of approximately 70-75%.

Stage 2: Transition to XGBoost

The model was upgraded to an XGBoost Classifier, a more powerful gradient boosting algorithm, while using the same four basic features. This change in model architecture provided the first significant performance boost.

Model: XGBoost Classifier

Accuracy Achieved: 77.58%

Stage 3: Enhanced Lexical Features

The model's intelligence was improved by adding more nuanced features based on the URL's text content, without relying on external lookups.

New Features: Count of suspicious keywords (e.g., 'login', 'security'), count of special characters, and presence of a hyphen in the domain name.

Accuracy Achieved: 80.50%

Stage 4: Hyperparameter Tuning

A GridSearchCV was performed to find the optimal internal parameters for the XGBoost model. While this is a critical step, it yielded only a marginal gain, indicating the default parameters were already highly effective for this dataset.

Process: Automated search for optimal max_depth, n_estimators, and learning_rate.

Accuracy Achieved: 80.55%

Stage 5: Advanced Lexical Analysis

To better distinguish between human-readable and machine-generated URLs, advanced lexical features were introduced.

New Features: Shannon Entropy (to measure randomness), count of numeric characters, and the vowel-to-consonant ratio of the URL string.

Accuracy Achieved: 84.91%

Stage 6: External API Integration

The final and most impactful improvement was the integration of the Google Web Risk API. A new feature was created to check if a URL exists on Google's real-time blacklist of malicious sites. This provided the model with expert, external knowledge.

New Feature: is_in_gsb_blacklist (1 if flagged by Google, 0 otherwise).

Final Accuracy Achieved: 87.86%

How to Run

Ensure all dependencies are installed and your API key is in place.

Run the Flask application from the terminal:

python app.py


Open a web browser and navigate to http://127.0.0.1:5000 to use the application.
