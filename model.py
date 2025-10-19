import pandas as pd
import re
from sklearn.model_selection import train_test_split

print("Step 1: Loading and preparing data...")

csv_file_name = "phishing_site_urls.csv"

try:
    df = pd.read_csv(csv_file_name)
    df = df.rename(columns = {"URL": "url", "Label": "label"})

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

df["url_length"] = df['url'].apply(get_url_length)
df['has_at'] = df['url'].apply(has_at_symbol)
df['has_ip'] = df['url'].apply(has_ip_address)
df['dot_count'] = df['url'].apply(get_dot_count)

feature_columns = ['url_length', 'has_at', 'has_ip', 'dot_count']

X = df[feature_columns]
df['label_numeric'] = df['label'].apply(lambda label:1 if label == 'bad' else 0)
y = df['label_numeric']

print("Data preparation complete.")
print("\n" + "-"*25 + "\n")

print("Step 2: Splitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

print("Data split complete.")
print(f"Shape of training features (X_train): {X_train.shape}")
print(f"Shape of testing features (X_test):  {X_test.shape}")
print(f"Shape of training labels (y_train): {y_train.shape}")
print(f"Shape of testing labels (y_test):  {y_test.shape}")