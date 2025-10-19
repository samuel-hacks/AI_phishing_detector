import pandas as pd
import re
from urllib.parse import urlparse
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

dataset_slug = 'taruntiwarihp/phishing-site-urls'
csv_file_name = 'phishing_site_urls.csv'

if not os.path.exists(csv_file_name):
    print(f"Dataset '{csv_file_name}' not found. Downloading from Kaggle...")
    
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path='.', unzip=False)
    
    zip_file_name = dataset_slug.split('/')[1] + '.zip'
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(zip_file_name)
    
    print("Download and unzip complete!")

else:
    print(f"Dataset '{csv_file_name}' already exists. Skipping download.")

print("\n" + "-" * 25 + "\n")

print("Step 1: Loading local dataset...")

try:
    df = pd.read_csv(csv_file_name)
    df = df.rename(columns={'URL': 'url', 'Label': 'label'})
    
    print("Dataset loaded successfully!")
    print("Here's a preview of the data:")
    print(df.head())
    print("\n" + "-" * 25 + "\n")

except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found.")
    exit()

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Step 2: Defining feature extraction functions...")

def get_url_length(url):
    return len(str(url))

def has_at_symbol(url):
    return 1 if '@' in str(url) else 0

def has_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])',
        str(url)
    )
    return 1 if match else 0

def get_dot_count(url):
    return str(url).count('.')

print("Functions defined successfully.")
print("\n" + "-" * 25 + "\n")

print("Step 3: Applying functions to engineer features...")

df['url_length'] = df['url'].apply(get_url_length)
df['has_at'] = df['url'].apply(has_at_symbol)
df['has_ip'] = df['url'].apply(has_ip_address)
df['dot_count'] = df['url'].apply(get_dot_count)

print("Feature engineering complete!")
print("Here's the data with the new feature columns:")
print(df.head())
print("\n" + "*" * 50 + "\n")

print("Step 4: Creating the final feature matrix (X) and label vector (y)...")

feature_columns = ['url_length', 'has_at', 'has_ip', 'dot_count']
X = df[feature_columns]

df['label_numeric'] = df['label'].apply(lambda label: 1 if label == 'bad' else 0)
y = df['label_numeric']

print("Final data is ready for model training!")
print("\nFeature Matrix X (first 5 rows):")

print(X.head())
print("\nLabel Vector y (first 5 rows):")
print(y.head())

print("\n" + "=" * 50)
print("Data preparation is complete!")
print("=" * 50)