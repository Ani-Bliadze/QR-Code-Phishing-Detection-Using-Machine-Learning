"""
Generate a realistic QR Code Phishing Detection Dataset
This script creates a synthetic dataset with URL features for training the ML model
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples to generate
n_legitimate = 2000
n_phishing = 1800

# ============================================================================
# LEGITIMATE URLs
# ============================================================================
legitimate_data = {
    'url_length': np.random.normal(60, 15, n_legitimate).astype(int),  # Generally shorter
    'num_dots': np.random.normal(2, 0.8, n_legitimate).astype(int),     # Few subdomains
    'has_at': np.zeros(n_legitimate, dtype=int),                        # No @ symbol
    'https': np.ones(n_legitimate, dtype=int),                          # HTTPS is standard
    'suspicious_keywords': np.zeros(n_legitimate, dtype=int),           # No suspicious words
    'ip_based': np.zeros(n_legitimate, dtype=int),                      # Uses domain names
    'label': np.zeros(n_legitimate, dtype=int)                          # 0 = Legitimate
}

# Add some noise/variation
legitimate_data['url_length'] = np.clip(legitimate_data['url_length'], 20, 200)
legitimate_data['num_dots'] = np.clip(legitimate_data['num_dots'], 1, 8)
legitimate_data['https'] = np.random.choice([0, 1], n_legitimate, p=[0.05, 0.95])  # 95% use HTTPS
legitimate_data['has_at'] = np.random.choice([0, 1], n_legitimate, p=[0.99, 0.01])  # 1% have @

# ============================================================================
# PHISHING URLs
# ============================================================================
phishing_data = {
    'url_length': np.random.normal(110, 35, n_phishing).astype(int),    # Generally longer
    'num_dots': np.random.normal(4.5, 1.5, n_phishing).astype(int),     # More subdomains
    'has_at': np.random.choice([0, 1], n_phishing, p=[0.3, 0.7]),       # Often have @ symbol
    'https': np.random.choice([0, 1], n_phishing, p=[0.6, 0.4]),        # Less likely to use HTTPS
    'suspicious_keywords': np.random.choice([0, 1], n_phishing, p=[0.2, 0.8]),  # Often have keywords
    'ip_based': np.random.choice([0, 1], n_phishing, p=[0.7, 0.3]),     # Some use IP addresses
    'label': np.ones(n_phishing, dtype=int)                             # 1 = Phishing
}

# Ensure values are within reasonable bounds
phishing_data['url_length'] = np.clip(phishing_data['url_length'], 40, 300)
phishing_data['num_dots'] = np.clip(phishing_data['num_dots'], 1, 12)

# ============================================================================
# COMBINE DATASETS
# ============================================================================
legitimate_df = pd.DataFrame(legitimate_data)
phishing_df = pd.DataFrame(phishing_data)

# Combine and shuffle
df = pd.concat([legitimate_df, phishing_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================================
# SAVE DATASET
# ============================================================================
output_file = 'qr_phishing_urls_dataset.csv'
df.to_csv(output_file, index=False)

print("="*60)
print("DATASET GENERATION COMPLETED")
print("="*60)
print(f"\nDataset: {output_file}")
print(f"Total records: {len(df)}")
print(f"Legitimate URLs: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
print(f"Phishing URLs: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")

print("\nFeature Statistics:")
print(df.describe())

print("\nDataset Sample (First 10 rows):")
print(df.head(10))

print("\nColumn Names and Types:")
print(df.dtypes)
