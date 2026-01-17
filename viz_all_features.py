"""
Feature Distributions Visualization
Generates histograms comparing feature distributions between legitimate and phishing URLs
"""

import matplotlib.pyplot as plt


def visualize_feature_distributions(data, features=None, output_file='06_feature_distributions.png'):
    """
    Visualize distribution of each feature for legitimate vs phishing URLs.
    
    Args:
        data: DataFrame with features and 'label' column
        features: List of feature columns (default: standard URL features)
        output_file: Path to save visualization
    """
    if features is None:
        features = ['url_length', 'num_dots', 'has_at', 'https', 'suspicious_keywords', 'ip_based']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        legitimate_data = data[data['label'] == 0][feature]
        phishing_data = data[data['label'] == 1][feature]
        
        axes[idx].hist(legitimate_data, bins=20, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
        axes[idx].hist(phishing_data, bins=20, alpha=0.6, label='Phishing', color='red', edgecolor='black')
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontweight='bold')
        axes[idx].set_title(f'{feature} Distribution', fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Feature Distributions - Legitimate vs Phishing URLs', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Feature distributions visualization saved as '{output_file}'")
    plt.show()
