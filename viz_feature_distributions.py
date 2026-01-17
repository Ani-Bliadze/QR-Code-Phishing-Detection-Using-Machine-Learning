"""
Feature Correlation Visualization
Generates correlation heatmap between features and target variable
"""

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_feature_correlation(data, feature_columns=None, output_file='05_feature_correlation.png'):
    """
    Visualize correlation between features and target variable.
    
    Args:
        data: DataFrame with features and 'label' column
        feature_columns: List of feature columns (default: standard URL features)
        output_file: Path to save visualization
    """
    if feature_columns is None:
        feature_columns = ['url_length', 'num_dots', 'has_at', 'https', 
                          'suspicious_keywords', 'ip_based', 'label']
    
    correlation_matrix = data[feature_columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'},
               linewidths=1, linecolor='gray')
    
    ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=13)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Feature correlation visualization saved as '{output_file}'")
    plt.show()
