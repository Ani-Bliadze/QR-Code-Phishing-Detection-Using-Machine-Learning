"""
Feature Importance Visualization
Generates ranking of features by importance in the Random Forest model
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_feature_importance(model, feature_names=None, output_file='02_feature_importance.png'):
    """
    Visualize feature importance from the trained Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names (default: standard URL features)
        output_file: Path to save visualization
    """
    if feature_names is None:
        feature_names = ['url_length', 'num_dots', 'has_at', 'https', 
                        'suspicious_keywords', 'ip_based']
    
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    bars = ax.barh(sorted_features, sorted_importances, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=11)
    ax.set_ylabel('Features', fontweight='bold', fontsize=11)
    ax.set_title('Random Forest - Feature Importance Analysis', fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, importance in zip(bars, sorted_importances):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
               f'{importance:.4f} ({importance*100:.2f}%)', ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance visualization saved as '{output_file}'")
    plt.show()
