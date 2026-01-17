"""
Model Performance Analysis Visualization
Generates confusion matrix, metrics, ROC curve, and prediction distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve


def visualize_model_performance(y_test, y_pred, y_pred_proba, results, output_file='01_model_performance_analysis.png'):
    """
    Create comprehensive model performance visualization.
    
    Args:
        y_test: True test labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        results: Dictionary with evaluation metrics
        output_file: Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QR Code Phishing Detection - Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
               xticklabels=['Legitimate', 'Phishing'],
               yticklabels=['Legitimate', 'Phishing'],
               cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Actual Label', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label', fontweight='bold')
    
    # Add confusion matrix values annotation
    tn, fp, fn, tp = cm.ravel()
    axes[0, 0].text(0.5, -0.15, f'TN={tn} | FP={fp}\nFN={fn} | TP={tp}', 
                   transform=axes[0, 0].transAxes, ha='center', fontsize=10)
    
    # 2. Performance Metrics Bar Chart
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1']
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = axes[0, 1].bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Classification Metrics', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontweight='bold')
    axes[0, 1].set_ylim([0.8, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, (name, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=3, 
                   label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
    axes[1, 0].fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    axes[1, 0].set_xlim([-0.02, 1.02])
    axes[1, 0].set_ylim([-0.02, 1.02])
    axes[1, 0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[1, 0].set_ylabel('True Positive Rate', fontweight='bold')
    axes[1, 0].set_title('ROC Curve', fontweight='bold', fontsize=12)
    axes[1, 0].legend(loc="lower right", fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Prediction Probability Distribution
    legitimate_probs = y_pred_proba[y_test == 0]
    phishing_probs = y_pred_proba[y_test == 1]
    
    axes[1, 1].hist(legitimate_probs, bins=30, alpha=0.7, label='Legitimate (actual)', 
                   color='green', edgecolor='black', linewidth=0.5)
    axes[1, 1].hist(phishing_probs, bins=30, alpha=0.7, label='Phishing (actual)', 
                   color='red', edgecolor='black', linewidth=0.5)
    axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[1, 1].set_xlabel('Predicted Probability of Phishing', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold', fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Model performance visualization saved as '{output_file}'")
    plt.show()
