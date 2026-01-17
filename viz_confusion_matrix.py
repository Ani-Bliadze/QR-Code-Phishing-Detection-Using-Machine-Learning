"""
Detailed Confusion Matrix Visualization
Generates confusion matrix with detailed metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_confusion_matrix_detailed(cm, output_file='03_confusion_matrix_detailed.png'):
    """
    Create a detailed confusion matrix visualization with metrics.
    
    Args:
        cm: Confusion matrix (2x2 numpy array)
        output_file: Path to save visualization
    """
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create detailed confusion matrix
    sns.heatmap(cm, annot=False, cmap='RdYlGn', ax=ax, cbar=False,
               xticklabels=['Legitimate', 'Phishing'],
               yticklabels=['Legitimate', 'Phishing'])
    
    # Add custom annotations with metrics
    ax.text(0.5, 0.25, f'TN\n{tn}', ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
    ax.text(1.5, 0.25, f'FP\n{fp}', ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')
    ax.text(0.5, 1.25, f'FN\n{fn}', ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')
    ax.text(1.5, 1.25, f'TP\n{tp}', ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')
    
    ax.set_ylabel('Actual', fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
    ax.set_title('Detailed Confusion Matrix\n(TN=True Negative, FP=False Positive, FN=False Negative, TP=True Positive)',
                fontweight='bold', fontsize=13)
    
    # Add metrics summary
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics_text = (f'Accuracy: {accuracy:.4f}\n'
                   f'Precision: {precision:.4f}\n'
                   f'Recall: {recall:.4f}\n'
                   f'Specificity: {specificity:.4f}\n'
                   f'False Negative Rate: {fnr:.4f}\n'
                   f'False Positive Rate: {fpr:.4f}')
    fig.text(0.99, 0.01, metrics_text, ha='right', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed confusion matrix saved as '{output_file}'")
    plt.show()
