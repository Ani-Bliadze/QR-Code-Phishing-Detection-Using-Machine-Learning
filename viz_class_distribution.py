"""
Class Distribution Visualization
Generates pie chart and bar chart of legitimate vs phishing URLs
"""

import matplotlib.pyplot as plt


def visualize_class_distribution(data, output_file='04_class_distribution.png'):
    """
    Visualize class distribution in the dataset.
    
    Args:
        data: DataFrame with 'label' column (0=Legitimate, 1=Phishing)
        output_file: Path to save visualization
    """
    class_counts = data['label'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall class distribution
    colors_pie = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = axes[0].pie(class_counts.values, 
                                           labels=['Legitimate', 'Phishing'],
                                           autopct='%1.1f%%',
                                           colors=colors_pie,
                                           startangle=90,
                                           explode=(0.05, 0.05),
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Overall Class Distribution', fontweight='bold', fontsize=13)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    
    # Bar chart of class distribution
    axes[1].bar(['Legitimate', 'Phishing'], class_counts.values, 
               color=colors_pie, edgecolor='black', linewidth=2, width=0.6)
    axes[1].set_ylabel('Count', fontweight='bold', fontsize=11)
    axes[1].set_title('Class Distribution - Bar Chart', fontweight='bold', fontsize=13)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(class_counts.values):
        axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution visualization saved as '{output_file}'")
    plt.show()
