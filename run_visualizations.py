"""
Unified Visualization Runner
Generates all visualization files in one command
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Import individual visualization modules
from viz_model_performance import visualize_model_performance
from viz_feature_importance import visualize_feature_importance
from viz_confusion_matrix import visualize_confusion_matrix_detailed
from viz_class_distribution import visualize_class_distribution
from viz_feature_distributions import visualize_feature_distributions


def run_all_visualizations(dataset_path='qr_phishing_urls_dataset.csv', 
                           test_size=0.3, random_state=42, n_estimators=100):
    """
    Run all visualizations on the dataset.
    
    Args:
        dataset_path: Path to the dataset CSV
        test_size: Test set proportion
        random_state: Random seed
        n_estimators: Number of trees in Random Forest
    """
    print("\n" + "="*70)
    print("QR CODE PHISHING DETECTION - COMPREHENSIVE VISUALIZATION SUITE")
    print("="*70)
    
    # Load data
    print("\n[1/7] Loading dataset...")
    data = pd.read_csv(dataset_path)
    print(f"✓ Dataset loaded: {len(data)} records")
    
    # Prepare features
    print("\n[2/7] Preparing features and labels...")
    feature_columns = ['url_length', 'num_dots', 'has_at', 'https', 
                      'suspicious_keywords', 'ip_based']
    X = data[feature_columns]
    y = data['label']
    
    # Train-test split
    print("\n[3/7] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    print("\n[4/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n[5/7] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=15,
        min_samples_split=5
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\n[6/7] Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n✓ Model Performance:")
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1']:.4f}")
    print(f"   ROC-AUC:   {results['roc_auc']:.4f}")
    
    # Generate visualizations
    print("\n[7/7] Generating all visualizations...")
    
    print("\n  → Generating model performance analysis...")
    visualize_model_performance(y_test, y_pred, y_pred_proba, results)
    
    print("\n  → Generating feature importance chart...")
    visualize_feature_importance(model, feature_columns)
    
    print("\n  → Generating confusion matrix...")
    visualize_confusion_matrix_detailed(results['confusion_matrix'])
    
    print("\n  → Generating class distribution...")
    visualize_class_distribution(data)
    
    print("\n  → Generating feature distributions...")
    visualize_feature_distributions(data, feature_columns)
    
    # Print summary
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  ✓ 01_model_performance_analysis.png")
    print("  ✓ 02_feature_importance.png")
    print("  ✓ 03_confusion_matrix_detailed.png")
    print("  ✓ 04_class_distribution.png")
    print("  ✓ 05_feature_distributions.png")
    print("\nAll visualizations have been saved to the current directory!")
    print("="*70)
    
    return data, model, results


if __name__ == "__main__":
    run_all_visualizations()
