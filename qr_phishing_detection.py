"""
This module implements a machine learning solution for detecting phishing URLs
embedded in QR codes using supervised learning techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class QRPhishingDetector:
    """
    A machine learning-based detector for QR code phishing URLs.
    
    This class handles data loading, preprocessing, model training, 
    and evaluation for phishing URL detection.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the QR Phishing Detector.
        
        Args:
            dataset_path (str): Path to the CSV dataset containing URL features
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load the dataset from CSV file.
        
        The dataset should contain the following columns:
        - url_length: Length of the URL
        - num_dots: Number of dots in the URL
        - has_at: Whether the URL contains '@' character (0 or 1)
        - https: Whether the URL uses HTTPS (0 or 1)
        - suspicious_keywords: Number of suspicious keywords found (0 or 1)
        - ip_based: Whether the URL uses IP address instead of domain (0 or 1)
        - label: Target variable (0 = Legitimate, 1 = Phishing)
        """
        try:
            self.data = pd.read_csv(self.dataset_path)
            print(f"✓ Dataset loaded successfully: {len(self.data)} records")
            print(f"✓ Features: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"✗ Error: Dataset file not found at {self.dataset_path}")
            return None
    
    def explore_data(self):
        """
        Perform exploratory data analysis (EDA) on the dataset.
        """
        if self.data is None:
            print("✗ Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        # Basic statistics
        print("\n1. Dataset Overview:")
        print(f"   - Total records: {len(self.data)}")
        print(f"   - Total features: {len(self.data.columns) - 1}")
        
        # Class distribution
        print("\n2. Class Distribution (Label):")
        class_dist = self.data['label'].value_counts()
        print(f"   - Legitimate URLs (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(self.data)*100:.2f}%)")
        print(f"   - Phishing URLs (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(self.data)*100:.2f}%)")
        
        # Feature statistics
        print("\n3. Feature Statistics:")
        print(self.data.describe())
        
        # Missing values
        print("\n4. Missing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("   ✓ No missing values detected")
        else:
            print(missing[missing > 0])
    
    def preprocess_data(self, test_size=0.3, random_state=42):
        """
        Preprocess the data: feature selection, train-test split, and scaling.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.3)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        if self.data is None:
            print("✗ Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Feature selection
        feature_columns = ['url_length', 'num_dots', 'has_at', 'https', 
                          'suspicious_keywords', 'ip_based']
        
        # Verify all features exist
        missing_features = [f for f in feature_columns if f not in self.data.columns]
        if missing_features:
            print(f"✗ Missing features in dataset: {missing_features}")
            return
        
        X = self.data[feature_columns]
        y = self.data['label']
        
        print(f"✓ Features selected: {feature_columns}")
        print(f"✓ Target variable: label (Phishing/Legitimate)")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train-Test Split (Test size: {test_size*100:.0f}%):")
        print(f"   - Training set: {len(self.X_train)} records")
        print(f"   - Testing set: {len(self.X_test)} records")
        
        # Feature scaling (standardization)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Feature scaling applied (StandardScaler)")
        print(f"   - Mean of features set to 0")
        print(f"   - Standard deviation set to 1")
    
    def train_random_forest(self, n_estimators=100, random_state=42):
        """
        Train a Random Forest classifier for phishing detection.
        
        The Random Forest algorithm creates an ensemble of decision trees,
        where each tree votes on the classification. The final decision is
        determined by majority voting, which reduces overfitting and improves
        model robustness.
        
        Args:
            n_estimators (int): Number of trees in the forest (default: 100)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        if self.X_train is None:
            print("✗ Data not preprocessed. Call preprocess_data() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL TRAINING - RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        # Initialize Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Train the model
        print(f"\n✓ Training Random Forest with {n_estimators} trees...")
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model training completed")
        
        # Display feature importance
        feature_names = ['url_length', 'num_dots', 'has_at', 'https', 
                        'suspicious_keywords', 'ip_based']
        importances = self.model.feature_importances_
        
        print("\n✓ Feature Importance:")
        for name, importance in sorted(zip(feature_names, importances), 
                                      key=lambda x: x[1], reverse=True):
            print(f"   - {name}: {importance:.4f}")
    
    def train_logistic_regression(self, random_state=42):
        """
        Train a Logistic Regression classifier for comparison.
        
        Logistic Regression uses the logistic function to model the probability
        that a URL is phishing:
        P(y=1|x) = 1 / (1 + e^(-(w·x + b)))
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        if self.X_train is None:
            print("✗ Data not preprocessed. Call preprocess_data() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL TRAINING - LOGISTIC REGRESSION")
        print("="*60)
        
        # Initialize Logistic Regression
        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        
        print("\n✓ Training Logistic Regression...")
        lr_model.fit(self.X_train, self.y_train)
        print(f"✓ Model training completed")
        
        # Make predictions
        y_pred_lr = lr_model.predict(self.X_test)
        
        # Evaluate
        accuracy_lr = accuracy_score(self.y_test, y_pred_lr)
        print(f"\n✓ Logistic Regression Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set using multiple metrics.
        
        Metrics include:
        - Accuracy: Overall correctness of predictions
        - Precision: True positives / (True positives + False positives)
        - Recall: True positives / (True positives + False negatives)
        - F1-Score: Harmonic mean of precision and recall
        - ROC-AUC: Area under the ROC curve
        """
        if self.model is None:
            print("✗ Model not trained. Call train_random_forest() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\n✓ Classification Metrics:")
        print(f"   - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   - Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   - Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   - F1-Score:  {f1:.4f}")
        print(f"   - ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n✓ Confusion Matrix:")
        print(f"   - True Negatives (TN):  {tn} (Legitimate correctly classified)")
        print(f"   - False Positives (FP): {fp} (Legitimate classified as Phishing)")
        print(f"   - False Negatives (FN): {fn} (Phishing classified as Legitimate) ⚠️")
        print(f"   - True Positives (TP):  {tp} (Phishing correctly classified)")
        
        # Classification Report
        print(f"\n✓ Detailed Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Legitimate', 'Phishing']))
        
        # Critical cybersecurity insights
        print("\n" + "="*60)
        print("CYBERSECURITY INSIGHTS")
        print("="*60)
        print(f"\n⚠️  Critical Metric - False Negative Rate (FNR):")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"   - FNR: {fnr:.4f} ({fnr*100:.2f}%)")
        print(f"   - {fn} phishing URLs were NOT detected (residual risk)")
        
        print(f"\nℹ️  False Positive Rate (FPR):")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"   - FPR: {fpr:.4f} ({fpr*100:.2f}%)")
        print(f"   - {fp} legitimate URLs were incorrectly flagged")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def visualize_results(self, results):
        """
        Create comprehensive visualizations for model performance analysis.
        
        Args:
            results (dict): Dictionary containing evaluation metrics and predictions
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
        fpr, tpr, thresholds = roc_curve(self.y_test, results['y_pred_proba'])
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
        legitimate_probs = results['y_pred_proba'][self.y_test == 0]
        phishing_probs = results['y_pred_proba'][self.y_test == 1]
        
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
        plt.savefig('01_model_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as '01_model_performance_analysis.png'")
        plt.show()
    
    def visualize_feature_importance(self):
        """
        Visualize feature importance from the trained Random Forest model.
        """
        if self.model is None:
            print("✗ Model not trained. Call train_random_forest() first.")
            return
        
        feature_names = ['url_length', 'num_dots', 'has_at', 'https', 
                        'suspicious_keywords', 'ip_based']
        importances = self.model.feature_importances_
        
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
                   f'{importance:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('02_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance visualization saved as '02_feature_importance.png'")
        plt.show()
    
    def visualize_confusion_matrix_detailed(self, results):
        """
        Create a detailed confusion matrix visualization with metrics.
        """
        cm = results['confusion_matrix']
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
        
        metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nSpecificity: {specificity:.4f}'
        fig.text(0.99, 0.01, metrics_text, ha='right', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
        
        plt.tight_layout()
        plt.savefig('03_confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
        print("✓ Detailed confusion matrix saved as '03_confusion_matrix_detailed.png'")
        plt.show()
    
    def visualize_class_distribution(self):
        """
        Visualize class distribution in the dataset.
        """
        if self.data is None:
            print("✗ Data not loaded. Call load_data() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall class distribution
        class_counts = self.data['label'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = axes[0].pie(class_counts.values, 
                                               labels=['Legitimate', 'Phishing'],
                                               autopct='%1.1f%%',
                                               colors=colors_pie,
                                               startangle=90,
                                               explode=(0.05, 0.05),
                                               textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[0].set_title('Overall Class Distribution', fontweight='bold', fontsize=13)
        
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
        plt.savefig('04_class_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Class distribution visualization saved as '04_class_distribution.png'")
        plt.show()
    
    def visualize_feature_correlation(self):
        """
        Visualize correlation between features and target variable.
        """
        if self.data is None:
            print("✗ Data not loaded. Call load_data() first.")
            return
        
        feature_columns = ['url_length', 'num_dots', 'has_at', 'https', 
                          'suspicious_keywords', 'ip_based', 'label']
        correlation_matrix = self.data[feature_columns].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'},
                   linewidths=1, linecolor='gray')
        
        ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig('05_feature_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Feature correlation visualization saved as '05_feature_correlation.png'")
        plt.show()
    
    def visualize_feature_distributions(self):
        """
        Visualize distribution of each feature for legitimate vs phishing URLs.
        """
        if self.data is None:
            print("✗ Data not loaded. Call load_data() first.")
            return
        
        features = ['url_length', 'num_dots', 'has_at', 'https', 'suspicious_keywords', 'ip_based']
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(features):
            legitimate_data = self.data[self.data['label'] == 0][feature]
            phishing_data = self.data[self.data['label'] == 1][feature]
            
            axes[idx].hist(legitimate_data, bins=20, alpha=0.6, label='Legitimate', color='green', edgecolor='black')
            axes[idx].hist(phishing_data, bins=20, alpha=0.6, label='Phishing', color='red', edgecolor='black')
            axes[idx].set_xlabel(feature, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontweight='bold')
            axes[idx].set_title(f'{feature} Distribution', fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.suptitle('Feature Distributions - Legitimate vs Phishing URLs', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('06_feature_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Feature distributions visualization saved as '06_feature_distributions.png'")
        plt.show()
    
    def visualize_all_results(self, results):
        """
        Generate all available visualizations.
        
        Args:
            results (dict): Dictionary containing evaluation metrics and predictions
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*60)
        
        self.visualize_results(results)
        self.visualize_feature_importance()
        self.visualize_confusion_matrix_detailed(results)
        self.visualize_class_distribution()
        self.visualize_feature_correlation()
        self.visualize_feature_distributions()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. 01_model_performance_analysis.png - Main performance metrics")
        print("  2. 02_feature_importance.png - Feature importance ranking")
        print("  3. 03_confusion_matrix_detailed.png - Detailed confusion matrix")
        print("  4. 04_class_distribution.png - Class distribution analysis")
        print("  5. 05_feature_correlation.png - Feature correlation heatmap")
        print("  6. 06_feature_distributions.png - Feature distributions by class")
    
    def predict_url(self, url_features):
        """
        Predict whether a given URL is phishing or legitimate.
        
        Args:
            url_features (dict or array): URL features in the format:
                {'url_length': int, 'num_dots': int, 'has_at': int,
                 'https': int, 'suspicious_keywords': int, 'ip_based': int}
        
        Returns:
            dict: Prediction result with probability
        """
        if self.model is None:
            print("✗ Model not trained.")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(url_features, dict):
            features_df = pd.DataFrame([url_features])
        else:
            features_df = pd.DataFrame([url_features], 
                                      columns=['url_length', 'num_dots', 'has_at', 
                                             'https', 'suspicious_keywords', 'ip_based'])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'confidence': probability[prediction] * 100,
            'probability_legitimate': probability[0] * 100,
            'probability_phishing': probability[1] * 100
        }
        
        return result


def main():
    """
    Main execution function demonstrating the complete workflow.
    """
    print("\n" + "="*60)
    print("QR CODE PHISHING DETECTION - MACHINE LEARNING MODEL")
    print("="*60)
    
    # 1. Initialize detector
    detector = QRPhishingDetector('qr_phishing_urls_dataset.csv')
    
    # 2. Load data
    detector.load_data()
    
    # 3. Explore data
    detector.explore_data()
    
    # 4. Preprocess data
    detector.preprocess_data(test_size=0.3, random_state=42)
    
    # 5. Train model
    detector.train_random_forest(n_estimators=100, random_state=42)
    
    # 6. Evaluate model
    results = detector.evaluate_model()
    
    # 7. Generate comprehensive visualizations
    detector.visualize_all_results(results)
    
    # 8. Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    example_url = {
        'url_length': 75,
        'num_dots': 3,
        'has_at': 0,
        'https': 0,
        'suspicious_keywords': 1,
        'ip_based': 0
    }
    
    prediction = detector.predict_url(example_url)
    print(f"\nExample URL features: {example_url}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.2f}%")
    print(f"Probability of Legitimate: {prediction['probability_legitimate']:.2f}%")
    print(f"Probability of Phishing: {prediction['probability_phishing']:.2f}%")


if __name__ == "__main__":
    main()
