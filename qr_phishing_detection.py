"""
QR Code Phishing Detection Using Machine Learning
Course: AI and ML for Cybersecurity
Author: Team
Date: January 2026

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
        Create visualizations for model performance analysis.
        
        Args:
            results (dict): Dictionary containing evaluation metrics and predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QR Code Phishing Detection - Model Performance', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. Performance Metrics Bar Chart
        metrics = {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1']
        }
        axes[0, 1].bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(metrics.values()):
            axes[0, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {results["roc_auc"]:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(results['y_pred_proba'][self.y_test == 0], bins=30, 
                       alpha=0.6, label='Legitimate (actual)', color='blue')
        axes[1, 1].hist(results['y_pred_proba'][self.y_test == 1], bins=30, 
                       alpha=0.6, label='Phishing (actual)', color='red')
        axes[1, 1].set_xlabel('Predicted Probability of Being Phishing')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qr_phishing_detection_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'qr_phishing_detection_results.png'")
        plt.show()
    
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
    
    # 7. Visualize results
    detector.visualize_results(results)
    
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
