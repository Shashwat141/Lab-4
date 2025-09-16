#!/usr/bin/env python3
"""
Model evaluation script for Iris dataset classification.
"""

import numpy as np
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)


def load_test_data():
    """Load preprocessed test data"""
    print("Loading test data...")
    
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_test, y_test


def load_model(model_path='models/model.pkl'):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and calculate metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_score_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_score_weighted': float(f1_weighted),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'test_samples': len(y_test)
    }
    
    print(f"Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    
    return metrics


def save_metrics(metrics, metrics_path='metrics.json'):
    """Save evaluation metrics to JSON file"""
    
    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    
    # Save main metrics file (for DVC)
    with open(metrics_path, 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_score_macro': metrics['f1_score_macro'],
            'f1_score_weighted': metrics['f1_score_weighted']
        }, f, indent=2)
    
    # Save detailed metrics
    with open('metrics/detailed_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path} and metrics/detailed_metrics.json")


def print_detailed_report(metrics):
    """Print detailed evaluation report"""
    print("\n" + "="*50)
    print("DETAILED EVALUATION REPORT")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {metrics['f1_score_macro']:.4f}")
    print(f"  F1-Score (weighted): {metrics['f1_score_weighted']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, (prec, rec, f1) in enumerate(zip(
        metrics['precision_per_class'],
        metrics['recall_per_class'], 
        metrics['f1_per_class']
    )):
        print(f"  Class {i}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    conf_matrix = np.array(metrics['confusion_matrix'])
    for i, row in enumerate(conf_matrix):
        print(f"  Class {i}: {row}")
    
    print("="*50)


def main():
    """Main evaluation pipeline"""
    print("Starting model evaluation pipeline...")
    
    # Load test data and model
    X_test, y_test = load_test_data()
    model = load_model()
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics
    save_metrics(metrics)
    
    # Print detailed report
    print_detailed_report(metrics)
    
    print("Model evaluation completed successfully!")


if __name__ == "__main__":
    main()
