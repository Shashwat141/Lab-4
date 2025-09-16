#!/usr/bin/env python3
"""
Model training script for Iris dataset classification.
"""

import numpy as np
import joblib
import yaml
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_training_data():
    """Load preprocessed training data"""
    print("Loading training data...")
    
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    
    print(f"Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    return X_train, y_train


def create_model(model_params):
    """
    Create model based on parameters.
    
    Args:
        model_params: Dictionary containing model configuration
        
    Returns:
        Configured model instance
    """
    model_type = model_params['type']
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            random_state=model_params['random_state']
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            C=model_params['C'],
            random_state=model_params['random_state'],
            max_iter=model_params.get('max_iter', 1000)
        )
    elif model_type == 'svm':
        model = SVC(
            C=model_params['C'],
            kernel=model_params['kernel'],
            random_state=model_params['random_state']
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Created {model_type} model with parameters: {model_params}")
    return model


def train_model(model, X_train, y_train):
    """
    Train the model.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training model...")
    
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"Training completed. Training accuracy: {train_accuracy:.4f}")
    
    return model


def save_model(model, model_path='models/model.pkl'):
    """Save trained model"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    
    # Load parameters
    params = load_params()
    
    # Load training data
    X_train, y_train = load_training_data()
    
    # Create and train model
    model = create_model(params['model'])
    trained_model = train_model(model, X_train, y_train)
    
    # Save model
    save_model(trained_model)
    
    # Save model metadata
    model_metadata = {
        'model_type': params['model']['type'],
        'model_params': params['model'],
        'training_samples': len(X_train),
        'features': X_train.shape[1]
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("Model training completed successfully!")


if __name__ == "__main__":
    main()
