#!/usr/bin/env python3
"""
Model training script for Iris dataset classification.
This script now includes MLflow for experiment tracking.
"""

import numpy as np
import joblib
import yaml
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Import MLflow and Pathlib ---
import mlflow
import mlflow.sklearn
import pathlib # Add this import


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
    print(f"Training data loaded: {X_train.shape[0]} samples")
    return X_train, y_train


def create_model(model_params):
    """Create model based on parameters."""
    model_type = model_params['type']
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            random_state=model_params['random_state']
        )
    # ... (other model types are the same)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Created {model_type} model.")
    return model


def train_model(model, X_train, y_train):
    """Train the model."""
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training completed.")
    return model


def save_model(model, model_path='models/model.pkl'):
    """Save trained model for DVC."""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def log_experiment_to_mlflow(model, params, X_train, y_train):
    """
    Load test data, evaluate the model, and log everything to MLflow.
    """
    print("Starting MLflow logging...")
    
    # --- Explicitly set the MLflow tracking URI ---
    # This ensures the script saves data to the local 'mlruns' folder
    '''mlflow.set_tracking_uri(pathlib.Path("mlruns").resolve().as_uri())

    mlflow.set_experiment("Iris Classification")'''

    with mlflow.start_run():
        mlflow.log_params(params['model'])
        print("Logged hyperparameters to MLflow.")

        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score_weighted": f1_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)
        print("Logged evaluation metrics to MLflow.")

        mlflow.sklearn.log_model(model, "model")
        print("Logged model artifact to MLflow.")
        
        print("MLflow logging completed successfully!")


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    params = load_params()
    X_train, y_train = load_training_data()
    model = create_model(params['model'])
    trained_model = train_model(model, X_train, y_train)
    save_model(trained_model)
    log_experiment_to_mlflow(trained_model, params, X_train, y_train)
    print("Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()