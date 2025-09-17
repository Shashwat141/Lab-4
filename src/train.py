#!/usr/bin/env python3
"""
Model training script for Iris dataset classification.
This script now includes MLflow for experiment tracking.
"""

import numpy as np
import joblib
import yaml
import json
import os # Make sure os is imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# --- Import MLflow ---
import mlflow
import mlflow.sklearn


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
    else: # Other model types omitted for brevity
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def train_model(model, X_train, y_train):
    """Train the model."""
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path='models/model.pkl'):
    """Save trained model for DVC."""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def log_experiment_to_mlflow(model, params):
    """
    Load test data, evaluate the model, and log everything to MLflow.
    """
    print("Starting MLflow logging...")
    mlflow.set_experiment("Iris Classification")

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

        # --- Conditional Artifact Logging ---
        # Only log the model artifact if not running in a CI environment
        if os.getenv("CI") != "true":
            mlflow.sklearn.log_model(model, "model")
            print("Logged model artifact to MLflow.")
        else:
            print("Skipping model artifact logging in CI environment.")
        
        print("MLflow logging completed successfully!")


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    params = load_params()
    X_train, y_train = load_training_data()
    model = create_model(params['model'])
    trained_model = train_model(model, X_train, y_train)
    save_model(trained_model)
    log_experiment_to_mlflow(trained_model, params)
    print("Model training pipeline completed successfully!")


if __name__ == "__main__":
    main()