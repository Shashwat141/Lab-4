#!/usr/bin/env python3
"""
Data loading and preprocessing for Iris dataset classification.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_and_prepare_data():
    """
    Load the Iris dataset and perform initial preprocessing.
    
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    print("Loading Iris dataset...")
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {iris.target_names}")
    print(f"Features: {iris.feature_names}")
    
    return X, y, iris.feature_names, iris.target_names


def preprocess_data(X, y):
    """
    Preprocess the data: split and scale.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    params = load_params()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )
    
    print(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """Save processed data to files"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save training data
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    
    # Save test data
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'data/scaler.pkl')
    
    print("Processed data saved to data/ directory")


def main():
    """Main data processing pipeline"""
    print("Starting data processing pipeline...")
    
    # Load and prepare data
    X, y, feature_names, target_names = load_and_prepare_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    
    # Save metadata - ensure all values are JSON serializable
    metadata = {
        'feature_names': feature_names,
        'target_names': target_names.tolist(),  # Convert numpy array to list
        'n_samples_train': int(len(X_train)),   # Convert numpy int to Python int
        'n_samples_test': int(len(X_test)),     # Convert numpy int to Python int
        'n_features': int(X_train.shape[1])     # Convert numpy int to Python int
    }
    
    with open('data/metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print("Data processing completed successfully!")


if __name__ == "__main__":
    main()
