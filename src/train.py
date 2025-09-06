"""
Model training module for Telco Customer Churn prediction.
Trains a LightGBM classifier and saves the complete pipeline.
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
from preprocess import preprocess_data, clean_data


def train_model(data_path: str, model_path: str = "artifacts/model.pkl") -> Pipeline:
    """
    Train a LightGBM model on the Telco Customer Churn dataset.
    
    Args:
        data_path: Path to the CSV data file
        model_path: Path to save the trained model
        
    Returns:
        Trained pipeline (preprocessor + model)
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Separate features and target
    exclude_cols = ['Churn', 'customerID']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    X = df_clean[feature_cols]
    y = df_clean['Churn']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess the training data
    from preprocess import create_preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Create the complete pipeline
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the pipeline
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating model...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names for later use
    feature_names_path = model_path.replace('.pkl', '_features.pkl')
    joblib.dump(pipeline.named_steps['preprocessor'].get_feature_names_out(), feature_names_path)
    print(f"Feature names saved to: {feature_names_path}")
    
    return pipeline


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description='Train LightGBM churn prediction model')
    parser.add_argument('--data', type=str, default='data/telco_churn.csv',
                       help='Path to the training data CSV file')
    parser.add_argument('--model', type=str, default='artifacts/model.pkl',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        return
    
    # Train the model
    try:
        pipeline = train_model(args.data, args.model)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
