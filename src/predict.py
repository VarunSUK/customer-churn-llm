"""
Prediction module for single row churn probability prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Union
from preprocess import preprocess_single_row


class ChurnPredictor:
    """Churn prediction class for single row predictions."""
    
    def __init__(self, model_path: str = "artifacts/model.pkl"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model pipeline
        """
        self.model_path = model_path
        self.pipeline = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature names."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load the pipeline
        self.pipeline = joblib.load(self.model_path)
        
        # Load feature names
        feature_names_path = self.model_path.replace('.pkl', '_features.pkl')
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
        else:
            # Fallback: get feature names from preprocessor
            try:
                self.feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
            except AttributeError:
                # Older sklearn versions
                self.feature_names = [f'f{i}' for i in range(len(self.pipeline.named_steps['preprocessor'].transformers_))]
    
    def predict_proba(self, row: Dict[str, Any]) -> float:
        """
        Predict churn probability for a single row.
        
        Args:
            row: Dictionary with feature values
            
        Returns:
            Churn probability (float between 0 and 1)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use the preprocessor to handle missing columns
        preprocessor = self.pipeline.named_steps['preprocessor']
        X_processed = preprocess_single_row(row, preprocessor)
        
        # Predict probability using the classifier
        classifier = self.pipeline.named_steps['classifier']
        proba = classifier.predict_proba(X_processed)[0, 1]  # Probability of churn (class 1)
        
        return float(proba)
    
    def predict(self, row: Dict[str, Any]) -> int:
        """
        Predict churn class for a single row.
        
        Args:
            row: Dictionary with feature values
            
        Returns:
            Churn prediction (0 or 1)
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use the preprocessor to handle missing columns
        preprocessor = self.pipeline.named_steps['preprocessor']
        X_processed = preprocess_single_row(row, preprocessor)
        
        # Predict class using the classifier
        classifier = self.pipeline.named_steps['classifier']
        prediction = classifier.predict(X_processed)[0]
        
        return int(prediction)


def predict_churn_probability(row: Dict[str, Any], model_path: str = "artifacts/model.pkl") -> float:
    """
    Convenience function to predict churn probability for a single row.
    
    Args:
        row: Dictionary with feature values
        model_path: Path to the saved model
        
    Returns:
        Churn probability (float between 0 and 1)
    """
    predictor = ChurnPredictor(model_path)
    return predictor.predict_proba(row)


def main():
    """Test the prediction functionality."""
    # Example usage
    sample_row = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    try:
        # Test prediction
        predictor = ChurnPredictor()
        proba = predictor.predict_proba(sample_row)
        prediction = predictor.predict(sample_row)
        
        print(f"Sample row prediction:")
        print(f"Churn probability: {proba:.4f}")
        print(f"Churn prediction: {prediction}")
        
    except FileNotFoundError:
        print("Model not found. Please train the model first using:")
        print("python src/train.py --data data/telco_churn.csv")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
