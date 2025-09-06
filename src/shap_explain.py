"""
SHAP explanation module for churn prediction.
Provides feature importance explanations for individual predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import shap
from typing import Dict, Any, List, Tuple
from predict import ChurnPredictor
from preprocess import preprocess_single_row


class ChurnExplainer:
    """SHAP explainer for churn predictions."""
    
    def __init__(self, model_path: str = "artifacts/model.pkl"):
        """
        Initialize the explainer with a trained model.
        
        Args:
            model_path: Path to the saved model pipeline
        """
        self.model_path = model_path
        self.predictor = ChurnPredictor(model_path)
        self.explainer = None
        self.feature_names = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup the SHAP explainer."""
        if self.predictor.pipeline is None:
            raise ValueError("Model not loaded.")
        
        # Get the LightGBM model from the pipeline
        lgb_model = self.predictor.pipeline.named_steps['classifier']
        
        # Get the preprocessor
        preprocessor = self.predictor.pipeline.named_steps['preprocessor']
        
        # Create a wrapper function for SHAP
        def model_predict(X):
            return self.predictor.pipeline.predict_proba(X)[:, 1]
        
        # Create TreeExplainer for LightGBM
        self.explainer = shap.TreeExplainer(lgb_model)
        
        # Get feature names
        self.feature_names = self.predictor.feature_names
    
    def explain_prediction(self, row: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            row: Dictionary with feature values
            top_k: Number of top features to return
            
        Returns:
            List of dictionaries with feature names and their SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not setup. Call _setup_explainer() first.")
        
        # Use the preprocessor to handle missing columns
        preprocessor = self.predictor.pipeline.named_steps['preprocessor']
        X_transformed = preprocess_single_row(row, preprocessor)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_transformed)
        
        # For binary classification, use the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        
        # Create feature importance list
        feature_importance = []
        for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values[0])):
            feature_importance.append({
                'feature': feature_name,
                'impact': float(shap_value)
            })
        
        # Sort by absolute impact and return top_k
        feature_importance.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return feature_importance[:top_k]
    
    def get_feature_importance_global(self, X_sample: pd.DataFrame = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get global feature importance using SHAP values.
        
        Args:
            X_sample: Sample data to compute global importance (optional)
            top_k: Number of top features to return
            
        Returns:
            List of dictionaries with feature names and their mean absolute SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not setup. Call _setup_explainer() first.")
        
        # If no sample provided, create a small sample from the training data
        if X_sample is None:
            # This is a simplified approach - in practice, you'd want to use actual training data
            print("Warning: No sample data provided. Using model's feature importance instead.")
            lgb_model = self.predictor.pipeline.named_steps['classifier']
            importance = lgb_model.feature_importances_
            feature_names = self.feature_names
            
            feature_importance = []
            for feature_name, imp in zip(feature_names, importance):
                feature_importance.append({
                    'feature': feature_name,
                    'importance': float(imp)
                })
            
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            return feature_importance[:top_k]
        
        # Transform the sample data
        preprocessor = self.predictor.pipeline.named_steps['preprocessor']
        X_transformed = preprocessor.transform(X_sample)
        
        # Get SHAP values for the sample
        shap_values = self.explainer.shap_values(X_transformed)
        
        # For binary classification, use the positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        feature_importance = []
        for feature_name, importance in zip(feature_names, mean_abs_shap):
            feature_importance.append({
                'feature': feature_name,
                'importance': float(importance)
            })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        return feature_importance[:top_k]


def explain_churn_prediction(row: Dict[str, Any], model_path: str = "artifacts/model.pkl", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to explain a churn prediction.
    
    Args:
        row: Dictionary with feature values
        model_path: Path to the saved model
        top_k: Number of top features to return
        
    Returns:
        List of dictionaries with feature names and their SHAP values
    """
    explainer = ChurnExplainer(model_path)
    return explainer.explain_prediction(row, top_k)


def main():
    """Test the SHAP explanation functionality."""
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
        # Test explanation
        explainer = ChurnExplainer()
        explanations = explainer.explain_prediction(sample_row, top_k=5)
        
        print(f"Top 5 features contributing to churn prediction:")
        for i, exp in enumerate(explanations, 1):
            print(f"{i}. {exp['feature']}: {exp['impact']:.4f}")
        
        # Test prediction probability
        proba = explainer.predictor.predict_proba(sample_row)
        print(f"\nChurn probability: {proba:.4f}")
        
    except FileNotFoundError:
        print("Model not found. Please train the model first using:")
        print("python src/train.py --data data/telco_churn.csv")
    except Exception as e:
        print(f"Error during explanation: {e}")


if __name__ == "__main__":
    main()
