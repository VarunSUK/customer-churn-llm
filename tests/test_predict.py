"""
Unit tests for the prediction module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import joblib
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from predict import ChurnPredictor


class TestChurnPredictor:
    """Test ChurnPredictor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline.named_steps = {
            'preprocessor': Mock(),
            'classifier': Mock()
        }
        
        # Mock preprocessor
        self.mock_preprocessor = Mock()
        self.mock_preprocessor.get_feature_names_out.return_value = [
            'num__tenure', 'num__MonthlyCharges', 'cat__gender_Male', 'cat__Contract_Month-to-month'
        ]
        self.mock_pipeline.named_steps['preprocessor'] = self.mock_preprocessor
        
        # Mock classifier
        self.mock_classifier = Mock()
        self.mock_classifier.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% churn probability
        self.mock_classifier.predict.return_value = np.array([1])  # Churn prediction
        self.mock_pipeline.named_steps['classifier'] = self.mock_classifier
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists, mock_load):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_load.return_value = self.mock_pipeline
        
        predictor = ChurnPredictor("test_model.pkl")
        
        assert predictor.pipeline is not None
        assert predictor.feature_names is not None
        # Should be called twice: once for model, once for feature names
        assert mock_load.call_count == 2
        mock_load.assert_any_call("test_model.pkl")
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_model_file_not_found(self, mock_exists, mock_load):
        """Test model loading when file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            ChurnPredictor("nonexistent_model.pkl")
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_predict_proba(self, mock_exists, mock_load):
        """Test churn probability prediction."""
        mock_exists.return_value = True
        mock_load.return_value = self.mock_pipeline
        
        predictor = ChurnPredictor("test_model.pkl")
        
        # Mock the preprocess_single_row function
        with patch('predict.preprocess_single_row') as mock_preprocess:
            mock_preprocess.return_value = np.array([[1, 2, 3, 4]])
            
            row = {
                'tenure': 12,
                'MonthlyCharges': 50.0,
                'gender': 'Male',
                'Contract': 'Month-to-month'
            }
            
            proba = predictor.predict_proba(row)
            
            assert proba == 0.7
            mock_preprocess.assert_called_once_with(row, self.mock_preprocessor)
            self.mock_classifier.predict_proba.assert_called_once()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_predict(self, mock_exists, mock_load):
        """Test churn class prediction."""
        mock_exists.return_value = True
        mock_load.return_value = self.mock_pipeline
        
        predictor = ChurnPredictor("test_model.pkl")
        
        # Mock the preprocess_single_row function
        with patch('predict.preprocess_single_row') as mock_preprocess:
            mock_preprocess.return_value = np.array([[1, 2, 3, 4]])
            
            row = {
                'tenure': 12,
                'MonthlyCharges': 50.0,
                'gender': 'Male',
                'Contract': 'Month-to-month'
            }
            
            prediction = predictor.predict(row)
            
            assert prediction == 1
            mock_preprocess.assert_called_once_with(row, self.mock_preprocessor)
            self.mock_classifier.predict.assert_called_once()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_predict_without_model(self, mock_exists, mock_load):
        """Test prediction without loaded model."""
        mock_exists.return_value = True
        mock_load.return_value = None
        
        predictor = ChurnPredictor("test_model.pkl")
        predictor.pipeline = None
        
        row = {'tenure': 12}
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict_proba(row)
        
        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict(row)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('predict.ChurnPredictor')
    def test_predict_churn_probability(self, mock_predictor_class):
        """Test convenience function for churn probability prediction."""
        mock_predictor = Mock()
        mock_predictor.predict_proba.return_value = 0.75
        mock_predictor_class.return_value = mock_predictor
        
        from predict import predict_churn_probability
        
        row = {'tenure': 12}
        result = predict_churn_probability(row, "test_model.pkl")
        
        assert result == 0.75
        mock_predictor_class.assert_called_once_with("test_model.pkl")
        mock_predictor.predict_proba.assert_called_once_with(row)


if __name__ == "__main__":
    pytest.main([__file__])
