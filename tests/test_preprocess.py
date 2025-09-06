"""
Unit tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from preprocess import clean_data, get_feature_columns, create_preprocessor, preprocess_single_row


class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def test_clean_data_basic(self):
        """Test basic data cleaning."""
        # Create sample data
        df = pd.DataFrame({
            'customerID': ['123', '456'],
            'Churn': ['Yes', 'No'],
            'TotalCharges': ['100.5', '200.0'],
            'tenure': [12, 24],
            'gender': ['Male', 'Female']
        })
        
        cleaned = clean_data(df)
        
        # Check Churn conversion
        assert cleaned['Churn'].dtype == int
        assert cleaned['Churn'].tolist() == [1, 0]
        
        # Check TotalCharges conversion
        assert cleaned['TotalCharges'].dtype == float
        assert cleaned['TotalCharges'].tolist() == [100.5, 200.0]
    
    def test_clean_data_empty_charges(self):
        """Test handling of empty TotalCharges."""
        df = pd.DataFrame({
            'customerID': ['123'],
            'Churn': ['Yes'],
            'TotalCharges': [''],
            'tenure': [12],
            'gender': ['Male']
        })
        
        cleaned = clean_data(df)
        
        # Empty string should become NaN
        assert pd.isna(cleaned['TotalCharges'].iloc[0])


class TestFeatureDetection:
    """Test feature column detection."""
    
    def test_get_feature_columns(self):
        """Test automatic feature column detection."""
        df = pd.DataFrame({
            'customerID': ['123'],
            'Churn': ['Yes'],
            'tenure': [12],  # numeric
            'MonthlyCharges': [50.0],  # numeric
            'gender': ['Male'],  # categorical
            'Contract': ['Month-to-month']  # categorical
        })
        
        numeric_cols, categorical_cols = get_feature_columns(df)
        
        assert 'tenure' in numeric_cols
        assert 'MonthlyCharges' in numeric_cols
        assert 'gender' in categorical_cols
        assert 'Contract' in categorical_cols
        assert 'customerID' not in numeric_cols
        assert 'customerID' not in categorical_cols
        assert 'Churn' not in numeric_cols
        assert 'Churn' not in categorical_cols


class TestPreprocessor:
    """Test preprocessor creation and functionality."""
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        df = pd.DataFrame({
            'tenure': [12, 24, 36],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        preprocessor = create_preprocessor(df)
        
        # Check that preprocessor has the right transformers
        assert 'num' in [name for name, _, _ in preprocessor.transformers]
        assert 'cat' in [name for name, _, _ in preprocessor.transformers]
    
    def test_preprocess_single_row(self):
        """Test single row preprocessing."""
        # Create a preprocessor with sample data
        df = pd.DataFrame({
            'tenure': [12, 24, 36],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        preprocessor = create_preprocessor(df)
        preprocessor.fit(df)
        
        # Test with complete row
        row = {
            'tenure': 18,
            'MonthlyCharges': 60.0,
            'gender': 'Female',
            'Contract': 'One year'
        }
        
        result = preprocess_single_row(row, preprocessor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # Single row
        assert result.shape[1] > 0  # Has features
    
    def test_preprocess_single_row_missing_columns(self):
        """Test single row preprocessing with missing columns."""
        # Create a preprocessor with sample data
        df = pd.DataFrame({
            'tenure': [12, 24, 36],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        preprocessor = create_preprocessor(df)
        preprocessor.fit(df)
        
        # Test with missing columns
        row = {
            'tenure': 18,
            'MonthlyCharges': 60.0
            # Missing gender and Contract
        }
        
        result = preprocess_single_row(row, preprocessor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # Single row
        assert result.shape[1] > 0  # Has features


if __name__ == "__main__":
    pytest.main([__file__])
