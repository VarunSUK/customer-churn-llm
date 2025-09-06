"""
Data preprocessing module for Telco Customer Churn dataset.
Handles data cleaning and creates a ColumnTransformer for the ML pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Any


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric, handling empty strings
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Convert Churn to binary (0/1)
    df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)
    
    return df_clean


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Automatically detect numeric vs categorical features.
    
    Args:
        df: Dataframe to analyze
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    # Exclude target and ID columns
    exclude_cols = ['Churn', 'customerID']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Detect numeric columns (int64, float64, and TotalCharges which we converted)
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64'] or col == 'TotalCharges':
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, categorical_cols


def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing the data.
    
    Args:
        df: Training dataframe to fit the preprocessor
        
    Returns:
        Fitted ColumnTransformer
    """
    numeric_cols, categorical_cols = get_feature_columns(df)
    
    # Define transformers
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def preprocess_data(df: pd.DataFrame, preprocessor: ColumnTransformer = None) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw dataframe
        preprocessor: Optional pre-fitted preprocessor
        
    Returns:
        Tuple of (processed_features, fitted_preprocessor)
    """
    # Clean the data
    df_clean = clean_data(df)
    
    # Create or use existing preprocessor
    if preprocessor is None:
        preprocessor = create_preprocessor(df_clean)
    
    # Separate features and target
    exclude_cols = ['Churn', 'customerID']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    X = df_clean[feature_cols]
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Convert back to DataFrame with proper column names
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    
    return X_processed_df, preprocessor


def preprocess_single_row(row: Dict[str, Any], preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocess a single row for prediction.
    Handles missing columns by filling with appropriate defaults.
    
    Args:
        row: Dictionary with feature values
        preprocessor: Fitted preprocessor
        
    Returns:
        Preprocessed feature array
    """
    # Get the expected feature columns from the preprocessor
    numeric_cols, categorical_cols = [], []
    
    # Extract column names from transformers
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            numeric_cols = cols
        elif name == 'cat':
            categorical_cols = cols
    
    # Create a DataFrame with all expected columns
    all_cols = numeric_cols + categorical_cols
    df_row = pd.DataFrame([row])
    
    # Fill missing columns with appropriate defaults
    for col in all_cols:
        if col not in df_row.columns:
            if col in numeric_cols:
                df_row[col] = np.nan  # Will be imputed by SimpleImputer
            else:
                df_row[col] = None    # Will be imputed by SimpleImputer
    
    # Ensure TotalCharges is numeric if present
    if 'TotalCharges' in df_row.columns:
        df_row['TotalCharges'] = pd.to_numeric(df_row['TotalCharges'], errors='coerce')
    
    # Transform through preprocessor
    X_processed = preprocessor.transform(df_row)
    
    return X_processed


if __name__ == "__main__":
    # Test the preprocessing pipeline
    import sys
    import os
    
    # Add src to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Load and test with sample data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'telco_churn.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Test preprocessing
        X_processed, preprocessor = preprocess_data(df)
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Feature names: {list(X_processed.columns)[:10]}...")  # Show first 10
        
        # Show data types
        numeric_cols, categorical_cols = get_feature_columns(df)
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
    else:
        print(f"Data file not found at: {data_path}")
