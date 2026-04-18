"""
Data preprocessing module for handling data cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
from config import FEATURES, TARGET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False
    
    def clean_data(self, df):
        """Perform initial data cleaning"""
        logger.info("Starting data cleaning...")
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicates")
        
        # Remove student_id column if exists (not needed for prediction)
        if 'student_id' in df.columns:
            df = df.drop('student_id', axis=1)
            logger.info("Removed 'student_id' column")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer_num.fit_transform(df[numeric_cols])
        
        # Fill categorical missing values with mode
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.imputer_cat.fit_transform(df[categorical_cols])
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df
    
    def encode_features(self, df):
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col != TARGET:  # Don't encode target if it's categorical
                if self.is_fitted and col in self.label_encoders:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                logger.info(f"Encoded column: {col}")
        
        return df_encoded
    
    def scale_features(self, X):
        """Scale numerical features"""
        logger.info("Scaling features...")
        
        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def prepare_data(self, df, fit=True):
        """
        Complete data preparation pipeline
        
        Returns:
        - If target column exists: returns (X_scaled, y)
        - If no target column: returns X_scaled
        """
        logger.info("Preparing data for model training...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical variables
        df_encoded = self.encode_features(df_clean)
        
        # Check if target column exists
        target_exists = TARGET in df_encoded.columns
        
        if target_exists:
            # Separate features and target
            # Use FEATURES list if available, otherwise use all columns except target
            if FEATURES and all(f in df_encoded.columns for f in FEATURES):
                X = df_encoded[FEATURES]
            else:
                feature_cols = [col for col in df_encoded.columns if col != TARGET]
                X = df_encoded[feature_cols]
            y = df_encoded[TARGET]
            
            # For classification (if target is categorical), convert to numeric if needed
            if y.dtype == 'object':
                if self.is_fitted and TARGET in self.label_encoders:
                    y = self.label_encoders[TARGET].transform(y)
                else:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    self.label_encoders[TARGET] = le
                logger.info(f"Encoded target column: {TARGET}")
        else:
            # No target column, just return features
            if FEATURES and all(f in df_encoded.columns for f in FEATURES):
                X = df_encoded[FEATURES]
            else:
                X = df_encoded
            y = None
        
        # Scale features
        if fit:
            X_scaled = self.scale_features(X)
            self.is_fitted = True
        else:
            X_scaled = self.scale_features(X)
        
        logger.info(f"Data preparation complete. X shape: {X_scaled.shape}")
        
        if y is not None:
            logger.info(f"Target shape: {y.shape}")
            return X_scaled, y
        else:
            return X_scaled
    
    def prepare_features_only(self, df):
        """Prepare only features (no target) for prediction"""
        result = self.prepare_data(df, fit=False)
        if isinstance(result, tuple):
            return result[0]  # Return only X if tuple
        return result
    
    def save_preprocessors(self, path='models/'):
        """Save preprocessing objects"""
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}/encoders.pkl')
        joblib.dump(self.imputer_num, f'{path}/imputer_num.pkl')
        joblib.dump(self.imputer_cat, f'{path}/imputer_cat.pkl')
        logger.info(f"Preprocessors saved to {path}")
    
    def load_preprocessors(self, path='models/'):
        """Load preprocessing objects"""
        self.scaler = joblib.load(f'{path}/scaler.pkl')
        self.label_encoders = joblib.load(f'{path}/encoders.pkl')
        self.imputer_num = joblib.load(f'{path}/imputer_num.pkl')
        self.imputer_cat = joblib.load(f'{path}/imputer_cat.pkl')
        self.is_fitted = True
        logger.info(f"Preprocessors loaded from {path}")

# Function to generate sample data (updated to match your dataset structure)
def generate_sample_data(n_samples=1000):
    """Generate sample student performance data"""
    np.random.seed(42)
    
    data = {
        'student_id': range(1, n_samples + 1),
        'age': np.random.choice([14, 15, 16, 17, 18], n_samples),
        'gender': np.random.choice(['male', 'female', 'other'], n_samples),
        'school_type': np.random.choice(['public', 'private'], n_samples),
        'parent_education': np.random.choice(['high school', 'graduate', 'post graduate', 'masters'], n_samples),
        'study_hours': np.random.normal(5, 2, n_samples).clip(0, 15),
        'attendance_percentage': np.random.normal(85, 10, n_samples).clip(40, 100),
        'internet_access': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'travel_time': np.random.choice([0, 1, 2, 3], n_samples),  # 0: <30min, 1: 30-60min, etc.
        'extra_activities': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'study_method': np.random.choice(['self', 'group', 'tutor', 'online'], n_samples),
        'math_score': np.random.normal(65, 20, n_samples).clip(0, 100),
        'science_score': np.random.normal(65, 20, n_samples).clip(0, 100),
        'english_score': np.random.normal(65, 20, n_samples).clip(0, 100),
        'overall_score': np.random.normal(65, 20, n_samples).clip(0, 100),
        'final_grade': np.random.choice(['a', 'b', 'c', 'd', 'e', 'f'], n_samples)
    }
    
    df = pd.DataFrame(data)
    return df