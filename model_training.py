"""
Model training module for training and evaluating ML models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from config import MODEL_CONFIG, PERFORMANCE_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles model training, evaluation, and selection"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=MODEL_CONFIG['random_state']),
            'Random Forest': RandomForestRegressor(random_state=MODEL_CONFIG['random_state']),
            'Gradient Boosting': GradientBoostingRegressor(random_state=MODEL_CONFIG['random_state'])
        }
        
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
    
    def train_and_evaluate(self, X, y):
        """Train multiple models and evaluate performance"""
        logger.info("Starting model training and evaluation...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state']
        )
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=MODEL_CONFIG['cv_folds'], 
                                        scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, CV R²: {cv_scores.mean():.4f}")
        
        # Select best model (highest R² score)
        self.best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[self.best_model_name]['model']
        self.model_metrics = results[self.best_model_name]
        
        logger.info(f"Best model: {self.best_model_name} with R²: {self.model_metrics['r2']:.4f}")
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        logger.info("Performing hyperparameter tuning...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            logger.info(f"No hyperparameter tuning configured for {self.best_model_name}")
            return self.best_model
        
        grid_search = GridSearchCV(
            self.best_model, 
            param_grid, 
            cv=MODEL_CONFIG['cv_folds'],
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_and_evaluate first.")
        
        predictions = self.best_model.predict(X)
        return predictions
    
    def get_performance_category(self, score):
        """Convert numerical score to performance category"""
        for category, (min_score, max_score) in PERFORMANCE_CATEGORIES.items():
            if min_score <= score <= max_score:
                return category
        return "Unknown"
    
    def save_model(self, path='models/'):
        """Save the trained model"""
        import os
        os.makedirs(path, exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'metrics': self.model_metrics
        }
        
        joblib.dump(model_data, f'{path}/best_model.pkl')
        logger.info(f"Model saved to {path}/best_model.pkl")
    
    def load_model(self, path='models/'):
        """Load a saved model"""
        model_data = joblib.load(f'{path}/best_model.pkl')
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.model_metrics = model_data['metrics']
        logger.info(f"Model loaded from {path}/best_model.pkl")