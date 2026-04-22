"""
Model training module for training and evaluating ML models
70% Training / 30% Testing Split
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
            'Random Forest': RandomForestRegressor(random_state=MODEL_CONFIG['random_state'], n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=MODEL_CONFIG['random_state'])
        }
        
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, X, y):
        """Split data into 70% training and 30% testing"""
        logger.info(f"Splitting data: 70% training, 30% testing")
        logger.info(f"Total samples: {len(X)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3,  # 30% for testing
            random_state=MODEL_CONFIG['random_state'],
            shuffle=True
        )
        
        logger.info(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X, y):
        """Train multiple models and evaluate performance using 70/30 split"""
        logger.info("Starting model training and evaluation...")
        logger.info("="*50)
        
        # Split data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions on test set (30%)
            y_pred = model.predict(X_test)
            
            # Calculate metrics on test set
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation on training set only
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            logger.info(f"{name} Results:")
            logger.info(f"  Train samples: {len(X_train)}")
            logger.info(f"  Test samples: {len(X_test)}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R² Score: {r2:.4f}")
            logger.info(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            logger.info("-"*30)
        
        # Select best model (highest R² score on test set)
        self.best_model_name = max(results, key=lambda x: results[x]['r2'])
        self.best_model = results[self.best_model_name]['model']
        self.model_metrics = results[self.best_model_name]
        
        logger.info("="*50)
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        logger.info(f"   Test R² Score: {self.model_metrics['r2']:.4f}")
        logger.info(f"   Test RMSE: {self.model_metrics['rmse']:.4f}")
        logger.info(f"   Training samples: {self.model_metrics['train_size']}")
        logger.info(f"   Testing samples: {self.model_metrics['test_size']}")
        logger.info("="*50)
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        if self.X_train is None:
            raise ValueError("Please call train_and_evaluate first!")
        
        logger.info("Performing hyperparameter tuning...")
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=MODEL_CONFIG['random_state'], n_jobs=-1)
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=MODEL_CONFIG['random_state'])
            
        elif self.best_model_name == 'Decision Tree':
            param_grid = {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8]
            }
            model = DecisionTreeRegressor(random_state=MODEL_CONFIG['random_state'])
            
        else:
            logger.info(f"No hyperparameter tuning configured for {self.best_model_name}")
            return self.best_model
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        logger.info(f"Test R² Score after tuning: {test_r2:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.model_metrics['r2'] = test_r2
        self.model_metrics['rmse'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
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
            'metrics': self.model_metrics,
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0,
            'split_ratio': '70/30'
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
        logger.info(f"Model was trained with {model_data.get('split_ratio', 'unknown')} split")