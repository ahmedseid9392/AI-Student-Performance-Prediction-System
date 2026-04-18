# test_dataset.py
import pandas as pd
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

# Load your dataset
df = pd.read_csv('C:/Users/Admin/Downloads/Student_Performance.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Prepare data
print("\nPreprocessing data...")
X, y = preprocessor.prepare_data(df)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# Train model
print("\nTraining models...")
trainer = ModelTrainer()
results, X_test, y_test = trainer.train_and_evaluate(X, y)

print(f"\n✅ Best Model: {trainer.best_model_name}")
print(f"R² Score: {trainer.model_metrics['r2']:.4f}")
print(f"RMSE: {trainer.model_metrics['rmse']:.4f}")

# Save model
trainer.save_model()
preprocessor.save_preprocessors()
print("\n✅ Model saved successfully!")