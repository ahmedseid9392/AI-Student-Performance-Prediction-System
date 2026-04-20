"""
Student Performance Prediction Testing Script
Run this to test the prediction system with your dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    """Simple and effective student performance predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
    
    def load_and_prepare_data(self, csv_path):
        """Load and prepare the dataset"""
        print("="*60)
        print("📊 STUDENT PERFORMANCE PREDICTION SYSTEM")
        print("="*60)
        
        # Load data
        print(f"\n1. Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display column info
        print(f"\n📋 Available columns:")
        for col in df.columns:
            print(f"   • {col}")
        
        # Check for target column
        target_candidates = ['overall_score', 'performance_score', 'score', 'final_grade', 'grade']
        target_col = None
        
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        # If not found, ask user
        if target_col is None:
            print(f"\n⚠️  Target column not found automatically.")
            print(f"Available numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
            target_col = input(f"\nEnter the target column name (the score to predict): ").strip()
            
            if target_col not in df.columns:
                print(f"❌ Column '{target_col}' not found!")
                return None, None
        
        print(f"\n🎯 Target column: {target_col}")
        
        # Prepare features (exclude non-numeric and target)
        exclude_cols = [target_col, 'student_id', 'name', 'id', 'Student_ID']
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
        
        print(f"\n📊 Feature columns ({len(feature_cols)}):")
        for col in feature_cols[:10]:  # Show first 10
            print(f"   • {col}")
        if len(feature_cols) > 10:
            print(f"   ... and {len(feature_cols)-10} more")
        
        # Prepare feature matrix
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        print(f"\n🔄 Encoding categorical variables...")
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"   • Encoded: {col}")
        
        # Handle missing values
        print(f"\n🛠️  Handling missing values...")
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Scale features
        print(f"📏 Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_columns = feature_cols
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """Train the prediction model"""
        print(f"\n🤖 Training Model...")
        print("-"*40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n📈 Model Performance:")
        print(f"   Training RMSE: {train_rmse:.4f}")
        print(f"   Testing RMSE:  {test_rmse:.4f}")
        print(f"   Training R²:   {train_r2:.4f}")
        print(f"   Testing R²:    {test_r2:.4f}")
        print(f"   MAE:           {mae:.4f}")
        
        # Feature importance
        print(f"\n⭐ Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']:<20} : {row['importance']:.4f}")
        
        return test_r2
    
    def predict_single_student(self):
        """Make prediction for a single student interactively"""
        print(f"\n" + "="*60)
        print("🎯 MAKE A PREDICTION")
        print("="*60)
        
        print(f"\n📝 Enter student information:")
        print("-"*40)
        
        student_data = {}
        
        # Get input for each feature
        for col in self.feature_columns[:15]:  # Limit to first 15 for simplicity
            if col in self.label_encoders:
                # Categorical feature
                options = self.label_encoders[col].classes_
                print(f"\n{col}:")
                for i, opt in enumerate(options):
                    print(f"   {i+1}. {opt}")
                choice = int(input(f"Choose (1-{len(options)}): ")) - 1
                student_data[col] = options[choice]
            else:
                # Numeric feature
                if 'score' in col.lower() or 'grade' in col.lower():
                    default = 70
                elif 'hour' in col.lower() or 'time' in col.lower():
                    default = 5
                elif 'attendance' in col.lower():
                    default = 85
                else:
                    default = 0
                
                value = input(f"{col} (default {default}): ").strip()
                student_data[col] = float(value) if value else default
        
        # Create dataframe
        input_df = pd.DataFrame([student_data])
        
        # Encode categorical
        for col, le in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Ensure correct column order
        input_df = input_df[self.feature_columns]
        
        # Scale
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        
        # Display result
        print(f"\n" + "="*60)
        print("📊 PREDICTION RESULT")
        print("="*60)
        print(f"\n🎯 Predicted Score: {prediction:.2f}/100")
        
        # Performance category
        if prediction >= 85:
            category = "🏆 EXCELLENT"
            color = "🌟"
            advice = "Outstanding! Keep up the great work!"
        elif prediction >= 70:
            category = "👍 GOOD"
            color = "✅"
            advice = "Good performance! Aim for excellence!"
        elif prediction >= 50:
            category = "📚 AVERAGE"
            color = "⚠️"
            advice = "Good base! Focus on improving weak areas."
        else:
            category = "⚠️ NEEDS IMPROVEMENT"
            color = "❌"
            advice = "Need significant improvement. Seek help and study more!"
        
        print(f"🏅 Category: {category}")
        print(f"\n💡 Advice: {advice}")
        print("="*60)
        
        return prediction
    
    def batch_predict(self, test_data_path):
        """Make predictions for multiple students from CSV"""
        print(f"\n📊 Batch Prediction Mode")
        print("="*60)
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        print(f"\nLoaded {len(test_df)} students for prediction")
        
        # Prepare features
        X_test = test_df[self.feature_columns].copy()
        
        # Encode categorical
        for col, le in self.label_encoders.items():
            if col in X_test.columns:
                X_test[col] = le.transform(X_test[col].astype(str))
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in X_test.columns:
                X_test[col] = 0
        
        # Ensure correct order
        X_test = X_test[self.feature_columns]
        
        # Scale
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        predictions = self.model.predict(X_test_scaled)
        
        # Add predictions to dataframe
        test_df['Predicted_Score'] = predictions
        
        # Add categories
        def get_category(score):
            if score >= 85:
                return 'Excellent'
            elif score >= 70:
                return 'Good'
            elif score >= 50:
                return 'Average'
            else:
                return 'Poor'
        
        test_df['Performance_Category'] = test_df['Predicted_Score'].apply(get_category)
        
        # Save results
        output_path = test_data_path.replace('.csv', '_predictions.csv')
        test_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Predictions saved to: {output_path}")
        print(f"\n📊 Summary:")
        print(test_df['Performance_Category'].value_counts())
        
        return test_df
    
    def save_model(self, path='models/'):
        """Save trained model"""
        import os
        os.makedirs(path, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, f'{path}/performance_predictor.pkl')
        print(f"\n✅ Model saved to {path}/performance_predictor.pkl")
    
    def load_model(self, path='models/performance_predictor.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"✅ Model loaded from {path}")


def main():
    """Main function to run the testing script"""
    print("\n" + "🎓" * 30)
    print("STUDENT PERFORMANCE PREDICTION TESTING SYSTEM")
    print("🎓" * 30 + "\n")
    
    # Initialize predictor
    predictor = StudentPerformancePredictor()
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train model on dataset")
        print("2. Predict single student (interactive)")
        print("3. Batch predict from CSV file")
        print("4. Load saved model")
        print("5. Test with sample data")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Train model
            csv_path = input("Enter path to CSV dataset: ").strip()
            if not csv_path:
                csv_path = "C:/Users/Admin/Downloads/Student_Performance.csv"
            
            try:
                X, y = predictor.load_and_prepare_data(csv_path)
                if X is not None:
                    r2 = predictor.train_model(X, y)
                    predictor.save_model()
                    
                    # Ask if user wants to test prediction
                    test = input("\n🎯 Make a test prediction? (yes/no): ").strip().lower()
                    if test == 'yes':
                        predictor.predict_single_student()
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            # Predict single student
            try:
                predictor.predict_single_student()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            # Batch predict
            csv_path = input("Enter path to test CSV file: ").strip()
            try:
                predictor.batch_predict(csv_path)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '4':
            # Load model
            model_path = input("Enter model path (default: models/performance_predictor.pkl): ").strip()
            if not model_path:
                model_path = 'models/performance_predictor.pkl'
            try:
                predictor.load_model(model_path)
                print("✅ Model loaded successfully!")
                
                # Test prediction
                test = input("\n🎯 Make a prediction? (yes/no): ").strip().lower()
                if test == 'yes':
                    predictor.predict_single_student()
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        
        elif choice == '5':
            # Test with sample data
            print("\n📊 Creating sample dataset for testing...")
            from data_preprocessing import generate_sample_data
            sample_df = generate_sample_data(100)
            sample_path = "sample_test_data.csv"
            sample_df.to_csv(sample_path, index=False)
            print(f"✅ Sample dataset created: {sample_path}")
            
            # Train on sample data
            X, y = predictor.load_and_prepare_data(sample_path)
            if X is not None:
                predictor.train_model(X, y)
                predictor.save_model()
                
                # Make a test prediction
                print("\n🎯 Testing prediction with sample student...")
                predictor.predict_single_student()
        
        elif choice == '6':
            print("\n👋 Goodbye! Thanks for using the prediction system!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


# Quick prediction function for testing
def quick_predict():
    """Quick prediction without training (uses default values)"""
    print("\n" + "="*60)
    print("QUICK PREDICTION (Using default model)")
    print("="*60)
    
    # Simple formula for quick testing
    print("\n📝 Enter student information:")
    study_hours = float(input("Study hours per day: ") or 5)
    attendance = float(input("Attendance percentage: ") or 85)
    previous_grade = float(input("Previous grade (0-100): ") or 70)
    math_score = float(input("Math score (0-100): ") or 65)
    science_score = float(input("Science score (0-100): ") or 65)
    english_score = float(input("English score (0-100): ") or 65)
    
    # Simple prediction formula
    predicted_score = (
        study_hours * 3 +
        attendance * 0.5 +
        previous_grade * 0.4 +
        math_score * 0.3 +
        science_score * 0.3 +
        english_score * 0.3
    ) / 5.5  # Normalize
    
    predicted_score = min(100, max(0, predicted_score))
    
    print(f"\n📊 Predicted Score: {predicted_score:.2f}/100")
    
    if predicted_score >= 85:
        print("🏆 Category: EXCELLENT - Outstanding performance!")
    elif predicted_score >= 70:
        print("👍 Category: GOOD - Keep up the good work!")
    elif predicted_score >= 50:
        print("📚 Category: AVERAGE - Room for improvement!")
    else:
        print("⚠️ Category: NEEDS IMPROVEMENT - Seek help and study more!")


if __name__ == "__main__":
    # Check if we should run quick prediction or main menu
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_predict()
    else:
        main()