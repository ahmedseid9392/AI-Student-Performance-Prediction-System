"""
Configuration file for the Student Performance Prediction System
"""

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Change this to your MySQL password
    'database': 'student_performance_db'
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# File Paths
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/encoders.pkl'
LOG_FILE = 'logs/system.log'

# GUI Configuration
GUI_CONFIG = {
    'window_size': '1200x700',
    'theme': 'clam',
    'font_family': 'Arial',
    'font_size': 10
}

# Feature Columns (based on your dataset)
FEATURES = [
    'age', 'gender', 'school_type', 'parent_education', 'study_hours',
    'attendance_percentage', 'internet_access', 'travel_time', 
    'extra_activities', 'study_method', 'math_score', 'science_score', 'english_score'
]

# Target Column - Use 'overall_score' for regression (numeric prediction)
TARGET = 'overall_score'  # or use 'final_grade' for classification

# Performance Categories for scoring
PERFORMANCE_CATEGORIES = {
    'Excellent': (85, 100),
    'Good': (70, 84),
    'Average': (50, 69),
    'Poor': (0, 49)
}

# Alternative: If you want to use letter grades instead
GRADE_CATEGORIES = {
    'A': (90, 100),
    'B': (80, 89),
    'C': (70, 79),
    'D': (60, 69),
    'E': (50, 59),
    'F': (0, 49)
}

# Field validation ranges
FIELD_RANGES = {
    'age': (10, 25),
    'study_hours': (0, 24),
    'attendance_percentage': (0, 100),
    'math_score': (0, 100),
    'science_score': (0, 100),
    'english_score': (0, 100),
    'travel_time': (0, 3),
    'extra_activities': (0, 1),
    'internet_access': (0, 1)
}