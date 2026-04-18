"""
Utility functions for the Student Performance Prediction System
"""

import logging
import os
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/system_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )

def validate_input(data, field_ranges):
    """Validate input data against field ranges"""
    errors = []
    
    for field, value in data.items():
        if field in field_ranges:
            min_val, max_val = field_ranges[field]
            if value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}")
    
    return errors

def format_performance_category(score):
    """Format performance score into category with color coding"""
    if score >= 85:
        return "Excellent 🟢"
    elif score >= 70:
        return "Good 🟡"
    elif score >= 50:
        return "Average 🟠"
    else:
        return "Poor 🔴"

# Field validation ranges
FIELD_RANGES = {
    'study_time_hours': (0, 24),
    'attendance_percentage': (0, 100),
    'previous_grade': (0, 100),
    'sleep_hours': (0, 24),
    'extracurricular_activities': (0, 1),
    'parent_education_level': (1, 4),
    'family_income_level': (1, 3),
    'internet_access': (0, 1),
    'tutoring_sessions': (0, 20)
}