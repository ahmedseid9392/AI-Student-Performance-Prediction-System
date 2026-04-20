"""
Database module for MySQL integration and CRUD operations
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime
import logging
from config import DB_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations"""
    
    def __init__(self):
        self.connection = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database']
            )
            logger.info("Database connection established")
        except Error as e:
            logger.error(f"Error connecting to database: {e}")
            self.create_database()
            self.connect()
    
    def create_database(self):
        """Create database if it doesn't exist"""
        try:
            connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
            connection.commit()
            logger.info(f"Database {DB_CONFIG['database']} created")
        except Error as e:
            logger.error(f"Error creating database: {e}")
    
    def create_tables(self):
        """Create necessary tables"""
        try:
            cursor = self.connection.cursor()
            
            # Students table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100),
                    study_time_hours FLOAT,
                    attendance_percentage FLOAT,
                    previous_grade FLOAT,
                    sleep_hours FLOAT,
                    extracurricular_activities INT,
                    parent_education_level INT,
                    family_income_level INT,
                    internet_access INT,
                    tutoring_sessions INT,
                    gender VARCHAR(10),
                    school_type VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
                    student_id INT,
                    predicted_score FLOAT,
                    performance_category VARCHAR(20),
                    confidence_score FLOAT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(50),
                    accuracy FLOAT,
                    rmse FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            logger.info("Tables created successfully")
        except Error as e:
            logger.error(f"Error creating tables: {e}")
    
    def insert_student(self, student_data):
        """Insert student record"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO students (
                    name, study_time_hours, attendance_percentage, previous_grade,
                    sleep_hours, extracurricular_activities, parent_education_level,
                    family_income_level, internet_access, tutoring_sessions,
                    gender, school_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, student_data)
            self.connection.commit()
            student_id = cursor.lastrowid
            logger.info(f"Student inserted with ID: {student_id}")
            return student_id
        except Error as e:
            logger.error(f"Error inserting student: {e}")
            return None
    
    def insert_prediction(self, student_id, predicted_score, category, confidence):
        """Store prediction result"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO predictions (student_id, predicted_score, performance_category, confidence_score)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (student_id, predicted_score, category, confidence))
            self.connection.commit()
            logger.info(f"Prediction stored for student {student_id}")
        except Error as e:
            logger.error(f"Error storing prediction: {e}")
    
    def log_model_performance(self, model_name, metrics):
        """Log model performance metrics"""
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO model_performance (model_name, accuracy, rmse, precision_score, recall_score)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                model_name, 
                metrics.get('accuracy', 0),
                metrics.get('rmse', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0)
            ))
            self.connection.commit()
            logger.info(f"Model performance logged for {model_name}")
        except Error as e:
            logger.error(f"Error logging model performance: {e}")
    
    def get_all_students(self):
        """Retrieve all students"""
        try:
            query = "SELECT * FROM students ORDER BY created_at DESC"
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            logger.error(f"Error fetching students: {e}")
            return pd.DataFrame()
    
    def get_prediction_history(self):
        """Retrieve prediction history"""
        try:
            query = """
                SELECT s.name, p.predicted_score, p.performance_category, 
                       p.confidence_score, p.prediction_date
                FROM predictions p
                JOIN students s ON p.student_id = s.student_id
                ORDER BY p.prediction_date DESC
            """
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            logger.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")