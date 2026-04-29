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

STUDENT_SCHEMA = {
    'student_id': 'INT AUTO_INCREMENT PRIMARY KEY',
    'name': 'VARCHAR(100) NULL',
    'age': 'INT',
    'gender': 'VARCHAR(20)',
    'school_type': 'VARCHAR(20)',
    'parent_education': 'VARCHAR(50)',
    'study_hours': 'FLOAT',
    'attendance_percentage': 'FLOAT',
    'internet_access': 'VARCHAR(10)',
    'travel_time': 'VARCHAR(20)',
    'extra_activities': 'VARCHAR(10)',
    'study_method': 'VARCHAR(50)',
    'math_score': 'FLOAT',
    'science_score': 'FLOAT',
    'english_score': 'FLOAT',
    'overall_score': 'FLOAT NULL',
    'final_grade': 'VARCHAR(5) NULL',
    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}

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
            student_columns_sql = ",\n                    ".join(
                f"{column} {definition}" for column, definition in STUDENT_SCHEMA.items()
            )
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS students (
                    {student_columns_sql}
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
            self.sync_student_schema()
            logger.info("Tables created successfully")
        except Error as e:
            logger.error(f"Error creating tables: {e}")

    def sync_student_schema(self):
        """Synchronize students table with dataset-based schema."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute("SHOW COLUMNS FROM students")
            existing_columns = {column['Field']: column for column in cursor.fetchall()}

            for column_name, definition in STUDENT_SCHEMA.items():
                if column_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE students ADD COLUMN {column_name} {definition}")
                    logger.info(f"Added missing students column: {column_name}")
                    continue

                if column_name == 'student_id':
                    continue

                current_type = existing_columns[column_name]['Type'].lower()
                expected_type = definition.lower().split()[0]

                if expected_type == 'varchar':
                    needs_modify = not current_type.startswith('varchar')
                elif expected_type == 'timestamp':
                    needs_modify = not current_type.startswith('timestamp')
                else:
                    needs_modify = not current_type.startswith(expected_type)

                if needs_modify:
                    cursor.execute(f"ALTER TABLE students MODIFY COLUMN {column_name} {definition}")
                    logger.info(f"Updated students column definition: {column_name}")

            self.connection.commit()
        except Error as e:
            logger.error(f"Error synchronizing students schema: {e}")
    
    def insert_student(self, student_data):
        """Insert student record"""
        try:
            cursor = self.connection.cursor()
            if isinstance(student_data, dict):
                insertable_columns = [
                    column for column in STUDENT_SCHEMA
                    if column not in {'student_id', 'created_at'} and column in student_data
                ]
                insert_values = [student_data[column] for column in insertable_columns]
            else:
                insertable_columns = [
                    'name', 'age', 'gender', 'school_type', 'parent_education',
                    'study_hours', 'attendance_percentage', 'internet_access',
                    'travel_time', 'extra_activities', 'study_method',
                    'math_score', 'science_score', 'english_score',
                    'overall_score', 'final_grade'
                ]
                insert_values = list(student_data)

            placeholders = ', '.join(['%s'] * len(insertable_columns))
            columns_sql = ', '.join(insertable_columns)
            query = f"INSERT INTO students ({columns_sql}) VALUES ({placeholders})"
            cursor.execute(query, insert_values)
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
                SELECT COALESCE(s.name, CONCAT('Student ', s.student_id)) AS student_name,
                       p.predicted_score, p.performance_category, 
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
