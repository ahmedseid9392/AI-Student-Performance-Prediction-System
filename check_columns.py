# check_columns.py
import pandas as pd

# Load your CSV file
df = pd.read_csv('C:/Users/Admin/Downloads/Student_Performance.csv')
print("Columns in your dataset:")
print(list(df.columns))
print("\nFirst few rows:")
print(df.head())