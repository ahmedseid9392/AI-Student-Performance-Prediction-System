# fix_dataset.py
import pandas as pd
import os

def fix_dataset(csv_path):
    """Auto-detect and fix the dataset column names"""
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"\nOriginal columns: {list(df.columns)}")
    
    # Common target column names
    target_variations = [
        'Performance', 'performance', 'Score', 'score', 
        'Grade', 'grade', 'Marks', 'marks', 'Result', 'result',
        'GPA', 'gpa', 'Percentage', 'percentage'
    ]
    
    # Find the target column
    target_col = None
    for col in target_variations:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        print(f"\nFound target column: '{target_col}'")
        # Rename to performance_score
        df = df.rename(columns={target_col: 'performance_score'})
        print(f"Renamed to: 'performance_score'")
        
        # Save the fixed dataset
        output_path = csv_path.replace('.csv', '_fixed.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✅ Fixed dataset saved to: {output_path}")
        print(f"\nPlease use this file in the application: {output_path}")
        
        return output_path
    else:
        print("\n❌ Could not automatically detect target column.")
        print(f"Please check your CSV file and identify which column contains the student performance scores.")
        print(f"\nAvailable columns: {list(df.columns)}")
        return None

if __name__ == "__main__":
    csv_file = input("Enter path to your CSV file: ").strip()
    if not csv_file:
        csv_file = "C:/Users/Admin/Downloads/Student_Performance.csv"
    
    fix_dataset(csv_file)