# check_split.py - Quick verification of 70/30 split
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("C:/Users/Admin/Downloads/Student_Performance.csv")
print(f"Total samples: {len(df)}")

# Simulate split
train, test = train_test_split(df, test_size=0.3, random_state=42)

print(f"Training samples: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"Testing samples: {len(test)} ({len(test)/len(df)*100:.1f}%)")
print(f"\n✅ Split ratio: {len(train)}/{len(test)} = 70/30")