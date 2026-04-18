# simple_run.py - Run this without any installations (uses only Python built-ins)

import tkinter as tk
from tkinter import ttk, messagebox
import random

class SimpleStudentPerformanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction (Simple Version)")
        self.root.geometry("800x600")
        
        # Simple prediction logic (mock ML)
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Student Performance Prediction", 
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        # Input frame
        frame = ttk.LabelFrame(self.root, text="Student Information", padding=10)
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Input fields
        fields = [
            ("Study Time (hours/day):", "study_time"),
            ("Attendance (%):", "attendance"),
            ("Previous Grade (0-100):", "previous_grade"),
            ("Sleep Hours:", "sleep_hours")
        ]
        
        self.entries = {}
        for i, (label, key) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            var = tk.DoubleVar()
            entry = ttk.Entry(frame, textvariable=var)
            entry.grid(row=i, column=1, pady=5, padx=10)
            self.entries[key] = var
        
        # Predict button
        predict_btn = ttk.Button(frame, text="Predict Performance", 
                                 command=self.predict)
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
        # Result display
        self.result_text = tk.Text(self.root, height=10, font=('Arial', 11))
        self.result_text.pack(fill='both', expand=True, padx=20, pady=10)
    
    def predict(self):
        try:
            # Simple prediction formula
            study = self.entries['study_time'].get()
            attendance = self.entries['attendance'].get()
            previous = self.entries['previous_grade'].get()
            sleep = self.entries['sleep_hours'].get()
            
            # Mock ML prediction
            score = (study * 3 + attendance * 0.5 + previous * 0.4 + sleep * 1.5)
            score = min(100, max(0, score + random.uniform(-5, 5)))
            
            # Determine category
            if score >= 85:
                category = "Excellent 🌟"
                color = "green"
            elif score >= 70:
                category = "Good 👍"
                color = "blue"
            elif score >= 50:
                category = "Average 📚"
                color = "orange"
            else:
                category = "Needs Improvement ⚠"
                color = "red"
            
            # Display result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Predicted Performance Score: {score:.1f}/100\n\n")
            self.result_text.insert(tk.END, f"Performance Category: {category}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Please enter valid numbers: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleStudentPerformanceSystem(root)
    root.mainloop()