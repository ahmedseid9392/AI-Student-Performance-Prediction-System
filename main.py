"""
Main entry point for the Student Performance Prediction System
"""

import tkinter as tk
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui import StudentPerformanceGUI

def main():
    """Main function to run the application"""
    try:
        # Create root window
        root = tk.Tk()
        
        # Initialize GUI
        app = StudentPerformanceGUI(root)
        
        # Set up closing handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Run application
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()