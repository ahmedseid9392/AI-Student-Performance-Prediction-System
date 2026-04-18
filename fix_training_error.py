"""
Quick fix for the training error
Run this script to patch the model_training.py file
"""

import os
import re

def fix_model_training():
    """Automatically fix the model_training.py file"""
    
    file_path = 'model_training.py'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return False
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the return statement in train_and_evaluate method
    # Look for the pattern and replace
    old_pattern = r'return results, X_test, y_test, .+'
    new_return = 'return results, X_test, y_test'
    
    if 'return results, X_test, y_test' in content:
        print("✓ File already has correct return statement")
        return True
    
    # Alternative fix - replace any return with 4 values
    content = re.sub(
        r'return results, X_test, y_test, \w+',
        'return results, X_test, y_test',
        content
    )
    
    # Also fix the method call in gui.py if needed
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Fixed model_training.py")
    
    # Also check gui.py
    gui_path = 'gui.py'
    if os.path.exists(gui_path):
        with open(gui_path, 'r', encoding='utf-8') as f:
            gui_content = f.read()
        
        # Fix the method call in gui.py
        if 'results, X_test, y_test, _ =' in gui_content:
            gui_content = gui_content.replace('results, X_test, y_test, _ =', 'results, X_test, y_test =')
            with open(gui_path, 'w', encoding='utf-8') as f:
                f.write(gui_content)
            print("✓ Fixed gui.py")
    
    print("\n✅ Fix applied! You can now run the application again.")
    return True

if __name__ == "__main__":
    fix_model_training()
    print("\nRun 'python main.py' to start the application")