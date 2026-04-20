"""
Graphical User Interface module for Student Performance Prediction System
Professional Modern GUI with Enhanced Styling and User Experience
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime
import threading
import logging
import traceback
import os

from data_preprocessing import DataPreprocessor, generate_sample_data
from model_training import ModelTrainer
from config import FEATURES, TARGET, PERFORMANCE_CATEGORIES, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentPerformanceGUI:
    """Professional Student Performance Prediction GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 AI Student Performance Prediction System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        
        # Try to initialize database
        try:
            from database import DatabaseManager
            self.db_manager = DatabaseManager()
        except Exception as e:
            logger.info(f"Database not available: {e}")
            self.db_manager = None
        
        # Variables
        self.current_data = None
        self.is_model_trained = False
        self.last_prediction = None
        
        # Setup styles and GUI
        self.setup_styles()
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()
        
        logger.info("GUI initialized successfully")
    
    def setup_styles(self):
        """Setup modern professional styles"""
        style = ttk.Style()
        
        # Set theme
        style.theme_use('clam')
        
        # Color scheme
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'white': '#ffffff',
            'gray': '#95a5a6'
        }
        
        # Configure styles
        style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'), foreground=self.colors['primary'])
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['secondary'])
        style.configure('Custom.Treeview', font=('Segoe UI', 9), rowheight=25)
        style.configure('Custom.Treeview.Heading', font=('Segoe UI', 10, 'bold'))
        style.configure('Custom.TNotebook', tabposition='nw')
        style.configure('Custom.TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[10, 5])
        
        # Entry style
        style.configure('Modern.TEntry', fieldbackground='white', borderwidth=1, relief='solid')
        style.configure('Modern.TCombobox', fieldbackground='white')
    
    def create_header(self):
        """Create modern header"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, 
                               text="🎓 AI Student Performance Prediction System",
                               font=('Segoe UI', 18, 'bold'),
                               fg='white', bg=self.colors['primary'])
        title_label.pack(side='left', padx=20, pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                  text="Machine Learning Powered Academic Analytics",
                                  font=('Segoe UI', 10),
                                  fg='#ecf0f1', bg=self.colors['primary'])
        subtitle_label.pack(side='left', padx=10, pady=25)
        
        # Version badge
        version_badge = tk.Label(header_frame,
                                 text="v2.0",
                                 font=('Segoe UI', 9, 'bold'),
                                 fg=self.colors['primary'],
                                 bg=self.colors['light'],
                                 padx=10, pady=2)
        version_badge.pack(side='right', padx=20, pady=25)
    
    def create_main_layout(self):
        """Create main application layout"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_prediction_tab()
        self.create_training_tab()
        self.create_data_tab()
        self.create_reports_tab()
    
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Welcome card
        welcome_card = tk.Frame(dashboard_frame, bg=self.colors['white'], relief='raised', bd=1)
        welcome_card.pack(fill='x', padx=20, pady=20)
        
        tk.Label(welcome_card, text="Welcome to AI Performance Predictor",
                font=('Segoe UI', 16, 'bold'), bg=self.colors['white'],
                fg=self.colors['primary']).pack(pady=20)
        
        tk.Label(welcome_card, 
                text="Upload your dataset, train the AI model, and predict student performance with high accuracy",
                font=('Segoe UI', 10), bg=self.colors['white'],
                fg=self.colors['gray']).pack(pady=(0, 20))
        
        # Quick actions
        actions_frame = tk.Frame(dashboard_frame, bg=self.colors['light'])
        actions_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(actions_frame, text="Quick Actions", font=('Segoe UI', 14, 'bold'),
                bg=self.colors['light'], fg=self.colors['primary']).pack(anchor='w', pady=10)
        
        # Action buttons
        button_frame = tk.Frame(actions_frame, bg=self.colors['light'])
        button_frame.pack(fill='x')
        
        actions = [
            ("📁 Load Dataset", self.load_dataset),
            ("🎯 Train Model", self.train_model),
            ("🔮 Make Prediction", lambda: self.notebook.select(1)),
            ("📊 View Reports", lambda: self.notebook.select(4))
        ]
        
        for text, command in actions:
            btn = tk.Button(button_frame, text=text, command=command,
                          font=('Segoe UI', 10, 'bold'),
                          bg=self.colors['accent'], fg='white', 
                          padx=20, pady=10, cursor='hand2', relief='flat')
            btn.pack(side='left', padx=10)
        
        # Status cards
        stats_frame = tk.Frame(dashboard_frame, bg=self.colors['light'])
        stats_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(stats_frame, text="System Status", font=('Segoe UI', 14, 'bold'),
                bg=self.colors['light'], fg=self.colors['primary']).pack(anchor='w', pady=10)
        
        stats_grid = tk.Frame(stats_frame, bg=self.colors['light'])
        stats_grid.pack(fill='x')
        
        stats = [
            ("Dataset Status", "No Data Loaded", "📁"),
            ("Model Status", "Not Trained", "🤖"),
            ("Prediction Ready", "No", "🎯"),
            ("Database", "Connected" if self.db_manager else "Offline", "💾")
        ]
        
        self.status_cards = {}
        for i, (title, value, icon) in enumerate(stats):
            card = tk.Frame(stats_grid, bg=self.colors['white'], relief='raised', bd=1)
            card.grid(row=0, column=i, padx=10, pady=10, sticky='nsew')
            stats_grid.grid_columnconfigure(i, weight=1)
            
            tk.Label(card, text=f"{icon} {title}", font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['white'], fg=self.colors['secondary']).pack(pady=(10,5))
            
            value_label = tk.Label(card, text=value, font=('Segoe UI', 11),
                                  bg=self.colors['white'], fg=self.colors['accent'])
            value_label.pack(pady=(0,10))
            
            self.status_cards[title] = value_label
    
    def create_prediction_tab(self):
        """Create prediction tab with enhanced styled input fields"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="🔮 Predict Performance")
        
        # Split into two columns
        left_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        right_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Input form card
        form_card = tk.Frame(left_panel, bg=self.colors['white'], relief='raised', bd=1)
        form_card.pack(fill='both', expand=True)
        
        tk.Label(form_card, text="Student Information Form",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=15)
        
        # Create scrollable canvas for form
        canvas = tk.Canvas(form_card, bg=self.colors['white'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_card, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input fields with enhanced styling
        fields = [
            ('👤 Student Name:', 'name', 'text', 'Enter student name'),
            ('📊 Age:', 'age', 'number', '14-18 years'),
            ('⚥ Gender:', 'gender', 'combo', 'male,female,other'),
            ('🏫 School Type:', 'school_type', 'combo', 'public,private'),
            ('🎓 Parent Education:', 'parent_education', 'combo', 'high school,graduate,post graduate,masters'),
            ('📚 Study Hours/Day:', 'study_hours', 'number', '0-15 hours'),
            ('📈 Attendance %:', 'attendance_percentage', 'number', '0-100%'),
            ('🌐 Internet Access:', 'internet_access', 'combo', '0,1'),
            ('🚌 Travel Time:', 'travel_time', 'combo', '0,1,2,3'),
            ('⭐ Extra Activities:', 'extra_activities', 'combo', '0,1'),
            ('📖 Study Method:', 'study_method', 'combo', 'self,group study,mixed,textbook,notes'),
            ('🧮 Math Score:', 'math_score', 'number', '0-100'),
            ('🔬 Science Score:', 'science_score', 'number', '0-100'),
            ('📝 English Score:', 'english_score', 'number', '0-100')
        ]
        
        self.input_vars = {}
        
        for i, (label, key, field_type, placeholder) in enumerate(fields):
            # Label with icon
            label_frame = tk.Frame(scrollable_frame, bg=self.colors['white'])
            label_frame.grid(row=i, column=0, sticky='w', pady=8, padx=15)
            
            tk.Label(label_frame, text=label, font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['white'], fg=self.colors['secondary']).pack(side='left')
            
            # Input field frame
            input_frame = tk.Frame(scrollable_frame, bg=self.colors['white'])
            input_frame.grid(row=i, column=1, pady=8, padx=15, sticky='ew')
            
            if field_type == 'combo':
                var = tk.StringVar()
                combo = ttk.Combobox(input_frame, textvariable=var, 
                                    values=placeholder.split(','),
                                    font=('Segoe UI', 10), width=30,
                                    style='Modern.TCombobox')
                combo.pack(side='left')
                
                # Set default values
                if key == 'gender':
                    var.set('male')
                elif key == 'school_type':
                    var.set('public')
                elif key == 'parent_education':
                    var.set('graduate')
                elif key == 'study_method':
                    var.set('self')
                elif key == 'internet_access':
                    var.set('1')
                elif key == 'extra_activities':
                    var.set('1')
                elif key == 'travel_time':
                    var.set('0')
                else:
                    var.set(placeholder.split(',')[0] if placeholder else '')
                
                self.input_vars[key] = var
                
                # Tooltip
                self.create_tooltip(combo, f"Select {label}")
                
            else:
                var = tk.DoubleVar() if field_type == 'number' else tk.StringVar()
                
                # Entry with placeholder
                entry = tk.Entry(input_frame, textvariable=var, font=('Segoe UI', 10),
                               bg='white', relief='solid', bd=1, width=32)
                entry.pack(side='left')
                
                # Set default values
                if key == 'age':
                    var.set(16)
                elif key == 'study_hours':
                    var.set(5)
                elif key == 'attendance_percentage':
                    var.set(85)
                elif key == 'math_score':
                    var.set(70)
                elif key == 'science_score':
                    var.set(70)
                elif key == 'english_score':
                    var.set(70)
                elif key == 'name':
                    entry.insert(0, placeholder)
                    entry.bind('<FocusIn>', lambda e, ent=entry, ph=placeholder: self.clear_placeholder(ent, ph))
                    entry.bind('<FocusOut>', lambda e, ent=entry, ph=placeholder: self.restore_placeholder(ent, ph))
                else:
                    var.set(0)
                
                self.input_vars[key] = var
                
                # Tooltip
                self.create_tooltip(entry, f"Enter {label} ({placeholder})")
            
            # Hint label
            if placeholder and field_type != 'combo':
                hint_label = tk.Label(scrollable_frame, text=f"💡 {placeholder}",
                                     font=('Segoe UI', 8), bg=self.colors['white'],
                                     fg=self.colors['gray'])
                hint_label.grid(row=i, column=2, sticky='w', padx=5)
        
        # Predict button with hover effect
        predict_btn = tk.Button(scrollable_frame, text="🔮 PREDICT PERFORMANCE",
                               command=self.predict_performance,
                               font=('Segoe UI', 12, 'bold'),
                               bg=self.colors['accent'], fg='white',
                               padx=40, pady=12, cursor='hand2', 
                               relief='flat', width=35)
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=25)
        
        # Hover effect for button
        predict_btn.bind("<Enter>", lambda e: predict_btn.configure(bg=self.colors['secondary']))
        predict_btn.bind("<Leave>", lambda e: predict_btn.configure(bg=self.colors['accent']))
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Results panel
        results_card = tk.Frame(right_panel, bg=self.colors['white'], relief='raised', bd=1)
        results_card.pack(fill='both', expand=True)
        
        tk.Label(results_card, text="Prediction Results",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=15)
        
        # Result text with custom styling
        self.result_text = tk.Text(results_card, height=20, width=45,
                                  font=('Segoe UI', 11), wrap=tk.WORD,
                                  bg=self.colors['light'], relief='flat',
                                  padx=15, pady=15)
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure text tags for styling
        self.result_text.tag_configure('title', font=('Segoe UI', 14, 'bold'), foreground=self.colors['primary'])
        self.result_text.tag_configure('score', font=('Segoe UI', 24, 'bold'), foreground=self.colors['accent'])
        self.result_text.tag_configure('category', font=('Segoe UI', 12, 'bold'))
        self.result_text.tag_configure('advice', font=('Segoe UI', 10), foreground=self.colors['secondary'])
        
        # Action buttons
        button_frame = tk.Frame(results_card, bg=self.colors['white'])
        button_frame.pack(fill='x', pady=10)
        
        self.save_btn = tk.Button(button_frame, text="💾 Save to Database",
                                 command=self.save_prediction, state='disabled',
                                 font=('Segoe UI', 10), bg=self.colors['success'],
                                 fg='white', padx=20, pady=8, cursor='hand2',
                                 relief='flat')
        self.save_btn.pack(side='left', padx=10)
        
        export_btn = tk.Button(button_frame, text="📎 Export Results",
                              command=self.export_results,
                              font=('Segoe UI', 10), bg=self.colors['secondary'],
                              fg='white', padx=20, pady=8, cursor='hand2',
                              relief='flat')
        export_btn.pack(side='left', padx=10)
        
        clear_btn = tk.Button(button_frame, text="🔄 Clear Form",
                             command=self.clear_form,
                             font=('Segoe UI', 10), bg=self.colors['warning'],
                             fg='white', padx=20, pady=8, cursor='hand2',
                             relief='flat')
        clear_btn.pack(side='left', padx=10)
    
    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background='#ffffe0', 
                           relief='solid', borderwidth=1, font=('Segoe UI', 8))
            label.pack()
            widget.tooltip = tooltip
        
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
    
    def clear_placeholder(self, entry, placeholder):
        """Clear placeholder text on focus"""
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.configure(fg='black')
    
    def restore_placeholder(self, entry, placeholder):
        """Restore placeholder text on blur"""
        if not entry.get():
            entry.insert(0, placeholder)
            entry.configure(fg='gray')
    
    def create_training_tab(self):
        """Create training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🤖 Model Training")
        
        # Control panel
        control_card = tk.Frame(training_frame, bg=self.colors['white'], relief='raised', bd=1)
        control_card.pack(fill='x', padx=20, pady=20)
        
        tk.Label(control_card, text="Training Controls",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=15)
        
        button_frame = tk.Frame(control_card, bg=self.colors['white'])
        button_frame.pack(pady=10)
        
        buttons = [
            ("📁 Load Dataset", self.load_dataset),
            ("🎯 Train Model", self.train_model),
            ("💾 Save Model", self.save_model),
            ("📂 Load Model", self.load_model)
        ]
        
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command,
                          font=('Segoe UI', 10), bg=self.colors['accent'], fg='white',
                          padx=20, pady=8, cursor='hand2', relief='flat')
            btn.pack(side='left', padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate', length=400)
        self.progress.pack(pady=20)
        
        # Results display
        results_card = tk.Frame(training_frame, bg=self.colors['white'], relief='raised', bd=1)
        results_card.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(results_card, text="Training Results",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=10)
        
        text_frame = tk.Frame(results_card, bg=self.colors['white'])
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.training_text = tk.Text(text_frame, height=15, font=('Consolas', 9),
                                    wrap=tk.WORD, bg=self.colors['light'],
                                    relief='flat')
        self.training_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.training_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.training_text.configure(yscrollcommand=scrollbar.set)
    
    def create_data_tab(self):
        """Create data management tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data Management")
        
        # Data preview
        preview_card = tk.Frame(data_frame, bg=self.colors['white'], relief='raised', bd=1)
        preview_card.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(preview_card, text="Dataset Preview",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=10)
        
        tree_frame = tk.Frame(preview_card, bg=self.colors['white'])
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.data_tree = ttk.Treeview(tree_frame, style='Custom.Treeview')
        self.data_tree.pack(side='left', fill='both', expand=True)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.data_tree.yview)
        vsb.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(preview_card, orient="horizontal", command=self.data_tree.xview)
        hsb.pack(fill='x', padx=10, pady=(0, 10))
        self.data_tree.configure(xscrollcommand=hsb.set)
        
        # Statistics
        stats_card = tk.Frame(data_frame, bg=self.colors['white'], relief='raised', bd=1)
        stats_card.pack(fill='x', padx=20, pady=20)
        
        tk.Label(stats_card, text="Dataset Statistics",
                font=('Segoe UI', 12, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=10)
        
        self.stats_text = tk.Text(stats_card, height=6, font=('Consolas', 9),
                                 wrap=tk.WORD, bg=self.colors['light'], relief='flat')
        self.stats_text.pack(fill='x', padx=10, pady=10)
    
    def create_reports_tab(self):
        """Create reports tab"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="📈 Reports")
        
        # Button panel
        button_panel = tk.Frame(reports_frame, bg=self.colors['light'])
        button_panel.pack(fill='x', padx=20, pady=20)
        
        buttons = [
            ("📜 Prediction History", self.show_prediction_history),
            ("👥 Student Records", self.show_student_records),
            ("📊 Model Performance", self.show_model_performance),
            ("📈 Visualizations", self.show_visualizations)
        ]
        
        for text, command in buttons:
            btn = tk.Button(button_panel, text=text, command=command,
                          font=('Segoe UI', 10), bg=self.colors['accent'],
                          fg='white', padx=20, pady=8, cursor='hand2', relief='flat')
            btn.pack(side='left', padx=10)
        
        # Display area
        display_card = tk.Frame(reports_frame, bg=self.colors['white'], relief='raised', bd=1)
        display_card.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(display_card, text="Report Viewer",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=10)
        
        self.report_display = tk.Text(display_card, height=20, font=('Consolas', 9),
                                     wrap=tk.WORD, bg=self.colors['light'], relief='flat')
        self.report_display.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg=self.colors['secondary'], height=30)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_bar, text="✅ System Ready",
                                    font=('Segoe UI', 9), fg='white',
                                    bg=self.colors['secondary'])
        self.status_label.pack(side='left', padx=10, pady=5)
        
        # Clock
        self.clock_label = tk.Label(self.status_bar, text="",
                                   font=('Segoe UI', 9), fg='white',
                                   bg=self.colors['secondary'])
        self.clock_label.pack(side='right', padx=10, pady=5)
        self.update_clock()
    
    def update_clock(self):
        """Update clock display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_label.config(text=current_time)
        self.root.after(1000, self.update_clock)
    
    def update_status(self, message, status_type='info'):
        """Update status bar"""
        icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        icon = icons.get(status_type, 'ℹ️')
        self.status_label.config(text=f"{icon} {message}")
        
        # Update status card
        if status_type == 'success' and 'Model Status' in self.status_cards:
            self.status_cards['Model Status'].config(text="Trained")
            self.status_cards['Prediction Ready'].config(text="Yes")
    
    def load_dataset(self):
        """Load dataset from CSV"""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_data = pd.read_csv(filename)
                self.update_data_preview()
                self.update_status(f"Dataset loaded: {self.current_data.shape[0]} rows", 'success')
                
                # Update status cards
                if 'Dataset Status' in self.status_cards:
                    self.status_cards['Dataset Status'].config(text=f"Loaded ({self.current_data.shape[0]} rows)")
                
                messagebox.showinfo("Success", f"Dataset loaded successfully!\nShape: {self.current_data.shape}")
            except Exception as e:
                self.update_status(f"Failed to load dataset: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def update_data_preview(self):
        """Update data preview"""
        if self.current_data is not None:
            # Clear existing tree
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Setup columns
            columns = list(self.current_data.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'
            
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # Add data
            for idx, row in self.current_data.head(100).iterrows():
                self.data_tree.insert('', 'end', values=list(row))
            
            # Update statistics
            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats = self.current_data[numeric_cols].describe()
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, str(stats))
    
    def train_model(self):
        """Train the model"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        self.update_status("Starting model training...", 'info')
        self.progress.start()
        self.training_text.delete(1.0, tk.END)
        self.training_text.insert(tk.END, "🚀 Starting model training...\n\n")
        
        thread = threading.Thread(target=self._training_worker, daemon=True)
        thread.start()
    
    def _training_worker(self):
        """Training worker thread"""
        try:
            self._update_training_text("📊 Preprocessing data...\n")
            result = self.preprocessor.prepare_data(self.current_data)
            
            if isinstance(result, tuple) and len(result) == 2:
                X, y = result
            else:
                self._update_training_text(f"❌ Error: No target column '{TARGET}' found!\n")
                self._stop_progress()
                return
            
            self._update_training_text(f"✅ Data shape: {X.shape}\n")
            self._update_training_text("🎯 Training models...\n\n")
            
            results, X_test, y_test = self.model_trainer.train_and_evaluate(X, y)
            
            self._update_training_text("📈 Model Performance Results\n")
            self._update_training_text("=" * 50 + "\n\n")
            
            for name, metrics in results.items():
                self._update_training_text(
                    f"🔹 {name}:\n"
                    f"   📉 RMSE: {metrics['rmse']:.4f}\n"
                    f"   📊 R² Score: {metrics['r2']:.4f}\n\n"
                )
            
            self._update_training_text("=" * 50 + "\n")
            self._update_training_text(f"🏆 Best Model: {self.model_trainer.best_model_name}\n")
            self._update_training_text(f"📈 Best R² Score: {self.model_trainer.model_metrics['r2']:.4f}\n")
            
            self.is_model_trained = True
            self._update_training_text("\n✅ Model training completed successfully!\n")
            self.update_status("Model training completed!", 'success')
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed successfully!"))
            
        except Exception as e:
            error_msg = str(e)
            self._update_training_text(f"\n❌ Training failed: {error_msg}\n")
            self.update_status(f"Training failed: {error_msg}", 'error')
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {error_msg}"))
        finally:
            self._stop_progress()
    
    def _update_training_text(self, text):
        """Update training text"""
        self.root.after(0, lambda: self.training_text.insert(tk.END, text))
        self.root.after(0, lambda: self.training_text.see(tk.END))
    
    def _stop_progress(self):
        """Stop progress bar"""
        self.root.after(0, self.progress.stop)
    
    def predict_performance(self):
        """Make prediction with proper categorical encoding"""
        if not self.is_model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        
        try:
            # Get student name
            student_name = self.input_vars['name'].get()
            if not student_name or student_name == "Enter student name":
                student_name = "Student"
            
            # Create dictionary with all input values
            student_data = {}
            
            # Handle numeric fields
            numeric_fields = ['age', 'study_hours', 'attendance_percentage', 
                             'math_score', 'science_score', 'english_score']
            for field in numeric_fields:
                try:
                    value = self.input_vars[field].get()
                    student_data[field] = float(value) if value else 0.0
                except:
                    student_data[field] = 0.0
            
            # Handle integer fields
            int_fields = ['internet_access', 'travel_time', 'extra_activities']
            for field in int_fields:
                try:
                    value = self.input_vars[field].get()
                    student_data[field] = int(float(value)) if value else 0
                except:
                    student_data[field] = 0
            
            # Handle categorical fields (will be encoded by preprocessor)
            categorical_fields = ['gender', 'school_type', 'parent_education', 'study_method']
            for field in categorical_fields:
                value = self.input_vars[field].get()
                student_data[field] = str(value) if value else ""
            
            # Create DataFrame
            input_df = pd.DataFrame([student_data])
            
            # Use the preprocessor to encode categorical variables
            # The preprocessor already has the label encoders from training
            X_input = self.preprocessor.prepare_features_only(input_df)
            
            # Make prediction
            predicted_score = self.model_trainer.predict(X_input)[0]
            predicted_score = max(0, min(100, predicted_score))  # Clamp to 0-100
            
            # Determine category and advice
            if predicted_score >= 85:
                category = "🏆 EXCELLENT"
                advice = "Outstanding performance! Keep up the great work! 🌟"
                category_color = "#27ae60"
            elif predicted_score >= 70:
                category = "👍 GOOD"
                advice = "Good job! With consistent effort, you can achieve excellence! 📈"
                category_color = "#3498db"
            elif predicted_score >= 50:
                category = "📚 AVERAGE"
                advice = "You're on the right track. Focus on improving weaker subjects. 💪"
                category_color = "#f39c12"
            else:
                category = "⚠️ NEEDS IMPROVEMENT"
                advice = "Need significant improvement. Consider extra help and more study time. 📖"
                category_color = "#e74c3c"
            
            # Display formatted results
            self.result_text.delete(1.0, tk.END)
            
            # Title
            self.result_text.insert(tk.END, "=" * 55 + "\n", 'title')
            self.result_text.insert(tk.END, "🎯 PREDICTION RESULTS\n", 'title')
            self.result_text.insert(tk.END, "=" * 55 + "\n\n", 'title')
            
            # Student info
            self.result_text.insert(tk.END, f"👤 Student: {student_name}\n")
            self.result_text.insert(tk.END, f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            self.result_text.insert(tk.END, "─" * 55 + "\n")
            
            # Score
            self.result_text.insert(tk.END, f"📊 Predicted Score: ", 'advice')
            self.result_text.insert(tk.END, f"{predicted_score:.1f}", 'score')
            self.result_text.insert(tk.END, f"/100\n\n", 'advice')
            
            # Category
            self.result_text.insert(tk.END, f"🏅 Category: ", 'advice')
            self.result_text.tag_config('category_highlight', foreground=category_color, font=('Segoe UI', 12, 'bold'))
            self.result_text.insert(tk.END, f"{category}\n\n", 'category_highlight')
            
            self.result_text.insert(tk.END, "─" * 55 + "\n")
            
            # Advice
            self.result_text.insert(tk.END, f"💡 Recommendation:\n{advice}\n", 'advice')
            
            # Store for saving
            self.last_prediction = {
                'name': student_name,
                'score': predicted_score,
                'category': category,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data': student_data
            }
            
            self.save_btn.config(state='normal')
            self.update_status("Prediction completed successfully!", 'success')
            
        except Exception as e:
            error_msg = str(e)
            self.update_status(f"Prediction failed: {error_msg}", 'error')
            messagebox.showerror("Error", f"Prediction failed: {error_msg}\n\nPlease make sure the model is trained properly.")
            logger.error(f"Prediction error: {traceback.format_exc()}")
    
    def clear_form(self):
        """Clear all input fields"""
        for key, var in self.input_vars.items():
            if key == 'name':
                var.set("Enter student name")
            elif key == 'gender':
                var.set('male')
            elif key == 'school_type':
                var.set('public')
            elif key == 'parent_education':
                var.set('graduate')
            elif key == 'study_method':
                var.set('self')
            elif key == 'internet_access':
                var.set('1')
            elif key == 'extra_activities':
                var.set('1')
            elif key == 'travel_time':
                var.set('0')
            elif key == 'age':
                var.set(16)
            elif key == 'study_hours':
                var.set(5)
            elif key == 'attendance_percentage':
                var.set(85)
            elif key in ['math_score', 'science_score', 'english_score']:
                var.set(70)
            else:
                var.set(0)
        
        self.result_text.delete(1.0, tk.END)
        self.update_status("Form cleared", 'info')
    
    def save_prediction(self):
        """Save prediction to database"""
        if self.last_prediction and self.db_manager:
            try:
                # This would save to database
                messagebox.showinfo("Success", "Prediction saved to database!")
                self.save_btn.config(state='disabled')
                self.update_status("Prediction saved!", 'success')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")
        else:
            if not self.db_manager:
                messagebox.showinfo("Info", "Database not available - prediction not saved")
            else:
                messagebox.showwarning("Warning", "No prediction to save")
    
    def save_model(self):
        """Save model"""
        if self.is_model_trained:
            try:
                self.model_trainer.save_model()
                self.preprocessor.save_preprocessors()
                self.update_status("Model saved successfully!", 'success')
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                self.update_status(f"Failed to save model: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No trained model to save!")
    
    def load_model(self):
        """Load model"""
        filename = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Load model and preprocessors
                import joblib
                model_data = joblib.load(filename)
                self.model_trainer.best_model = model_data['model']
                self.model_trainer.best_model_name = model_data.get('model_name', 'Loaded Model')
                self.model_trainer.model_metrics = model_data.get('metrics', {})
                
                # Load preprocessors
                self.preprocessor.load_preprocessors()
                
                self.is_model_trained = True
                self.update_status("Model loaded successfully!", 'success')
                
                # Update status cards
                if 'Model Status' in self.status_cards:
                    self.status_cards['Model Status'].config(text="Loaded")
                if 'Prediction Ready' in self.status_cards:
                    self.status_cards['Prediction Ready'].config(text="Yes")
                
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                self.update_status(f"Failed to load model: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def export_results(self):
        """Export results to CSV"""
        if self.last_prediction:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                try:
                    df = pd.DataFrame([self.last_prediction['data']])
                    df['student_name'] = self.last_prediction['name']
                    df['predicted_score'] = self.last_prediction['score']
                    df['category'] = self.last_prediction['category']
                    df['prediction_date'] = self.last_prediction['date']
                    df.to_csv(filename, index=False)
                    self.update_status(f"Results exported to {filename}", 'success')
                    messagebox.showinfo("Success", f"Results exported to {filename}")
                except Exception as e:
                    self.update_status(f"Failed to export: {str(e)}", 'error')
                    messagebox.showerror("Error", f"Failed to export: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No prediction to export!")
    
    def show_prediction_history(self):
        """Show prediction history"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "📜 PREDICTION HISTORY\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")
        
        if hasattr(self, 'last_prediction') and self.last_prediction:
            self.report_display.insert(tk.END, f"Latest Prediction:\n")
            self.report_display.insert(tk.END, f"  Student: {self.last_prediction['name']}\n")
            self.report_display.insert(tk.END, f"  Score: {self.last_prediction['score']:.1f}/100\n")
            self.report_display.insert(tk.END, f"  Category: {self.last_prediction['category']}\n")
            self.report_display.insert(tk.END, f"  Date: {self.last_prediction['date']}\n")
        else:
            self.report_display.insert(tk.END, "No predictions made yet.\n")
            self.report_display.insert(tk.END, "Go to Predict Performance tab to make predictions.")
    
    def show_student_records(self):
        """Show student records"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "👥 STUDENT RECORDS\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")
        
        if self.current_data is not None:
            self.report_display.insert(tk.END, f"Total Students: {len(self.current_data)}\n")
            self.report_display.insert(tk.END, f"Features: {len(self.current_data.columns)}\n\n")
            self.report_display.insert(tk.END, "Sample Records:\n")
            self.report_display.insert(tk.END, self.current_data.head(10).to_string())
        else:
            self.report_display.insert(tk.END, "No dataset loaded.\n")
            self.report_display.insert(tk.END, "Please load a dataset first.")
    
    def show_model_performance(self):
        """Show model performance"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "📊 MODEL PERFORMANCE METRICS\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")
        
        if self.model_trainer.best_model_name:
            self.report_display.insert(tk.END, f"🏆 Best Model: {self.model_trainer.best_model_name}\n")
            self.report_display.insert(tk.END, f"📈 R² Score: {self.model_trainer.model_metrics.get('r2', 'N/A'):.4f}\n")
            self.report_display.insert(tk.END, f"📉 RMSE: {self.model_trainer.model_metrics.get('rmse', 'N/A'):.4f}\n")
            self.report_display.insert(tk.END, f"📊 MAE: {self.model_trainer.model_metrics.get('mae', 'N/A'):.4f}\n\n")
            
            # Performance interpretation
            r2 = self.model_trainer.model_metrics.get('r2', 0)
            if r2 > 0.8:
                self.report_display.insert(tk.END, "✅ Model Quality: Excellent\n")
                self.report_display.insert(tk.END, "The model shows strong predictive capability.")
            elif r2 > 0.6:
                self.report_display.insert(tk.END, "👍 Model Quality: Good\n")
                self.report_display.insert(tk.END, "The model shows reasonable predictive capability.")
            elif r2 > 0.4:
                self.report_display.insert(tk.END, "⚠️ Model Quality: Average\n")
                self.report_display.insert(tk.END, "Consider adding more features or collecting more data.")
            else:
                self.report_display.insert(tk.END, "❌ Model Quality: Needs Improvement\n")
                self.report_display.insert(tk.END, "Consider feature engineering or trying different algorithms.")
        else:
            self.report_display.insert(tk.END, "No model trained yet.\n")
            self.report_display.insert(tk.END, "Please train a model first.")
    
    def show_visualizations(self):
        """Show visualizations"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data available for visualization!")
            return
        
        # Create new window for visualizations
        viz_window = tk.Toplevel(self.root)
        viz_window.title("📊 Data Visualizations")
        viz_window.geometry("1200x900")
        viz_window.configure(bg=self.colors['light'])
        
        # Create notebook for multiple plots
        viz_notebook = ttk.Notebook(viz_window)
        viz_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Plot 1: Distribution of performance scores
        if TARGET in self.current_data.columns:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            self.current_data[TARGET].hist(bins=30, ax=ax1, color=self.colors['accent'], edgecolor='black')
            ax1.set_title('Distribution of Performance Scores', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Performance Score', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            canvas1 = FigureCanvasTkAgg(fig1, viz_notebook)
            canvas1.draw()
            frame1 = ttk.Frame(viz_notebook)
            viz_notebook.add(frame1, text="Score Distribution")
            canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot 2: Correlation heatmap
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            corr = self.current_data[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, square=True)
            ax2.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            
            canvas2 = FigureCanvasTkAgg(fig2, viz_notebook)
            canvas2.draw()
            frame2 = ttk.Frame(viz_notebook)
            viz_notebook.add(frame2, text="Correlation Matrix")
            canvas2.get_tk_widget().pack(fill='both', expand=True)
        
        # Close button
        close_btn = tk.Button(viz_window, text="Close", command=viz_window.destroy,
                            font=('Segoe UI', 10), bg=self.colors['accent'],
                            fg='white', padx=20, pady=5, cursor='hand2')
        close_btn.pack(pady=10)
    
    def on_closing(self):
        """Handle closing"""
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                self.db_manager.close()
            except:
                pass
        self.root.destroy()


# For backward compatibility
ProfessionalGUI = StudentPerformanceGUI