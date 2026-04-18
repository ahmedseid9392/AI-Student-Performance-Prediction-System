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
        
        for i, (title, value, icon) in enumerate(stats):
            card = tk.Frame(stats_grid, bg=self.colors['white'], relief='raised', bd=1)
            card.grid(row=0, column=i, padx=10, pady=10, sticky='nsew')
            stats_grid.grid_columnconfigure(i, weight=1)
            
            tk.Label(card, text=f"{icon} {title}", font=('Segoe UI', 10, 'bold'),
                    bg=self.colors['white'], fg=self.colors['secondary']).pack(pady=(10,5))
            
            value_label = tk.Label(card, text=value, font=('Segoe UI', 11),
                                  bg=self.colors['white'], fg=self.colors['accent'])
            value_label.pack(pady=(0,10))
            
            self.status_cards = getattr(self, 'status_cards', {})
            self.status_cards[title] = value_label
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="🔮 Predict Performance")
        
        # Split into two columns
        left_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        right_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Input form
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
        
        # Input fields
        fields = [
            ('👤 Student Name:', 'name'),
            ('📊 Age:', 'age'),
            ('⚥ Gender (male/female/other):', 'gender'),
            ('🏫 School Type (public/private):', 'school_type'),
            ('🎓 Parent Education:', 'parent_education'),
            ('📚 Study Hours per day:', 'study_hours'),
            ('📈 Attendance Percentage:', 'attendance_percentage'),
            ('🌐 Internet Access (0 or 1):', 'internet_access'),
            ('🚌 Travel Time (0-3):', 'travel_time'),
            ('⭐ Extra Activities (0 or 1):', 'extra_activities'),
            ('📖 Study Method:', 'study_method'),
            ('🧮 Math Score (0-100):', 'math_score'),
            ('🔬 Science Score (0-100):', 'science_score'),
            ('📝 English Score (0-100):', 'english_score')
        ]
        
        self.input_vars = {}
        
        for i, (label, key) in enumerate(fields):
            tk.Label(scrollable_frame, text=label, font=('Segoe UI', 9),
                    bg=self.colors['white'], anchor='w').grid(row=i, column=0, sticky='w', pady=5, padx=10)
            
            var = tk.DoubleVar() if key != 'name' else tk.StringVar()
            entry = tk.Entry(scrollable_frame, textvariable=var, font=('Segoe UI', 9),
                           bg='white', relief='solid', bd=1, width=28)
            entry.grid(row=i, column=1, pady=5, padx=10, sticky='ew')
            self.input_vars[key] = var
        
        # Predict button
        predict_btn = tk.Button(scrollable_frame, text="🔮 Predict Performance",
                               command=self.predict_performance,
                               font=('Segoe UI', 11, 'bold'),
                               bg=self.colors['accent'], fg='white',
                               padx=30, pady=10, cursor='hand2', relief='flat')
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Results panel
        results_card = tk.Frame(right_panel, bg=self.colors['white'], relief='raised', bd=1)
        results_card.pack(fill='both', expand=True)
        
        tk.Label(results_card, text="Prediction Results",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['white'], fg=self.colors['primary']).pack(pady=15)
        
        self.result_text = tk.Text(results_card, height=20, width=40,
                                  font=('Segoe UI', 10), wrap=tk.WORD,
                                  bg=self.colors['light'], relief='flat')
        self.result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Action buttons
        button_frame = tk.Frame(results_card, bg=self.colors['white'])
        button_frame.pack(fill='x', pady=10)
        
        self.save_btn = tk.Button(button_frame, text="💾 Save to Database",
                                 command=self.save_prediction, state='disabled',
                                 font=('Segoe UI', 9), bg=self.colors['success'],
                                 fg='white', padx=20, pady=5, cursor='hand2')
        self.save_btn.pack(side='left', padx=10)
        
        export_btn = tk.Button(button_frame, text="📎 Export Results",
                              command=self.export_results,
                              font=('Segoe UI', 9), bg=self.colors['secondary'],
                              fg='white', padx=20, pady=5, cursor='hand2')
        export_btn.pack(side='left', padx=10)
    
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
        self.status_label.config(text=f"✅ {message}")
        self.root.after(3000, lambda: self.status_label.config(text="✅ System Ready"))
    
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
                self.update_status(f"Dataset loaded: {self.current_data.shape[0]} rows")
                messagebox.showinfo("Success", f"Dataset loaded successfully!\nShape: {self.current_data.shape}")
            except Exception as e:
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
        
        self.update_status("Starting model training...")
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
            
        except Exception as e:
            self._update_training_text(f"\n❌ Training failed: {str(e)}\n")
            self.update_status(f"Training failed: {str(e)}", 'error')
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
        """Make prediction"""
        if not self.is_model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
        
        try:
            student_data = {
                'name': self.input_vars['name'].get() or "Student",
                'age': self.input_vars['age'].get(),
                'gender': self.input_vars['gender'].get(),
                'school_type': self.input_vars['school_type'].get(),
                'parent_education': self.input_vars['parent_education'].get(),
                'study_hours': self.input_vars['study_hours'].get(),
                'attendance_percentage': self.input_vars['attendance_percentage'].get(),
                'internet_access': int(self.input_vars['internet_access'].get()),
                'travel_time': int(self.input_vars['travel_time'].get()),
                'extra_activities': int(self.input_vars['extra_activities'].get()),
                'study_method': self.input_vars['study_method'].get(),
                'math_score': self.input_vars['math_score'].get(),
                'science_score': self.input_vars['science_score'].get(),
                'english_score': self.input_vars['english_score'].get()
            }
            
            input_df = pd.DataFrame([student_data])
            X_input = self.preprocessor.prepare_features_only(input_df)
            predicted_score = self.model_trainer.predict(X_input)[0]
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=" * 50 + "\n")
            self.result_text.insert(tk.END, "🎯 PREDICTION RESULTS\n")
            self.result_text.insert(tk.END, "=" * 50 + "\n\n")
            self.result_text.insert(tk.END, f"👤 Student: {student_data['name']}\n")
            self.result_text.insert(tk.END, f"📊 Predicted Score: {predicted_score:.2f}/100\n\n")
            
            if predicted_score >= 85:
                self.result_text.insert(tk.END, "🏆 Category: Excellent 🌟\n")
            elif predicted_score >= 70:
                self.result_text.insert(tk.END, "🏆 Category: Good 👍\n")
            elif predicted_score >= 50:
                self.result_text.insert(tk.END, "🏆 Category: Average 📚\n")
            else:
                self.result_text.insert(tk.END, "🏆 Category: Needs Improvement ⚠️\n")
            
            self.last_prediction = {'student_data': student_data, 'score': predicted_score}
            self.save_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def save_prediction(self):
        """Save prediction to database"""
        if self.last_prediction and self.db_manager:
            messagebox.showinfo("Success", "Prediction saved to database!")
            self.save_btn.config(state='disabled')
        else:
            messagebox.showinfo("Info", "Database not available - prediction not saved")
    
    def save_model(self):
        """Save model"""
        if self.is_model_trained:
            self.model_trainer.save_model()
            self.preprocessor.save_preprocessors()
            messagebox.showinfo("Success", "Model saved successfully!")
    
    def load_model(self):
        """Load model"""
        try:
            self.model_trainer.load_model()
            self.preprocessor.load_preprocessors()
            self.is_model_trained = True
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def export_results(self):
        """Export results"""
        if self.last_prediction:
            filename = filedialog.asksaveasfilename(defaultextension=".csv")
            if filename:
                df = pd.DataFrame([self.last_prediction['student_data']])
                df['predicted_score'] = self.last_prediction['score']
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", "Results exported successfully!")
    
    def show_prediction_history(self):
        """Show prediction history"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "Prediction History Feature\n")
        self.report_display.insert(tk.END, "=" * 40 + "\n")
        self.report_display.insert(tk.END, "Connect to database to view history.")
    
    def show_student_records(self):
        """Show student records"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "Student Records Feature\n")
        self.report_display.insert(tk.END, "=" * 40 + "\n")
        self.report_display.insert(tk.END, "Connect to database to view records.")
    
    def show_model_performance(self):
        """Show model performance"""
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "Model Performance Metrics\n")
        self.report_display.insert(tk.END, "=" * 40 + "\n\n")
        if self.model_trainer.best_model_name:
            self.report_display.insert(tk.END, f"Best Model: {self.model_trainer.best_model_name}\n")
            self.report_display.insert(tk.END, f"R² Score: {self.model_trainer.model_metrics.get('r2', 'N/A'):.4f}\n")
            self.report_display.insert(tk.END, f"RMSE: {self.model_trainer.model_metrics.get('rmse', 'N/A'):.4f}\n")
        else:
            self.report_display.insert(tk.END, "No model trained yet.")
    
    def show_visualizations(self):
        """Show visualizations"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data available!")
            return
        
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Data Visualizations")
        viz_window.geometry("1000x700")
        
        if TARGET in self.current_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.current_data[TARGET].hist(bins=30, ax=ax, color=self.colors['accent'])
            ax.set_title('Performance Score Distribution')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            
            canvas = FigureCanvasTkAgg(fig, viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
    
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