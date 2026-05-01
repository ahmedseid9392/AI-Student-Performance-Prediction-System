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
        self.root.geometry("1440x860")
        self.root.minsize(1240, 760)
        self.root.configure(bg='#eef3f8')
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        
        # Try to initialize database
        try:
            from database import DatabaseManager
            self.db_manager = DatabaseManager()
            logger.info("Database connected successfully")
        except Exception as e:
            logger.info(f"Database not available: {e}")
            self.db_manager = None
        
        # Variables
        self.current_data = None
        self.is_model_trained = False
        self.last_prediction = None
        self.current_page = 0
        self.total_pages = 0
        self.displayed_data = None
        
        # Setup styles and GUI
        self.setup_styles()
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()
        
        logger.info("GUI initialized successfully")
    
    def setup_styles(self):
        """Setup modern professional styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.colors = {
            'primary': '#17324d',
            'primary_soft': '#254b73',
            'secondary': '#5f6f82',
            'accent': '#1f7ae0',
            'accent_dark': '#165eb1',
            'success': '#1f9d68',
            'warning': '#d98b18',
            'danger': '#d14d41',
            'light': '#eef3f8',
            'panel': '#f7fafc',
            'dark': '#2c3e50',
            'white': '#ffffff',
            'gray': '#7a8793',
            'border': '#d9e2ec',
            'muted': '#9aa8b6'
        }

        self.fonts = {
            'title': ('Segoe UI Semibold', 20),
            'hero': ('Segoe UI Semibold', 18),
            'section': ('Segoe UI Semibold', 13),
            'card_title': ('Segoe UI Semibold', 11),
            'body': ('Segoe UI', 10),
            'body_bold': ('Segoe UI Semibold', 10),
            'mono': ('Consolas', 9)
        }

        self.style.configure('App.TFrame', background=self.colors['light'])
        self.style.configure(
            'Custom.TNotebook',
            background=self.colors['light'],
            borderwidth=0,
            tabmargins=[8, 8, 8, 0]
        )
        self.style.configure(
            'Custom.TNotebook.Tab',
            font=self.fonts['body_bold'],
            padding=[18, 10],
            background='#dde7f0',
            foreground=self.colors['secondary'],
            borderwidth=0
        )
        self.style.map(
            'Custom.TNotebook.Tab',
            background=[('selected', self.colors['white']), ('active', '#e8f0f7')],
            foreground=[('selected', self.colors['primary']), ('active', self.colors['primary'])]
        )
        self.style.configure(
            'Custom.Treeview',
            font=self.fonts['body'],
            rowheight=28,
            background=self.colors['white'],
            fieldbackground=self.colors['white'],
            borderwidth=0
        )
        self.style.configure(
            'Custom.Treeview.Heading',
            font=self.fonts['body_bold'],
            background='#eaf1f7',
            foreground=self.colors['primary'],
            relief='flat'
        )
        self.style.map(
            'Custom.Treeview',
            background=[('selected', '#dcebff')],
            foreground=[('selected', self.colors['primary'])]
        )
        self.style.configure(
            'Modern.TCombobox',
            fieldbackground=self.colors['white'],
            background=self.colors['white'],
            foreground=self.colors['primary'],
            arrowcolor=self.colors['accent']
        )
        self.style.configure(
            'Modern.Horizontal.TProgressbar',
            troughcolor='#dde7f0',
            background=self.colors['accent'],
            bordercolor='#dde7f0',
            lightcolor=self.colors['accent'],
            darkcolor=self.colors['accent']
        )

    def create_card(self, parent, **pack_kwargs):
        """Create a reusable card container."""
        card = tk.Frame(
            parent,
            bg=self.colors['white'],
            bd=1,
            relief='solid',
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['border']
        )
        if pack_kwargs:
            card.pack(**pack_kwargs)
        return card

    def create_section_title(self, parent, title, subtitle=None, bg=None):
        """Create consistent section titles."""
        bg = bg or self.colors['white']
        wrapper = tk.Frame(parent, bg=bg)
        wrapper.pack(fill='x', padx=22, pady=(20, 12))

        tk.Label(
            wrapper,
            text=title,
            font=self.fonts['section'],
            fg=self.colors['primary'],
            bg=bg
        ).pack(anchor='w')

        if subtitle:
            tk.Label(
                wrapper,
                text=subtitle,
                font=self.fonts['body'],
                fg=self.colors['gray'],
                bg=bg
            ).pack(anchor='w', pady=(4, 0))

        return wrapper

    def create_action_button(self, parent, text, command, variant='accent', width=None):
        """Create consistently styled action buttons."""
        palette = {
            'accent': self.colors['accent'],
            'secondary': self.colors['primary_soft'],
            'success': self.colors['success'],
            'warning': self.colors['warning']
        }
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=self.fonts['body_bold'],
            bg=palette.get(variant, self.colors['accent']),
            fg='white',
            activebackground=self.colors['accent_dark'] if variant == 'accent' else palette.get(variant, self.colors['accent']),
            activeforeground='white',
            padx=18,
            pady=10,
            relief='flat',
            bd=0,
            cursor='hand2',
            width=width
        )

    def create_text_surface(self, parent, font=None, height=12):
        """Create a text area with consistent surface styling."""
        return tk.Text(
            parent,
            height=height,
            font=font or self.fonts['body'],
            wrap=tk.WORD,
            bg=self.colors['panel'],
            fg=self.colors['primary'],
            relief='flat',
            bd=0,
            padx=14,
            pady=12,
            insertbackground=self.colors['primary']
        )
    
    def create_header(self):
        """Create modern header"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=92)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        left_block = tk.Frame(header_frame, bg=self.colors['primary'])
        left_block.pack(side='left', fill='y', padx=26, pady=16)

        tk.Label(
            left_block,
            text="🎓 AI Student Performance Prediction System",
            font=self.fonts['hero'],
            fg='white',
            bg=self.colors['primary']
        ).pack(anchor='w')

        tk.Label(
            left_block,
            text="Machine learning powered academic analytics for smarter student support",
            font=self.fonts['body'],
            fg='#d8e5f2',
            bg=self.colors['primary']
        ).pack(anchor='w', pady=(4, 0))

        version_badge = tk.Label(
            header_frame,
            text="",
            font=self.fonts['body_bold'],
            fg=self.colors['primary'],
            bg='#dfeaf5',
            padx=14,
            pady=5
        )
        version_badge.pack(side='right', padx=24, pady=24)
    
    def create_main_layout(self):
        """Create main application layout"""
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill='both', expand=True, padx=16, pady=16)
        
        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        self.create_dashboard_tab()
        self.create_prediction_tab()
        self.create_training_tab()
        self.create_data_tab()
        self.create_reports_tab()
    
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(dashboard_frame, text="📊 Dashboard")

        welcome_card = self.create_card(dashboard_frame, fill='x', padx=22, pady=(22, 14))
        self.create_section_title(
            welcome_card,
            "Welcome to the AI Performance Predictor",
            "Load student data, train your model, and generate prediction insights from one workspace."
        )

        hero_strip = tk.Frame(welcome_card, bg='#edf5ff', height=78)
        hero_strip.pack(fill='x', padx=22, pady=(0, 22))
        hero_strip.pack_propagate(False)

        tk.Label(
            hero_strip,
            text="Designed for fast dataset review, training workflows, and clean reporting.",
            font=self.fonts['body_bold'],
            fg=self.colors['primary'],
            bg='#edf5ff'
        ).pack(anchor='w', padx=18, pady=(16, 4))

        tk.Label(
            hero_strip,
            text="Everything below keeps the same functionality while presenting it in a clearer, more professional layout.",
            font=self.fonts['body'],
            fg=self.colors['secondary'],
            bg='#edf5ff'
        ).pack(anchor='w', padx=18)

        actions_frame = tk.Frame(dashboard_frame, bg=self.colors['light'])
        actions_frame.pack(fill='x', padx=22, pady=6)
        self.create_section_title(actions_frame, "Quick Actions", "Jump directly into the most common tasks.", bg=self.colors['light'])

        button_frame = tk.Frame(actions_frame, bg=self.colors['light'])
        button_frame.pack(fill='x')

        actions = [
            ("📁 Load Dataset", self.load_dataset, 'accent'),
            ("🎯 Train Model", self.train_model, 'secondary'),
            ("🔮 Make Prediction", lambda: self.notebook.select(1), 'success'),
            ("📈 View Reports", lambda: self.notebook.select(4), 'warning')
        ]

        for text, command, variant in actions:
            btn = self.create_action_button(button_frame, text, command, variant=variant)
            btn.pack(side='left', padx=(0, 12))

        stats_frame = tk.Frame(dashboard_frame, bg=self.colors['light'])
        stats_frame.pack(fill='x', padx=22, pady=(10, 22))
        self.create_section_title(stats_frame, "System Status", "A quick summary of the current workspace state.", bg=self.colors['light'])

        stats_grid = tk.Frame(stats_frame, bg=self.colors['light'])
        stats_grid.pack(fill='x')

        stats = [
            ("Dataset Status", "No Data Loaded", "DATASET"),
            ("Model Status", "Not Trained", "MODEL"),
            ("Prediction Ready", "No", "READY"),
            ("Database", "Connected" if self.db_manager else "Offline", "STORAGE")
        ]

        self.status_cards = {}
        for i, (title, value, eyebrow) in enumerate(stats):
            card = self.create_card(stats_grid)
            card.grid(row=0, column=i, padx=8, pady=8, sticky='nsew')
            stats_grid.grid_columnconfigure(i, weight=1)

            tk.Label(
                card,
                text=eyebrow,
                font=('Segoe UI', 8, 'bold'),
                bg=self.colors['white'],
                fg=self.colors['muted']
            ).pack(anchor='w', padx=16, pady=(14, 6))

            tk.Label(
                card,
                text=title,
                font=self.fonts['card_title'],
                bg=self.colors['white'],
                fg=self.colors['primary']
            ).pack(anchor='w', padx=16)

            value_label = tk.Label(
                card,
                text=value,
                font=('Segoe UI Semibold', 11),
                bg=self.colors['white'],
                fg=self.colors['accent']
            )
            value_label.pack(anchor='w', padx=16, pady=(8, 16))

            self.status_cards[title] = value_label
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        prediction_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(prediction_frame, text="🔮 Predict Performance")

        left_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        left_panel.pack(side='left', fill='both', expand=True, padx=(22, 10), pady=22)

        right_panel = tk.Frame(prediction_frame, bg=self.colors['light'])
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 22), pady=22)

        form_card = self.create_card(left_panel)
        form_card.pack(fill='both', expand=True)

        self.create_section_title(
            form_card,
            "Student Information Form",
            "Enter the learner profile below to generate a performance estimate."
        )

        canvas = tk.Canvas(form_card, bg=self.colors['white'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_card, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        fields = [
            ('Student Name', 'name', 'text'),
            ('Age', 'age', 'number'),
            ('Gender', 'gender', 'combo'),
            ('School Type', 'school_type', 'combo'),
            ('Parent Education', 'parent_education', 'combo'),
            ('Study Hours / Day', 'study_hours', 'number'),
            ('Attendance %', 'attendance_percentage', 'number'),
            ('Internet Access', 'internet_access', 'combo'),
            ('Travel Time', 'travel_time', 'combo'),
            ('Extra Activities', 'extra_activities', 'combo'),
            ('Study Method', 'study_method', 'combo'),
            ('Math Score', 'math_score', 'number'),
            ('Science Score', 'science_score', 'number'),
            ('English Score', 'english_score', 'number')
        ]

        self.input_vars = {}
        scrollable_frame.grid_columnconfigure(1, weight=1)

        for i, (label, key, field_type) in enumerate(fields):
            tk.Label(
                scrollable_frame,
                text=label,
                font=self.fonts['body_bold'],
                fg=self.colors['primary'],
                bg=self.colors['white'],
                anchor='w'
            ).grid(row=i, column=0, sticky='w', pady=8, padx=(18, 10))

            if field_type == 'combo':
                var = tk.StringVar()
                if key == 'gender':
                    values = ['male', 'female', 'other']
                    var.set('male')
                elif key == 'school_type':
                    values = ['public', 'private']
                    var.set('public')
                elif key == 'parent_education':
                    values = ['high school', 'phd', 'graduate', 'diploma', 'post graduate', 'masters', 'no formal']
                    var.set('graduate')
                elif key == 'study_method':
                    values = ['textbook', 'group study', 'coaching', 'mixed', 'online videos', 'notes']
                    var.set('textbook')
                elif key in ['internet_access', 'extra_activities']:
                    values = ['yes', 'no']
                    var.set('yes')
                elif key == 'travel_time':
                    values = ['<15 min', '15-30 min', '30-60 min', '>60 min']
                    var.set('<15 min')
                else:
                    values = []

                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=values,
                                    font=self.fonts['body'], width=28, style='Modern.TCombobox', state='readonly')
                combo.grid(row=i, column=1, pady=8, padx=(0, 18), sticky='ew')
                self.input_vars[key] = var
            else:
                var = tk.DoubleVar() if field_type == 'number' else tk.StringVar()
                if key == 'age':
                    var.set(16)
                elif key == 'study_hours':
                    var.set(5)
                elif key == 'attendance_percentage':
                    var.set(85)
                elif key in ['math_score', 'science_score', 'english_score']:
                    var.set(70)
                elif key == 'name':
                    var.set("Student")

                entry = tk.Entry(scrollable_frame, textvariable=var, font=('Segoe UI', 10),
                               bg='white', relief='solid', bd=1, width=30,
                               highlightthickness=1, highlightbackground=self.colors['border'],
                               highlightcolor=self.colors['accent'])
                entry.grid(row=i, column=1, pady=8, padx=(0, 18), sticky='ew')
                self.input_vars[key] = var

        predict_btn = self.create_action_button(
            scrollable_frame,
            "Generate Performance Prediction",
            self.predict_performance,
            variant='accent'
        )
        predict_btn.grid(row=len(fields), column=0, columnspan=2, pady=24, padx=18, sticky='ew')

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y")

        results_card = self.create_card(right_panel)
        results_card.pack(fill='both', expand=True)

        self.create_section_title(
            results_card,
            "Prediction Results",
            "Model output and performance guidance will appear here after a prediction is made."
        )

        self.result_text = self.create_text_surface(results_card, font=('Segoe UI', 11), height=20)
        self.result_text.pack(fill='both', expand=True, padx=22, pady=(0, 14))

        button_frame = tk.Frame(results_card, bg=self.colors['white'])
        button_frame.pack(fill='x', padx=22, pady=(0, 18))

        self.save_btn = self.create_action_button(
            button_frame,
            "Save to Database",
            self.save_prediction,
            variant='success'
        )
        self.save_btn.config(state='disabled')
        self.save_btn.pack(side='left', padx=(0, 10))

        export_btn = self.create_action_button(
            button_frame,
            "Export Results",
            self.export_results,
            variant='secondary'
        )
        export_btn.pack(side='left', padx=(0, 10))

        clear_btn = self.create_action_button(
            button_frame,
            "Clear Form",
            self.clear_form,
            variant='warning'
        )
        clear_btn.pack(side='left')
    
    def create_training_tab(self):
        """Create training tab"""
        training_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(training_frame, text="🤖 Model Training")

        control_card = self.create_card(training_frame, fill='x', padx=22, pady=(22, 14))
        self.create_section_title(
            control_card,
            "Training Controls",
            "Manage dataset loading, model training, and saved model lifecycle from this panel."
        )

        button_frame = tk.Frame(control_card, bg=self.colors['white'])
        button_frame.pack(fill='x', padx=22, pady=(0, 20))

        buttons = [
            ("📁 Load Dataset", self.load_dataset, 'accent'),
            ("🎯 Train Model", self.train_model, 'secondary'),
            ("💾 Save Model", self.save_model, 'success'),
            ("📂 Load Model", self.load_model, 'warning')
        ]

        for text, command, variant in buttons:
            btn = self.create_action_button(button_frame, text, command, variant=variant)
            btn.pack(side='left', padx=(0, 10))

        self.progress = ttk.Progressbar(training_frame, mode='indeterminate', length=460, style='Modern.Horizontal.TProgressbar')
        self.progress.pack(pady=(0, 14))

        results_card = self.create_card(training_frame, fill='both', expand=True, padx=22, pady=(0, 22))
        self.create_section_title(
            results_card,
            "Training Results",
            "Live preprocessing and training output is shown here."
        )

        text_frame = tk.Frame(results_card, bg=self.colors['white'])
        text_frame.pack(fill='both', expand=True, padx=22, pady=(0, 18))

        self.training_text = self.create_text_surface(text_frame, font=self.fonts['mono'], height=15)
        self.training_text.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.training_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.training_text.configure(yscrollcommand=scrollbar.set)
    
    def create_data_tab(self):
        """Create data management tab with pagination"""
        data_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(data_frame, text="📊 Data Management")

        preview_card = self.create_card(data_frame, fill='both', expand=True, padx=22, pady=(22, 14))
        header_frame = tk.Frame(preview_card, bg=self.colors['white'])
        header_frame.pack(fill='x', padx=22, pady=(18, 12))

        self.dataset_info_label = tk.Label(header_frame, text="No dataset loaded", 
                                           font=self.fonts['card_title'],
                                           bg=self.colors['white'], fg=self.colors['primary'])
        self.dataset_info_label.pack(side='left')

        pagination_frame = tk.Frame(header_frame, bg=self.colors['white'])
        pagination_frame.pack(side='right')

        tk.Label(pagination_frame, text="Rows per page:", bg=self.colors['white'],
                font=self.fonts['body']).pack(side='left', padx=5)

        self.rows_per_page = tk.StringVar(value="100")
        rows_combo = ttk.Combobox(pagination_frame, textvariable=self.rows_per_page,
                                  values=["50", "100", "500", "1000", "5000", "All"],
                                  width=8, state='readonly', style='Modern.TCombobox')
        rows_combo.pack(side='left', padx=5)
        rows_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_data_preview())

        self.page_var = tk.StringVar(value="Page 1")
        page_label = tk.Label(pagination_frame, textvariable=self.page_var,
                             bg=self.colors['white'], font=self.fonts['body'])
        page_label.pack(side='left', padx=10)

        self.prev_btn = self.create_action_button(
            pagination_frame,
            "◀ Previous",
            self.prev_page,
            variant='secondary'
        )
        self.prev_btn.config(state='disabled', padx=12, pady=6)
        self.prev_btn.pack(side='left', padx=2)

        self.next_btn = self.create_action_button(
            pagination_frame,
            "Next ▶",
            self.next_page,
            variant='secondary'
        )
        self.next_btn.config(state='disabled', padx=12, pady=6)
        self.next_btn.pack(side='left', padx=2)

        export_btn = self.create_action_button(header_frame, "Export View", self.export_current_view, variant='success')
        export_btn.config(padx=14, pady=6)
        export_btn.pack(side='right', padx=10)

        tree_frame = tk.Frame(preview_card, bg=self.colors['white'])
        tree_frame.pack(fill='both', expand=True, padx=22, pady=(0, 12))

        self.data_tree = ttk.Treeview(tree_frame, style='Custom.Treeview', height=20)
        self.data_tree.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.data_tree.yview)
        vsb.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(preview_card, orient="horizontal", command=self.data_tree.xview)
        hsb.pack(fill='x', padx=22, pady=(0, 18))
        self.data_tree.configure(xscrollcommand=hsb.set)

        stats_card = self.create_card(data_frame, fill='x', padx=22, pady=(0, 22))
        self.create_section_title(
            stats_card,
            "Dataset Statistics",
            "Summary metrics and quick profiling for the currently loaded dataset."
        )

        stats_frame = tk.Frame(stats_card, bg=self.colors['white'])
        stats_frame.pack(fill='both', expand=True, padx=22, pady=(0, 18))

        self.stats_text = self.create_text_surface(stats_frame, font=self.fonts['mono'], height=8)
        self.stats_text.pack(side='left', fill='both', expand=True)

        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        stats_scrollbar.pack(side='right', fill='y')
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.current_page = 0
        self.total_pages = 0
        self.displayed_data = None
    
    def refresh_data_preview(self):
        """Refresh data preview with current pagination settings"""
        if self.current_data is not None:
            rows, cols = self.current_data.shape
            self.dataset_info_label.config(text=f"📊 Dataset: {rows:,} rows × {cols} columns")
            
            rows_per_page_str = self.rows_per_page.get()
            if rows_per_page_str == "All":
                rows_per_page = rows
            else:
                rows_per_page = int(rows_per_page_str)
            
            self.total_pages = (rows + rows_per_page - 1) // rows_per_page
            
            if self.current_page >= self.total_pages:
                self.current_page = self.total_pages - 1
            if self.current_page < 0:
                self.current_page = 0
            
            start_idx = self.current_page * rows_per_page
            end_idx = min(start_idx + rows_per_page, rows)
            
            self.displayed_data = self.current_data.iloc[start_idx:end_idx]
            self.update_treeview()
            
            self.page_var.set(f"Page {self.current_page + 1} of {max(1, self.total_pages)}")
            self.prev_btn.config(state='normal' if self.current_page > 0 else 'disabled')
            self.next_btn.config(state='normal' if self.current_page < self.total_pages - 1 else 'disabled')
            
            self.update_status(f"Showing rows {start_idx+1:,} to {end_idx:,} of {rows:,}", 'info')
    
    def update_treeview(self):
        """Update treeview with current displayed data"""
        if self.displayed_data is not None and not self.displayed_data.empty:
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            columns = list(self.displayed_data.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'
            
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            for idx, row in self.displayed_data.iterrows():
                values = [str(row[col])[:50] for col in columns]
                self.data_tree.insert('', 'end', values=values)
            
            self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display"""
        if self.current_data is not None:
            self.stats_text.delete(1.0, tk.END)
            rows, cols = self.current_data.shape
            self.stats_text.insert(tk.END, f"📊 DATASET OVERVIEW\n")
            self.stats_text.insert(tk.END, "=" * 50 + "\n\n")
            self.stats_text.insert(tk.END, f"Total Rows: {rows:,}\n")
            self.stats_text.insert(tk.END, f"Total Columns: {cols}\n\n")
            
            missing = self.current_data.isnull().sum()
            if missing.sum() > 0:
                self.stats_text.insert(tk.END, f"⚠️ Missing Values:\n")
                for col, count in missing[missing > 0].items():
                    self.stats_text.insert(tk.END, f"  {col}: {count:,} ({count/rows*100:.1f}%)\n")
            
            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.stats_text.insert(tk.END, f"\n📈 Numeric Columns: {len(numeric_cols)}\n")
                for col in numeric_cols[:5]:
                    self.stats_text.insert(tk.END, f"  {col}: {self.current_data[col].mean():.2f} ± {self.current_data[col].std():.2f}\n")
    
    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_data_preview()
    
    def next_page(self):
        """Go to next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.refresh_data_preview()
    
    def export_current_view(self):
        """Export current view to CSV"""
        if self.displayed_data is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".csv")
            if filename:
                self.displayed_data.to_csv(filename, index=False)
                self.update_status(f"Exported {len(self.displayed_data)} rows", 'success')
                messagebox.showinfo("Success", f"Exported {len(self.displayed_data)} rows")
    
    def create_reports_tab(self):
        """Create reports tab"""
        reports_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(reports_frame, text="📈 Reports")

        button_panel = tk.Frame(reports_frame, bg=self.colors['light'])
        button_panel.pack(fill='x', padx=22, pady=(22, 10))
        self.create_section_title(button_panel, "Reports and Insights", "Open prediction history, student records, model metrics, and charts from this view.", bg=self.colors['light'])

        buttons = [
            ("Prediction History", self.show_prediction_history, 'accent'),
            ("Student Records", self.show_student_records, 'secondary'),
            ("Model Performance", self.show_model_performance, 'success'),
            ("Feature Importance", self.show_feature_importance_dashboard, 'accent'),
            ("Visualizations", self.show_visualizations, 'warning')
        ]

        actions_row = tk.Frame(button_panel, bg=self.colors['light'])
        actions_row.pack(fill='x')

        for text, command, variant in buttons:
            btn = self.create_action_button(actions_row, text, command, variant=variant)
            btn.pack(side='left', padx=(0, 10))

        summary_card = self.create_card(reports_frame, fill='x', padx=22, pady=(0, 14))
        self.create_section_title(
            summary_card,
            "Visual Summary",
            "Live report metrics based on the loaded dataset, trained model, and latest prediction."
        )

        self.report_metric_tiles = {}
        tiles_frame = tk.Frame(summary_card, bg=self.colors['white'])
        tiles_frame.pack(fill='x', padx=22, pady=(0, 14))

        metric_specs = [
            ("Dataset Rows", "dataset_rows"),
            ("Features", "features"),
            ("Avg Score", "avg_score"),
            ("Model Status", "model_status")
        ]

        for index, (title, key) in enumerate(metric_specs):
            tile = tk.Frame(
                tiles_frame,
                bg=self.colors['panel'],
                highlightthickness=1,
                highlightbackground=self.colors['border'],
                bd=0
            )
            tile.grid(row=0, column=index, padx=(0 if index == 0 else 10, 0), sticky='nsew')
            tiles_frame.grid_columnconfigure(index, weight=1)

            tk.Label(
                tile,
                text=title,
                font=self.fonts['body'],
                fg=self.colors['secondary'],
                bg=self.colors['panel']
            ).pack(anchor='w', padx=14, pady=(12, 4))

            value_label = tk.Label(
                tile,
                text="--",
                font=self.fonts['section'],
                fg=self.colors['primary'],
                bg=self.colors['panel']
            )
            value_label.pack(anchor='w', padx=14)

            detail_label = tk.Label(
                tile,
                text="Waiting for data",
                font=self.fonts['body'],
                fg=self.colors['gray'],
                bg=self.colors['panel']
            )
            detail_label.pack(anchor='w', padx=14, pady=(2, 12))

            self.report_metric_tiles[key] = {
                'value': value_label,
                'detail': detail_label
            }

        self.report_visual_text = self.create_text_surface(summary_card, font=self.fonts['body'], height=7)
        self.report_visual_text.pack(fill='x', padx=22, pady=(0, 20))

        display_card = self.create_card(reports_frame, fill='both', expand=True, padx=22, pady=(0, 22))
        self.create_section_title(
            display_card,
            "Report Viewer",
            "Generated report content will be displayed in this panel."
        )

        self.report_display = self.create_text_surface(display_card, font=self.fonts['mono'], height=20)
        self.report_display.pack(fill='both', expand=True, padx=22, pady=(0, 20))
        self.update_report_visual_summary()

    def update_report_visual_summary(self):
        """Update the report tab visual summary widgets."""
        if not hasattr(self, 'report_metric_tiles'):
            return

        rows = len(self.current_data) if self.current_data is not None else 0
        cols = len(self.current_data.columns) if self.current_data is not None else 0

        avg_score = None
        if self.current_data is not None and TARGET in self.current_data.columns:
            avg_score = pd.to_numeric(self.current_data[TARGET], errors='coerce').mean()

        model_name = self.model_trainer.best_model_name if self.model_trainer.best_model_name else "Not trained"
        latest_category = self.last_prediction.get('category') if self.last_prediction else "No prediction"

        self.report_metric_tiles['dataset_rows']['value'].config(text=f"{rows:,}" if rows else "--")
        self.report_metric_tiles['dataset_rows']['detail'].config(
            text="Loaded records" if rows else "No dataset loaded"
        )

        self.report_metric_tiles['features']['value'].config(text=str(cols) if cols else "--")
        self.report_metric_tiles['features']['detail'].config(
            text="Available columns" if cols else "Waiting for CSV"
        )

        avg_score_available = avg_score is not None and pd.notna(avg_score)
        self.report_metric_tiles['avg_score']['value'].config(
            text=f"{avg_score:.1f}" if avg_score_available else "--"
        )
        self.report_metric_tiles['avg_score']['detail'].config(
            text="Mean overall score" if avg_score_available else "Target not available"
        )

        self.report_metric_tiles['model_status']['value'].config(
            text="Ready" if self.model_trainer.best_model_name else "--"
        )
        self.report_metric_tiles['model_status']['detail'].config(text=model_name)

        self.report_visual_text.delete(1.0, tk.END)
        self.report_visual_text.insert(tk.END, "REPORT SNAPSHOT\n")
        self.report_visual_text.insert(tk.END, "=" * 48 + "\n")

        if rows:
            coverage_blocks = min(20, max(1, rows // 1500))
            self.report_visual_text.insert(
                tk.END,
                f"Dataset coverage : {'#' * coverage_blocks}\n"
            )
        else:
            self.report_visual_text.insert(tk.END, "Dataset coverage : No dataset loaded\n")

        if avg_score_available:
            score_blocks = int(max(0, min(20, round(avg_score / 5))))
            self.report_visual_text.insert(
                tk.END,
                f"Average score    : {'#' * score_blocks}{'.' * (20 - score_blocks)} {avg_score:.1f}/100\n"
            )
        else:
            self.report_visual_text.insert(tk.END, "Average score    : Target column not available\n")

        r2_score = self.model_trainer.model_metrics.get('r2') if self.model_trainer.model_metrics else None
        if isinstance(r2_score, (int, float)):
            r2_blocks = int(max(0, min(20, round(r2_score * 20))))
            self.report_visual_text.insert(
                tk.END,
                f"Model strength   : {'#' * r2_blocks}{'.' * (20 - r2_blocks)} R2 {r2_score:.2f}\n"
            )
        else:
            self.report_visual_text.insert(tk.END, "Model strength   : Train a model to see quality metrics\n")

        self.report_visual_text.insert(tk.END, f"Latest outcome   : {latest_category}\n")

        if self.last_prediction:
            student_name = self.last_prediction.get('name', 'Student')
            score = float(self.last_prediction.get('score', 0))
            self.report_visual_text.insert(
                tk.END,
                f"Latest student   : {student_name} ({score:.1f}/100)\n"
            )
        else:
            self.report_visual_text.insert(tk.END, "Latest student   : No prediction saved yet\n")
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Frame(self.root, bg=self.colors['primary'], height=34)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)

        self.status_label = tk.Label(self.status_bar, text="Ready",
                                    font=self.fonts['body'], fg='white',
                                    bg=self.colors['primary'])
        self.status_label.pack(side='left', padx=14, pady=7)

        self.clock_label = tk.Label(self.status_bar, text="",
                                   font=self.fonts['body'], fg='white',
                                   bg=self.colors['primary'])
        self.clock_label.pack(side='right', padx=14, pady=7)
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
        
        # Update status cards
        if status_type == 'success':
            if 'Model Status' in self.status_cards and "completed" in message:
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
                self.current_page = 0
                self.refresh_data_preview()
                self.update_report_visual_summary()
                self.update_status(f"Dataset loaded: {self.current_data.shape[0]} rows", 'success')
                
                if 'Dataset Status' in self.status_cards:
                    self.status_cards['Dataset Status'].config(text=f"Loaded ({self.current_data.shape[0]} rows)")
                
                messagebox.showinfo("Success", f"Dataset loaded successfully!\nShape: {self.current_data.shape}")
            except Exception as e:
                self.update_status(f"Failed to load dataset: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
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
                self._update_training_text(f"❌ Error: No target column found!\n")
                self._stop_progress()
                return
            
            self._update_training_text(f"✅ Data shape: {X.shape}\n")
            self._update_training_text("🎯 Training models...\n\n")
            
            results, X_test, y_test = self.model_trainer.train_and_evaluate(
                X, y, feature_names=FEATURES
            )
            
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
            
            if self.model_trainer.feature_importance:
                self._update_training_text("\nTop Feature Importance:\n")
                for item in self.model_trainer.feature_importance[:5]:
                    self._update_training_text(
                        f"   - {self.format_feature_name(item['feature'])}: {item['importance'] * 100:.1f}%\n"
                    )

            self.is_model_trained = True
            self.root.after(0, self.update_report_visual_summary)
            self._update_training_text("\n✅ Model training completed successfully!\n")
            self.update_status("Model training completed!", 'success')
            
            if 'Model Status' in self.status_cards:
                self.status_cards['Model Status'].config(text="Trained")
                self.status_cards['Prediction Ready'].config(text="Yes")
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed successfully!"))
            
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

    def format_feature_name(self, feature_name):
        """Convert raw feature names into user-friendly labels."""
        return feature_name.replace('_', ' ').title()

    def build_prediction_explanation(self, explanation):
        """Create readable explanation lines for prediction output."""
        lines = []
        baseline = explanation.get('baseline_prediction')
        if baseline is not None:
            lines.append(f"Baseline estimate from average profile: {baseline:.1f}/100")

        positive = explanation.get('top_positive', [])
        negative = explanation.get('top_negative', [])

        if positive:
            lines.append("Main factors increasing the score:")
            for item in positive:
                lines.append(
                    f"  + {self.format_feature_name(item['feature'])}: {item['contribution']:+.2f}"
                )

        if negative:
            lines.append("Main factors lowering the score:")
            for item in negative:
                lines.append(
                    f"  - {self.format_feature_name(item['feature'])}: {item['contribution']:+.2f}"
                )

        return lines
    
    def predict_performance(self):
        """Make prediction with detailed property analysis and improvement suggestions"""
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
            student_data['age'] = float(self.input_vars['age'].get()) if self.input_vars['age'].get() else 16
            student_data['study_hours'] = float(self.input_vars['study_hours'].get()) if self.input_vars['study_hours'].get() else 5
            student_data['attendance_percentage'] = float(self.input_vars['attendance_percentage'].get()) if self.input_vars['attendance_percentage'].get() else 85
            student_data['math_score'] = float(self.input_vars['math_score'].get()) if self.input_vars['math_score'].get() else 70
            student_data['science_score'] = float(self.input_vars['science_score'].get()) if self.input_vars['science_score'].get() else 70
            student_data['english_score'] = float(self.input_vars['english_score'].get()) if self.input_vars['english_score'].get() else 70
            
            # Handle categorical fields
            student_data['gender'] = self.input_vars['gender'].get() if self.input_vars['gender'].get() else "male"
            student_data['school_type'] = self.input_vars['school_type'].get() if self.input_vars['school_type'].get() else "public"
            student_data['parent_education'] = self.input_vars['parent_education'].get() if self.input_vars['parent_education'].get() else "graduate"
            student_data['study_method'] = self.input_vars['study_method'].get() if self.input_vars['study_method'].get() else "textbook"
            student_data['internet_access'] = self.input_vars['internet_access'].get() if self.input_vars['internet_access'].get() else "yes"
            student_data['extra_activities'] = self.input_vars['extra_activities'].get() if self.input_vars['extra_activities'].get() else "yes"
            student_data['travel_time'] = self.input_vars['travel_time'].get() if self.input_vars['travel_time'].get() else "<15 min"
            
            # Extract values for analysis
            math_score = student_data['math_score']
            science_score = student_data['science_score']
            english_score = student_data['english_score']
            study_hours = student_data['study_hours']
            attendance = student_data['attendance_percentage']
            travel_time_val = student_data['travel_time']
            extra_activities = student_data['extra_activities']
            internet_access = student_data['internet_access']
            has_extra_activities = extra_activities == 'yes'
            has_internet_access = internet_access == 'yes'
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([student_data])
            
            # Use the preprocessor to encode categorical variables
            X_input = self.preprocessor.prepare_features_only(input_df)
            
            # Make prediction
            predicted_score = self.model_trainer.predict(X_input)[0]
            predicted_score = max(0, min(100, predicted_score))
            explanation = self.model_trainer.explain_prediction(X_input)
            explanation_lines = self.build_prediction_explanation(explanation)
            
            travel_time_text = travel_time_val
            
            # Determine overall category
            if predicted_score >= 85:
                overall_category = "EXCELLENT"
                overall_icon = "🌟"
                overall_color = "#27ae60"
            elif predicted_score >= 70:
                overall_category = "GOOD"
                overall_icon = "👍"
                overall_color = "#3498db"
            elif predicted_score >= 50:
                overall_category = "AVERAGE"
                overall_icon = "📚"
                overall_color = "#f39c12"
            else:
                overall_category = "NEEDS IMPROVEMENT"
                overall_icon = "⚠️"
                overall_color = "#e74c3c"
            
            # Analyze strengths and weaknesses
            strengths = []
            weaknesses = []
            
            # Subject analysis
            if math_score >= 75:
                strengths.append(f"📐 Mathematics: {math_score:.0f}/100 - Strong performance!")
            elif math_score >= 60:
                strengths.append(f"📐 Mathematics: {math_score:.0f}/100 - Satisfactory")
            else:
                weaknesses.append(f"📐 Mathematics: {math_score:.0f}/100 - Needs significant improvement")
            
            if science_score >= 75:
                strengths.append(f"🔬 Science: {science_score:.0f}/100 - Strong performance!")
            elif science_score >= 60:
                strengths.append(f"🔬 Science: {science_score:.0f}/100 - Satisfactory")
            else:
                weaknesses.append(f"🔬 Science: {science_score:.0f}/100 - Needs significant improvement")
            
            if english_score >= 75:
                strengths.append(f"📝 English: {english_score:.0f}/100 - Strong performance!")
            elif english_score >= 60:
                strengths.append(f"📝 English: {english_score:.0f}/100 - Satisfactory")
            else:
                weaknesses.append(f"📝 English: {english_score:.0f}/100 - Needs significant improvement")
            
            # Study habits analysis
            if study_hours >= 6:
                strengths.append(f"📚 Study Hours: {study_hours:.1f} hours/day - Excellent dedication!")
            elif study_hours >= 4:
                strengths.append(f"📚 Study Hours: {study_hours:.1f} hours/day - Good consistency")
            else:
                weaknesses.append(f"📚 Study Hours: {study_hours:.1f} hours/day - Below recommended (aim for 5-6 hours)")
            
            # Attendance analysis
            if attendance >= 90:
                strengths.append(f"📈 Attendance: {attendance:.0f}% - Outstanding!")
            elif attendance >= 75:
                strengths.append(f"📈 Attendance: {attendance:.0f}% - Good")
            else:
                weaknesses.append(f"📈 Attendance: {attendance:.0f}% - Low attendance affects learning")
            
            # Travel time analysis
            if travel_time_val in ['<15 min', '15-30 min']:
                strengths.append(f"🚌 Travel Time: {travel_time_text} - Convenient")
            elif travel_time_val == '30-60 min':
                weaknesses.append(f"🚌 Travel Time: {travel_time_text} - Consider using travel time for review")
            else:
                weaknesses.append(f"🚌 Travel Time: {travel_time_text} - Long commute may affect study time")
            
            # Extra activities
            if has_extra_activities:
                strengths.append(f"⭐ Extra Activities: Participating - Good for holistic development")
            else:
                weaknesses.append(f"⭐ Extra Activities: Not participating - Consider joining activities")
            
            # Internet access
            if has_internet_access:
                strengths.append(f"🌐 Internet Access: Available - Good for online learning resources")
            else:
                weaknesses.append(f"🌐 Internet Access: Limited - Consider library resources")
            
            # Generate improvement suggestions
            improvement_tips = []
            
            if predicted_score < 70:
                improvement_tips.append("🎯 Set daily study goals and track progress")
            
            if math_score < 60:
                improvement_tips.append("📐 Math: Practice daily, focus on weak topics, use online tutorials")
            
            if science_score < 60:
                improvement_tips.append("🔬 Science: Create concept maps, watch educational videos, join study groups")
            
            if english_score < 60:
                improvement_tips.append("📝 English: Read regularly, practice writing, expand vocabulary")
            
            if study_hours < 5:
                improvement_tips.append("⏰ Increase study time gradually - aim for 5-6 hours daily")
            
            if attendance < 80:
                improvement_tips.append("📅 Improve attendance - regular classes are crucial for success")
            
            if travel_time_val in ['30-60 min', '>60 min']:
                improvement_tips.append("🎧 Use travel time productively - listen to educational podcasts")
            
            if not has_extra_activities:
                improvement_tips.append("🤝 Join extracurricular activities to develop soft skills")
            
            # Clear and display results
            self.result_text.delete(1.0, tk.END)
            
            # Configure text tags
            self.result_text.tag_configure('title', font=('Segoe UI', 16, 'bold'), foreground=self.colors['primary'])
            self.result_text.tag_configure('header', font=('Segoe UI', 12, 'bold'), foreground=self.colors['secondary'])
            self.result_text.tag_configure('score_big', font=('Segoe UI', 28, 'bold'), foreground=self.colors['accent'])
            self.result_text.tag_configure('category', font=('Segoe UI', 14, 'bold'), foreground=overall_color)
            self.result_text.tag_configure('strength', font=('Segoe UI', 10), foreground=self.colors['success'])
            self.result_text.tag_configure('weakness', font=('Segoe UI', 10), foreground=self.colors['danger'])
            self.result_text.tag_configure('tip', font=('Segoe UI', 10), foreground=self.colors['secondary'])
            self.result_text.tag_configure('divider', font=('Segoe UI', 10), foreground=self.colors['gray'])
            
            # Title
            self.result_text.insert(tk.END, "=" * 60 + "\n", 'divider')
            self.result_text.insert(tk.END, "🎓 STUDENT PERFORMANCE ANALYSIS\n", 'title')
            self.result_text.insert(tk.END, "=" * 60 + "\n\n", 'divider')
            
            # Student info
            self.result_text.insert(tk.END, f"👤 Student: {student_name}\n")
            self.result_text.insert(tk.END, f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Predicted Score
            self.result_text.insert(tk.END, "─" * 60 + "\n", 'divider')
            self.result_text.insert(tk.END, "🎯 PREDICTED OVERALL SCORE\n", 'header')
            self.result_text.insert(tk.END, f"{predicted_score:.1f}", 'score_big')
            self.result_text.insert(tk.END, f"/100\n\n", 'header')
            self.result_text.insert(tk.END, f"📊 Category: {overall_icon} {overall_category}\n", 'category')
            self.result_text.insert(tk.END, "─" * 60 + "\n\n", 'divider')
            
            # Model explanation
            self.result_text.insert(tk.END, "Model Explanation\n", 'header')
            self.result_text.insert(tk.END, "-" * 40 + "\n", 'divider')
            for line in explanation_lines:
                self.result_text.insert(tk.END, f"{line}\n", 'tip')
            self.result_text.insert(tk.END, "\n")

            # Subject-wise breakdown
            self.result_text.insert(tk.END, "📋 SUBJECT-WISE BREAKDOWN\n", 'header')
            self.result_text.insert(tk.END, "─" * 40 + "\n", 'divider')
            
            subjects = [
                ("Mathematics", math_score, "📐"),
                ("Science", science_score, "🔬"),
                ("English", english_score, "📝")
            ]
            
            for subj_name, subj_score, icon in subjects:
                self.result_text.insert(tk.END, f"{icon} {subj_name}: {subj_score:.0f}/100")
                bar_length = int(subj_score / 10)
                bar = "█" * bar_length + "░" * (10 - bar_length)
                self.result_text.insert(tk.END, f" [{bar}]\n")
            
            self.result_text.insert(tk.END, "\n")
            
            # Study habits
            self.result_text.insert(tk.END, "📚 STUDY HABITS\n", 'header')
            self.result_text.insert(tk.END, "─" * 40 + "\n", 'divider')
            self.result_text.insert(tk.END, f"⏰ Study Hours: {study_hours:.1f} hours/day\n")
            hour_bar_length = min(int(study_hours / 8 * 10), 10)
            hour_bar = "█" * hour_bar_length + "░" * (10 - hour_bar_length)
            self.result_text.insert(tk.END, f"   [{hour_bar}] Target: 6+ hours\n\n")
            
            self.result_text.insert(tk.END, f"📈 Attendance: {attendance:.0f}%\n")
            attendance_bar_length = int(attendance / 10)
            attendance_bar = "█" * attendance_bar_length + "░" * (10 - attendance_bar_length)
            self.result_text.insert(tk.END, f"   [{attendance_bar}] Target: 90%+\n\n")
            
            self.result_text.insert(tk.END, f"🚌 Travel Time: {travel_time_text}\n")
            self.result_text.insert(tk.END, f"⭐ Extra Activities: {'Yes ✅' if has_extra_activities else 'No ❌'}\n")
            self.result_text.insert(tk.END, f"🌐 Internet Access: {'Available ✅' if has_internet_access else 'Limited ❌'}\n\n")
            
            # Strengths
            if strengths:
                self.result_text.insert(tk.END, "✅ STRENGTHS\n", 'header')
                self.result_text.insert(tk.END, "─" * 40 + "\n", 'divider')
                for strength in strengths[:5]:
                    self.result_text.insert(tk.END, f"  {strength}\n", 'strength')
                self.result_text.insert(tk.END, "\n")
            
            # Areas for Improvement
            if weaknesses:
                self.result_text.insert(tk.END, "⚠️ AREAS FOR IMPROVEMENT\n", 'header')
                self.result_text.insert(tk.END, "─" * 40 + "\n", 'divider')
                for weakness in weaknesses[:5]:
                    self.result_text.insert(tk.END, f"  {weakness}\n", 'weakness')
                self.result_text.insert(tk.END, "\n")
            
            # Recommendations
            self.result_text.insert(tk.END, "💡 RECOMMENDATIONS FOR IMPROVEMENT\n", 'header')
            self.result_text.insert(tk.END, "─" * 40 + "\n", 'divider')
            
            if improvement_tips:
                for i, tip in enumerate(improvement_tips[:6], 1):
                    self.result_text.insert(tk.END, f"  {i}. {tip}\n", 'tip')
            else:
                self.result_text.insert(tk.END, "  🎉 Excellent work! Keep maintaining your good habits!\n", 'strength')
            
            self.result_text.insert(tk.END, "\n" + "=" * 60 + "\n", 'divider')
            self.result_text.insert(tk.END, "🎯 Focus on weak areas and maintain your strengths!\n", 'header')
            self.result_text.insert(tk.END, "=" * 60, 'divider')
            
            # Store for saving - COMPLETE DATA FOR DATABASE
            self.last_prediction = {
                'name': student_name,
                'score': predicted_score,
                'category': overall_category,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'math_score': math_score,
                'science_score': science_score,
                'english_score': english_score,
                'study_hours': study_hours,
                'attendance_percentage': attendance,
                'travel_time': travel_time_val,
                'travel_time_text': travel_time_text,
                'extra_activities': extra_activities,
                'internet_access': internet_access,
                'age': student_data['age'],
                'gender': student_data['gender'],
                'school_type': student_data['school_type'],
                'parent_education': student_data['parent_education'],
                'study_method': student_data['study_method'],
                'data': student_data,
                'explanation': explanation,
                'explanation_lines': explanation_lines
            }
            
            self.update_report_visual_summary()
            self.save_btn.config(state='normal')
            self.update_status("Prediction completed successfully!", 'success')
            
        except Exception as e:
            error_msg = str(e)
            self.update_status(f"Prediction failed: {error_msg}", 'error')
            messagebox.showerror("Error", f"Prediction failed: {error_msg}\n\nPlease make sure the model is trained properly.")
            logger.error(f"Prediction error: {traceback.format_exc()}")
    
    def save_prediction(self):
        """Save prediction to database using dataset-aligned column names."""
        if not self.last_prediction:
            messagebox.showwarning("Warning", "No prediction to save!")
            return
        
        if not self.db_manager:
            messagebox.showwarning("Warning", "Database not available. Running in offline mode.\nPrediction not saved.")
            return
        
        try:
            cursor = self.db_manager.connection.cursor()
            
            # Get actual table structure
            cursor.execute("DESCRIBE students")
            columns = cursor.fetchall()
            existing_columns = [col[0] for col in columns]
            
            # Map our data to possible column names
            column_map = {
                'name': self.last_prediction.get('name', 'Student'),
                'age': self.last_prediction.get('age', 16),
                'gender': self.last_prediction.get('gender', 'male'),
                'school_type': self.last_prediction.get('school_type', 'public'),
                'parent_education': self.last_prediction.get('parent_education', 'graduate'),
                'study_hours': float(self.last_prediction.get('study_hours', 5)),
                'attendance_percentage': float(self.last_prediction.get('attendance_percentage', 85)),
                'internet_access': self.last_prediction.get('internet_access', 'yes'),
                'travel_time': self.last_prediction.get('travel_time', '<15 min'),
                'extra_activities': self.last_prediction.get('extra_activities', 'yes'),
                'study_method': self.last_prediction.get('study_method', 'textbook'),
                'math_score': float(self.last_prediction.get('math_score', 70)),
                'science_score': float(self.last_prediction.get('science_score', 70)),
                'english_score': float(self.last_prediction.get('english_score', 70)),
                'overall_score': float(self.last_prediction.get('score', 0))
            }
            
            # Filter to only existing columns
            insert_columns = []
            insert_values = []
            for col in existing_columns:
                if col in column_map and col != 'student_id' and col != 'created_at':
                    insert_columns.append(col)
                    insert_values.append(column_map[col])
            
            if not insert_columns:
                raise Exception("No matching columns found in database table")
            
            # Build and execute dynamic insert query
            placeholders = ', '.join(['%s'] * len(insert_columns))
            columns_str = ', '.join(insert_columns)
            insert_query = f"INSERT INTO students ({columns_str}) VALUES ({placeholders})"
            
            cursor.execute(insert_query, insert_values)
            student_id = cursor.lastrowid
            
            # Insert prediction record
            insert_prediction_query = """
                INSERT INTO predictions (student_id, predicted_score, performance_category, confidence_score)
                VALUES (%s, %s, %s, %s)
            """
            
            prediction_values = (
                student_id,
                float(self.last_prediction.get('score', 0)),
                self.last_prediction.get('category', 'Average'),
                0.85
            )
            
            cursor.execute(insert_prediction_query, prediction_values)
            
            # Commit transaction
            self.db_manager.connection.commit()
            
            self.update_status(f"✅ Prediction saved to database! (Student ID: {student_id})", 'success')
            messagebox.showinfo("Success", f"Prediction saved to database!\nStudent ID: {student_id}")
            self.save_btn.config(state='disabled')
            
        except Exception as e:
            self.update_status(f"Failed to save to database: {str(e)}", 'error')
            messagebox.showerror("Error", f"Failed to save to database:\n{str(e)}")
            logger.error(f"Database save error: {traceback.format_exc()}")
    
    def clear_form(self):
        """Clear all input fields"""
        defaults = {
            'name': "Student",
            'age': 16,
            'gender': 'male',
            'school_type': 'public',
            'parent_education': 'graduate',
            'study_hours': 5,
            'attendance_percentage': 85,
            'internet_access': 'yes',
            'travel_time': '<15 min',
            'extra_activities': 'yes',
            'study_method': 'textbook',
            'math_score': 70,
            'science_score': 70,
            'english_score': 70
        }
        
        for key, value in defaults.items():
            if key in self.input_vars:
                self.input_vars[key].set(value)
        
        self.result_text.delete(1.0, tk.END)
        self.update_status("Form cleared", 'info')
    
    def save_model(self):
        """Save model"""
        if self.is_model_trained:
            try:
                self.model_trainer.save_model()
                self.preprocessor.save_preprocessors()
                if self.current_data is not None:
                    dataset_path = os.path.join('models', 'training_dataset.csv')
                    self.current_data.to_csv(dataset_path, index=False)
                self.update_status("Model saved!", 'success')
                messagebox.showinfo("Success", "Model saved successfully!")
            except Exception as e:
                self.update_status(f"Failed to save model: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No trained model to save!")
    
    def load_model(self):
        """Load model"""
        filename = filedialog.askopenfilename(
            title="Select saved model file",
            initialdir=os.path.abspath('models'),
            initialfile='best_model.pkl',
            filetypes=[("Model files", "best_model.pkl"), ("PKL files", "*.pkl")]
        )
        if filename:
            try:
                import joblib
                model_data = joblib.load(filename)
                if not isinstance(model_data, dict) or 'model' not in model_data:
                    raise ValueError(
                        "Selected file is not a saved model. Please choose 'best_model.pkl'."
                    )
                self.model_trainer.best_model = model_data['model']
                self.model_trainer.best_model_name = model_data.get('model_name', 'Loaded')
                self.model_trainer.model_metrics = model_data.get('metrics', {})
                self.model_trainer.feature_names = model_data.get('feature_names', FEATURES)
                self.model_trainer.feature_importance = model_data.get('feature_importance', [])
                self.preprocessor.load_preprocessors()
                self.is_model_trained = True

                dataset_path = os.path.join(os.path.dirname(filename), 'training_dataset.csv')
                if os.path.exists(dataset_path):
                    self.current_data = pd.read_csv(dataset_path)
                    self.current_page = 0
                    self.refresh_data_preview()
                    if 'Dataset Status' in self.status_cards:
                        self.status_cards['Dataset Status'].config(
                            text=f"Loaded ({self.current_data.shape[0]} rows)"
                        )

                self.update_report_visual_summary()
                self.update_status("Model loaded!", 'success')
                if self.current_data is not None:
                    messagebox.showinfo(
                        "Success",
                        "Model loaded successfully!\nSaved dataset preview restored in Data Management."
                    )
                else:
                    messagebox.showinfo(
                        "Success",
                        "Model loaded successfully!\nLoad a dataset separately to view Data Management rows."
                    )
            except Exception as e:
                self.update_status(f"Failed to load model: {str(e)}", 'error')
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def export_results(self):
        """Export results to CSV"""
        if self.last_prediction:
            filename = filedialog.asksaveasfilename(defaultextension=".csv")
            if filename:
                try:
                    df = pd.DataFrame([self.last_prediction])
                    df.to_csv(filename, index=False)
                    self.update_status("Results exported!", 'success')
                    messagebox.showinfo("Success", "Results exported!")
                except Exception as e:
                    self.update_status(f"Failed to export: {str(e)}", 'error')
                    messagebox.showerror("Error", f"Failed to export: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No prediction to export!")
    
    def show_prediction_history(self):
        """Show prediction history"""
        self.update_report_visual_summary()
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "📜 PREDICTION HISTORY\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")
        
        if hasattr(self, 'last_prediction') and self.last_prediction:
            self.report_display.insert(tk.END, f"Latest Prediction:\n")
            self.report_display.insert(tk.END, f"  Student: {self.last_prediction.get('name', 'N/A')}\n")
            self.report_display.insert(tk.END, f"  Score: {self.last_prediction.get('score', 0):.1f}/100\n")
            self.report_display.insert(tk.END, f"  Category: {self.last_prediction.get('category', 'N/A')}\n")
            self.report_display.insert(tk.END, f"  Date: {self.last_prediction.get('date', 'N/A')}\n")
            self.report_display.insert(tk.END, f"\n📊 Subject Scores:\n")
            self.report_display.insert(tk.END, f"  Math: {self.last_prediction.get('math_score', 0):.0f}\n")
            self.report_display.insert(tk.END, f"  Science: {self.last_prediction.get('science_score', 0):.0f}\n")
            self.report_display.insert(tk.END, f"  English: {self.last_prediction.get('english_score', 0):.0f}\n")
            explanation_lines = self.last_prediction.get('explanation_lines', [])
            if explanation_lines:
                self.report_display.insert(tk.END, "\nModel Explanation:\n")
                for line in explanation_lines:
                    self.report_display.insert(tk.END, f"  {line}\n")
        else:
            self.report_display.insert(tk.END, "No predictions made yet.\n")
            self.report_display.insert(tk.END, "Go to Predict Performance tab to make predictions.")
    
    def show_student_records(self):
        """Show student records from database"""
        self.update_report_visual_summary()
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "👥 STUDENT RECORDS\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")
        
        if self.db_manager:
            try:
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT * FROM students ORDER BY created_at DESC LIMIT 50")
                students = cursor.fetchall()
                
                if students:
                    self.report_display.insert(tk.END, f"Recent Students (last 50):\n\n")
                    columns = [desc[0] for desc in cursor.description]
                    for student in students:
                        student_record = dict(zip(columns, student))
                        student_name = student_record.get('name') or f"Student {student_record.get('student_id', 'N/A')}"
                        self.report_display.insert(
                            tk.END,
                            f"ID: {student_record.get('student_id', 'N/A')} | Name: {student_name}\n"
                        )
                        self.report_display.insert(
                            tk.END,
                            f"   Study Hours: {student_record.get('study_hours', 'N/A')} | Attendance: {student_record.get('attendance_percentage', 'N/A')}%\n"
                        )
                        self.report_display.insert(tk.END, "-" * 40 + "\n")
                else:
                    self.report_display.insert(tk.END, "No student records found in database.")
            except Exception as e:
                self.report_display.insert(tk.END, f"Error loading records: {str(e)}")
        elif self.current_data is not None:
            self.report_display.insert(tk.END, f"Total Students in Dataset: {len(self.current_data)}\n")
            self.report_display.insert(tk.END, f"Features: {len(self.current_data.columns)}\n\n")
            self.report_display.insert(tk.END, "Sample Records (first 20):\n")
            self.report_display.insert(tk.END, self.current_data.head(20).to_string())
        else:
            self.report_display.insert(tk.END, "No dataset loaded and database not available.\n")
            self.report_display.insert(tk.END, "Please load a dataset or connect to database.")
    
    def show_model_performance(self):
        """Show model performance"""
        self.update_report_visual_summary()
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
            if self.model_trainer.feature_importance:
                self.report_display.insert(tk.END, "\n\nTop Feature Importance:\n")
                for item in self.model_trainer.feature_importance[:8]:
                    self.report_display.insert(
                        tk.END,
                        f"  {self.format_feature_name(item['feature'])}: {item['importance'] * 100:.1f}%\n"
                    )
        else:
            self.report_display.insert(tk.END, "No model trained yet.\n")
            self.report_display.insert(tk.END, "Please train a model first.")
    
    def show_feature_importance_dashboard(self):
        """Show feature importance and latest prediction explanation."""
        self.update_report_visual_summary()
        self.report_display.delete(1.0, tk.END)
        self.report_display.insert(tk.END, "FEATURE IMPORTANCE DASHBOARD\n")
        self.report_display.insert(tk.END, "=" * 50 + "\n\n")

        if not self.model_trainer.best_model_name:
            self.report_display.insert(tk.END, "No trained model available.\n")
            self.report_display.insert(tk.END, "Train or load a model first.")
            return

        self.report_display.insert(tk.END, f"Best Model: {self.model_trainer.best_model_name}\n")
        r2_value = self.model_trainer.model_metrics.get('r2')
        if isinstance(r2_value, (int, float)):
            self.report_display.insert(tk.END, f"R2 Score: {r2_value:.4f}\n")
        self.report_display.insert(tk.END, "\n")

        if self.model_trainer.feature_importance:
            self.report_display.insert(tk.END, "Global Importance:\n")
            for item in self.model_trainer.feature_importance[:10]:
                bar = "#" * max(1, int(round(item['importance'] * 20)))
                self.report_display.insert(
                    tk.END,
                    f"  {self.format_feature_name(item['feature']):<24} {bar:<20} {item['importance'] * 100:5.1f}%\n"
                )
        else:
            self.report_display.insert(tk.END, "Global importance is not available for this model.\n")

        if self.last_prediction and self.last_prediction.get('explanation_lines'):
            self.report_display.insert(tk.END, "\nLatest Prediction Explanation:\n")
            for line in self.last_prediction['explanation_lines']:
                self.report_display.insert(tk.END, f"  {line}\n")
        else:
            self.report_display.insert(tk.END, "\nMake a prediction to see score drivers for one student.\n")

    def show_visualizations(self):
        """Show visualizations"""
        self.update_report_visual_summary()
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
