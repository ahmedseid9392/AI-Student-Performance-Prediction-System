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


class HoverButton(tk.Button):
    """Button with smooth hover color transition."""

    def __init__(self, master, hover_bg=None, hover_fg=None, **kwargs):
        self._default_bg = kwargs.get('bg', '#ffffff')
        self._default_fg = kwargs.get('fg', '#000000')
        self._hover_bg = hover_bg or self._default_bg
        self._hover_fg = hover_fg or self._default_fg
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)

    def _on_enter(self, _):
        if str(self['state']) != 'disabled':
            self.configure(bg=self._hover_bg, fg=self._hover_fg)

    def _on_leave(self, _):
        self.configure(bg=self._default_bg, fg=self._default_fg)


def make_shadow_card(parent, bg='#ffffff', pad_x=0, pad_y=0, **grid_or_pack):
    """Return a white card frame with a subtle drop-shadow illusion."""
    shadow = tk.Frame(parent, bg='#d1d5db')
    card = tk.Frame(shadow, bg=bg)
    card.pack(fill='both', expand=True, padx=(0, 2), pady=(0, 2))
    return shadow, card


class StudentPerformanceGUI:
    """Professional Student Performance Prediction GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("AI Student Performance Prediction System")
        self.root.geometry("1440x860")
        self.root.configure(bg='#f1f5f9')
        self.root.minsize(1100, 700)

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
        self.current_page = 0
        self.total_pages = 0
        self.displayed_data = None

        # Setup styles and GUI
        self.setup_styles()
        self.create_header()
        self.create_main_layout()
        self.create_status_bar()

        logger.info("GUI initialized successfully")

    # ------------------------------------------------------------------ #
    #  Styles                                                              #
    # ------------------------------------------------------------------ #

    def setup_styles(self):
        """Setup modern professional styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Refined color palette
        self.colors = {
            'primary':    '#0f172a',   # slate-900
            'secondary':  '#1e293b',   # slate-800
            'surface':    '#334155',   # slate-700
            'accent':     '#0284c7',   # sky-600
            'accent_hover':'#0369a1',  # sky-700
            'success':    '#059669',   # emerald-600
            'success_hover':'#047857',
            'warning':    '#d97706',   # amber-600
            'warning_hover':'#b45309',
            'danger':     '#dc2626',   # red-600
            'danger_hover':'#b91c1c',
            'light':      '#f1f5f9',   # slate-100
            'white':      '#ffffff',
            'card':       '#ffffff',
            'border':     '#e2e8f0',   # slate-200
            'muted':      '#64748b',   # slate-500
            'text':       '#0f172a',
            'text_soft':  '#475569',   # slate-600
            'header_bg':  '#0f172a',
            'header_band':'#1e293b',
        }

        # Notebook / tabs
        style.configure('Custom.TNotebook',
                        background=self.colors['light'],
                        borderwidth=0)
        style.configure('Custom.TNotebook.Tab',
                        font=('Segoe UI', 10, 'bold'),
                        padding=[18, 8],
                        background=self.colors['border'],
                        foreground=self.colors['surface'])
        style.map('Custom.TNotebook.Tab',
                  background=[('selected', self.colors['white']),
                               ('active',   '#dbeafe')],
                  foreground=[('selected', self.colors['accent']),
                               ('active',   self.colors['accent'])])

        # Treeview
        style.configure('Custom.Treeview',
                        font=('Segoe UI', 9),
                        rowheight=26,
                        background=self.colors['white'],
                        fieldbackground=self.colors['white'],
                        foreground=self.colors['text'])
        style.configure('Custom.Treeview.Heading',
                        font=('Segoe UI', 9, 'bold'),
                        background=self.colors['secondary'],
                        foreground='white',
                        padding=[6, 4])
        style.map('Custom.Treeview',
                  background=[('selected', '#dbeafe')],
                  foreground=[('selected', self.colors['accent'])])

        # Scrollbar
        style.configure('Vertical.TScrollbar',
                        background=self.colors['border'],
                        troughcolor=self.colors['light'],
                        borderwidth=0,
                        arrowsize=12)
        style.configure('Horizontal.TScrollbar',
                        background=self.colors['border'],
                        troughcolor=self.colors['light'],
                        borderwidth=0,
                        arrowsize=12)

        # Progressbar
        style.configure('Accent.Horizontal.TProgressbar',
                        troughcolor=self.colors['border'],
                        background=self.colors['accent'],
                        borderwidth=0,
                        thickness=6)

        # Combobox
        style.configure('TCombobox',
                        fieldbackground=self.colors['white'],
                        background=self.colors['white'],
                        foreground=self.colors['text'],
                        selectbackground=self.colors['accent'],
                        borderwidth=1)

    # ------------------------------------------------------------------ #
    #  Header                                                              #
    # ------------------------------------------------------------------ #

    def create_header(self):
        """Create a two-tone professional header."""
        header = tk.Frame(self.root, bg=self.colors['header_bg'], height=72)
        header.pack(fill='x')
        header.pack_propagate(False)

        # Left accent bar
        accent_bar = tk.Frame(header, bg=self.colors['accent'], width=6)
        accent_bar.pack(side='left', fill='y')

        # Logo / icon area
        icon_frame = tk.Frame(header, bg=self.colors['header_band'], width=56)
        icon_frame.pack(side='left', fill='y')
        icon_frame.pack_propagate(False)
        tk.Label(icon_frame, text="\U0001f393", font=('Segoe UI', 20),
                 bg=self.colors['header_band'], fg='white').place(relx=0.5, rely=0.5, anchor='center')

        # Title block
        title_block = tk.Frame(header, bg=self.colors['header_bg'])
        title_block.pack(side='left', fill='y', padx=16)

        tk.Label(title_block,
                 text="AI Student Performance Prediction System",
                 font=('Segoe UI', 15, 'bold'),
                 fg='white', bg=self.colors['header_bg']).pack(anchor='w', pady=(16, 0))
        tk.Label(title_block,
                 text="Machine Learning Powered Academic Analytics",
                 font=('Segoe UI', 9),
                 fg='#94a3b8', bg=self.colors['header_bg']).pack(anchor='w')

        # Right side badge
        badge_frame = tk.Frame(header, bg=self.colors['header_bg'])
        badge_frame.pack(side='right', padx=24, pady=20)

        tk.Label(badge_frame, text=" ",
                 font=('Segoe UI', 9, 'bold'),
                 fg=self.colors['accent'],
                 bg='#1e3a5f',
                 padx=12, pady=4).pack()

    # ------------------------------------------------------------------ #
    #  Main layout                                                         #
    # ------------------------------------------------------------------ #

    def create_main_layout(self):
        """Create main application layout."""
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill='both', expand=True, padx=12, pady=12)

        self.notebook = ttk.Notebook(main_container, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)

        self.create_dashboard_tab()
        self.create_prediction_tab()
        self.create_training_tab()
        self.create_data_tab()
        self.create_reports_tab()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _section_label(self, parent, text):
        """Render a section heading with a left accent line."""
        row = tk.Frame(parent, bg=self.colors['white'])
        row.pack(fill='x', pady=(18, 6))
        tk.Frame(row, bg=self.colors['accent'], width=4, height=22).pack(side='left')
        tk.Label(row, text=text, font=('Segoe UI', 11, 'bold'),
                 bg=self.colors['white'], fg=self.colors['primary']).pack(side='left', padx=10)
        return row

    def _card(self, parent, fill='x', expand=False, padx=20, pady=(10, 0)):
        """Return a white rounded-feel card frame."""
        shadow = tk.Frame(parent, bg='#cbd5e1', padx=1, pady=1)
        shadow.pack(fill=fill, expand=expand, padx=padx, pady=pady)
        inner = tk.Frame(shadow, bg=self.colors['white'])
        inner.pack(fill='both', expand=True)
        return inner

    def _accent_button(self, parent, text, command, color_key='accent', width=None):
        hover_map = {
            'accent':  self.colors['accent_hover'],
            'success': self.colors['success_hover'],
            'warning': self.colors['warning_hover'],
            'danger':  self.colors['danger_hover'],
            'surface': '#475569',
        }
        kw = dict(text=text, command=command,
                  font=('Segoe UI', 10, 'bold'),
                  bg=self.colors[color_key], fg='white',
                  hover_bg=hover_map.get(color_key, self.colors[color_key]),
                  padx=18, pady=8, cursor='hand2',
                  relief='flat', bd=0, activeforeground='white',
                  activebackground=hover_map.get(color_key, self.colors[color_key]))
        if width:
            kw['width'] = width
        return HoverButton(parent, **kw)

    def _divider(self, parent, color=None):
        tk.Frame(parent, bg=color or self.colors['border'], height=1).pack(fill='x', padx=16, pady=4)

    # ------------------------------------------------------------------ #
    #  Dashboard tab                                                       #
    # ------------------------------------------------------------------ #

    def create_dashboard_tab(self):
        """Create a rich, professional dashboard tab."""
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text="\U0001f4ca  Dashboard")

        # Scrollable host for the whole dashboard
        dash_canvas = tk.Canvas(outer, bg=self.colors['light'], highlightthickness=0)
        dash_vsb = ttk.Scrollbar(outer, orient='vertical', command=dash_canvas.yview)
        dash_canvas.configure(yscrollcommand=dash_vsb.set)
        dash_vsb.pack(side='right', fill='y')
        dash_canvas.pack(side='left', fill='both', expand=True)

        dash_frame = tk.Frame(dash_canvas, bg=self.colors['light'])
        dash_win = dash_canvas.create_window((0, 0), window=dash_frame, anchor='nw')
        dash_frame.bind('<Configure>',
                        lambda e: dash_canvas.configure(scrollregion=dash_canvas.bbox('all')))
        dash_canvas.bind('<Configure>',
                         lambda e: dash_canvas.itemconfig(dash_win, width=e.width))

        # ---- Hero banner with gradient-like layers ---- #
        hero_shadow = tk.Frame(dash_frame, bg='#0c4a6e', padx=1, pady=1)
        hero_shadow.pack(fill='x', padx=16, pady=(16, 0))
        hero = tk.Frame(hero_shadow, bg=self.colors['accent'])
        hero.pack(fill='both', expand=True)

        # Top accent strip
        tk.Frame(hero, bg='#38bdf8', height=4).pack(fill='x')

        hero_content = tk.Frame(hero, bg=self.colors['accent'])
        hero_content.pack(fill='x', padx=24, pady=(18, 0))

        tk.Label(hero_content, text="\U0001f393",
                 font=('Segoe UI', 28),
                 bg=self.colors['accent']).pack(side='left', padx=(0, 16))

        title_block = tk.Frame(hero_content, bg=self.colors['accent'])
        title_block.pack(side='left', fill='y')
        tk.Label(title_block, text="AI Student Performance Predictor",
                 font=('Segoe UI', 18, 'bold'), fg='white',
                 bg=self.colors['accent']).pack(anchor='w')
        tk.Label(title_block,
                 text="Upload data, train ML models, and predict student outcomes with high accuracy",
                 font=('Segoe UI', 10), fg='#bae6fd',
                 bg=self.colors['accent']).pack(anchor='w', pady=(2, 0))

        # Version badge on right
        tk.Label(hero_content, text="",
                 font=('Segoe UI', 9, 'bold'), fg='#bae6fd',
                 bg='#0369a1', padx=12, pady=4).pack(side='right')

        # Sub-info bar
        sub_bar = tk.Frame(hero, bg='#0369a1', height=36)
        sub_bar.pack(fill='x', padx=0, pady=(12, 0))
        sub_bar.pack_propagate(False)

        sub_items = [
            "\U0001f4ca  Machine Learning",
            "\U0001f9e9  Data Analytics",
            "\U0001f4c8  Performance Tracking",
            "\U0001f52e  AI Predictions",
        ]
        for item in sub_items:
            tk.Label(sub_bar, text=item, font=('Segoe UI', 8),
                     fg='#7dd3fc', bg='#0369a1').pack(side='left', padx=16, pady=8)

        tk.Frame(hero, bg=self.colors['accent'], height=8).pack(fill='x')

        # ---- Quick Actions as large icon cards ---- #
        self._section_label_on(dash_frame, "Quick Actions", bg=self.colors['light'])

        actions_grid = tk.Frame(dash_frame, bg=self.colors['light'])
        actions_grid.pack(fill='x', padx=16, pady=(4, 0))

        actions = [
            ("\U0001f4c1", "Load\nDataset",     self.load_dataset,               self.colors['accent'],  '#dbeafe'),
            ("\U0001f3af", "Train\nModel",       self.train_model,                self.colors['success'], '#d1fae5'),
            ("\U0001f52e", "Make\nPrediction",   lambda: self.notebook.select(1),  self.colors['warning'], '#fef3c7'),
            ("\U0001f4c8", "View\nReports",      lambda: self.notebook.select(4), self.colors['surface'], '#e2e8f0'),
        ]

        for i, (icon, label, cmd, color, tint) in enumerate(actions):
            card_shadow = tk.Frame(actions_grid, bg='#cbd5e1', padx=1, pady=1)
            card_shadow.grid(row=0, column=i, padx=8, pady=8, sticky='nsew')
            actions_grid.grid_columnconfigure(i, weight=1)

            card = tk.Frame(card_shadow, bg=self.colors['white'])
            card.pack(fill='both', expand=True)

            # Top color bar
            tk.Frame(card, bg=color, height=5).pack(fill='x')

            # Icon circle
            icon_host = tk.Frame(card, bg=tint, width=56, height=56)
            icon_host.pack(pady=(16, 0))
            icon_host.pack_propagate(False)
            icon_host.configure(bg=tint)
            tk.Label(icon_host, text=icon, font=('Segoe UI', 22),
                     bg=tint).place(relx=0.5, rely=0.5, anchor='center')

            tk.Label(card, text=label, font=('Segoe UI', 10, 'bold'),
                     bg=self.colors['white'], fg=self.colors['primary'],
                     justify='center').pack(pady=(10, 0))

            # Action button
            btn = HoverButton(card, text="Open  \u2192", command=cmd,
                              font=('Segoe UI', 9, 'bold'),
                              bg=color, fg='white',
                              hover_bg=self._darken(color),
                              padx=16, pady=6, cursor='hand2',
                              relief='flat', bd=0,
                              activeforeground='white',
                              activebackground=self._darken(color))
            btn.pack(pady=(8, 16))

        # ---- System Status as rich metric tiles ---- #
        self._section_label_on(dash_frame, "System Status", bg=self.colors['light'])

        status_grid = tk.Frame(dash_frame, bg=self.colors['light'])
        status_grid.pack(fill='x', padx=16, pady=(4, 0))

        stats = [
            ("Dataset Status",   "No Data Loaded",                             "\U0001f4c1", self.colors['accent'],  '#dbeafe'),
            ("Model Status",     "Not Trained",                                 "\U0001f916", self.colors['success'], '#d1fae5'),
            ("Prediction Ready", "No",                                          "\U0001f3af", self.colors['warning'], '#fef3c7'),
            ("Database",         "Connected" if self.db_manager else "Offline", "\U0001f4be", self.colors['surface'], '#e2e8f0'),
        ]

        self.status_cards = {}
        for i, (title, value, icon, color, tint) in enumerate(stats):
            tile_shadow = tk.Frame(status_grid, bg='#cbd5e1', padx=1, pady=1)
            tile_shadow.grid(row=0, column=i, padx=8, pady=8, sticky='nsew')
            status_grid.grid_columnconfigure(i, weight=1)

            tile = tk.Frame(tile_shadow, bg=self.colors['white'])
            tile.pack(fill='both', expand=True)

            tk.Frame(tile, bg=color, height=4).pack(fill='x')

            # Icon + title row
            top_row = tk.Frame(tile, bg=self.colors['white'])
            top_row.pack(fill='x', padx=14, pady=(14, 0))
            tk.Label(top_row, text=icon, font=('Segoe UI', 18),
                     bg=self.colors['white']).pack(side='left')
            tk.Label(top_row, text=title, font=('Segoe UI', 9, 'bold'),
                     bg=self.colors['white'], fg=self.colors['muted']).pack(side='left', padx=8)

            # Value
            val_lbl = tk.Label(tile, text=value, font=('Segoe UI', 14, 'bold'),
                               bg=self.colors['white'], fg=color)
            val_lbl.pack(pady=(6, 4), padx=14, anchor='w')

            # Status indicator dot
            dot_row = tk.Frame(tile, bg=self.colors['white'])
            dot_row.pack(fill='x', padx=14, pady=(0, 14))

            is_ok = (title == 'Database' and self.db_manager) or \
                    (title == 'Dataset Status' and False) or \
                    (title == 'Model Status' and False) or \
                    (title == 'Prediction Ready' and False)
            dot_color = self.colors['success'] if is_ok else self.colors['border']
            tk.Frame(dot_row, bg=dot_color, width=8, height=8).pack(side='left')
            tk.Label(dot_row, text="Active" if is_ok else "Inactive",
                     font=('Segoe UI', 8), fg=self.colors['muted'],
                     bg=self.colors['white']).pack(side='left', padx=6)

            self.status_cards[title] = val_lbl

        # ---- Workflow Steps ---- #
        self._section_label_on(dash_frame, "How It Works", bg=self.colors['light'])

        steps_grid = tk.Frame(dash_frame, bg=self.colors['light'])
        steps_grid.pack(fill='x', padx=16, pady=(4, 16))

        steps = [
            ("1", "Load Data",  "Import your CSV dataset\nwith student records",     self.colors['accent']),
            ("2", "Train AI",   "Build & evaluate ML\nmodels automatically",          self.colors['success']),
            ("3", "Predict",    "Enter student info\nand get predictions",            self.colors['warning']),
            ("4", "Analyze",    "View reports, charts\nand improvement tips",         self.colors['surface']),
        ]

        for i, (num, title, desc, color) in enumerate(steps):
            step_shadow = tk.Frame(steps_grid, bg='#cbd5e1', padx=1, pady=1)
            step_shadow.grid(row=0, column=i, padx=6, pady=6, sticky='nsew')
            steps_grid.grid_columnconfigure(i, weight=1)

            step = tk.Frame(step_shadow, bg=self.colors['white'])
            step.pack(fill='both', expand=True)

            # Number circle
            num_host = tk.Frame(step, bg=color, width=36, height=36)
            num_host.pack(pady=(16, 0))
            num_host.pack_propagate(False)
            tk.Label(num_host, text=num, font=('Segoe UI', 14, 'bold'),
                     fg='white', bg=color).place(relx=0.5, rely=0.5, anchor='center')

            tk.Label(step, text=title, font=('Segoe UI', 10, 'bold'),
                     bg=self.colors['white'], fg=self.colors['primary']).pack(pady=(8, 2))
            tk.Label(step, text=desc, font=('Segoe UI', 8),
                     bg=self.colors['white'], fg=self.colors['muted'],
                     justify='center').pack(pady=(0, 14), padx=8)

            # Connector arrow (except last)
            if i < len(steps) - 1:
                tk.Label(steps_grid, text="\u279c", font=('Segoe UI', 14),
                         bg=self.colors['light'], fg=self.colors['muted']).grid(
                             row=0, column=i, sticky='e', padx=(0, 0))

    def _section_label_on(self, parent, text, bg=None):
        """Render a section heading on a non-white background."""
        bg = bg or self.colors['white']
        row = tk.Frame(parent, bg=bg)
        row.pack(fill='x', pady=(18, 6), padx=16)
        tk.Frame(row, bg=self.colors['accent'], width=4, height=22).pack(side='left')
        tk.Label(row, text=text, font=('Segoe UI', 11, 'bold'),
                 bg=bg, fg=self.colors['primary']).pack(side='left', padx=10)
        return row

    def _darken(self, hex_color, factor=0.8):
        """Darken a hex color by a factor."""
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        r, g, b = [int(c * factor) for c in (r, g, b)]
        return f'#{r:02x}{g:02x}{b:02x}'

    # ------------------------------------------------------------------ #
    #  Prediction tab                                                      #
    # ------------------------------------------------------------------ #

    def create_prediction_tab(self):
        """Create a rich prediction tab with grouped form sections and styled results."""
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text="\U0001f52e  Predict Performance")

        panes = tk.PanedWindow(outer, orient='horizontal',
                               bg=self.colors['light'],
                               sashwidth=6, sashpad=2,
                               sashrelief='flat')
        panes.pack(fill='both', expand=True, padx=12, pady=12)

        # ---- Left: input form ---------------------------------------- #
        left_wrap = tk.Frame(panes, bg=self.colors['light'])
        panes.add(left_wrap, minsize=420)

        form_shadow = tk.Frame(left_wrap, bg='#cbd5e1', padx=1, pady=1)
        form_shadow.pack(fill='both', expand=True)
        form_card = tk.Frame(form_shadow, bg=self.colors['white'])
        form_card.pack(fill='both', expand=True)

        # Header with icon
        form_hdr = tk.Frame(form_card, bg=self.colors['accent'])
        form_hdr.pack(fill='x')
        tk.Frame(form_hdr, bg='#38bdf8', height=3).pack(fill='x')

        hdr_inner = tk.Frame(form_hdr, bg=self.colors['accent'])
        hdr_inner.pack(fill='x', padx=16, pady=12)
        tk.Label(hdr_inner, text="\U0001f52e",
                 font=('Segoe UI', 20), bg=self.colors['accent']).pack(side='left', padx=(0, 10))
        title_col = tk.Frame(hdr_inner, bg=self.colors['accent'])
        title_col.pack(side='left', fill='y')
        tk.Label(title_col, text="Student Information Form",
                 font=('Segoe UI', 13, 'bold'), fg='white',
                 bg=self.colors['accent']).pack(anchor='w')
        tk.Label(title_col, text="Complete all fields, then click Predict",
                 font=('Segoe UI', 9), fg='#bae6fd',
                 bg=self.colors['accent']).pack(anchor='w')

        # Scrollable area
        scroll_host = tk.Frame(form_card, bg=self.colors['white'])
        scroll_host.pack(fill='both', expand=True)

        canvas = tk.Canvas(scroll_host, bg=self.colors['white'], highlightthickness=0)
        vsb = ttk.Scrollbar(scroll_host, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])
        win_id = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)

        scrollable_frame.bind('<Configure>', _on_frame_configure)
        canvas.bind('<Configure>', _on_canvas_configure)

        # Field definitions grouped by category
        field_groups = [
            ("\U0001f464  Personal Info", self.colors['accent'], [
                ('Student Name',  'name',              'text'),
                ('Age',           'age',               'number'),
                ('Gender',        'gender',            'combo'),
                ('School Type',   'school_type',       'combo'),
                ('Parent Education', 'parent_education', 'combo'),
            ]),
            ("\U0001f4da  Academic Profile", self.colors['success'], [
                ('Study Hours/Day',  'study_hours',          'number'),
                ('Attendance %',     'attendance_percentage','number'),
                ('Study Method',     'study_method',         'combo'),
            ]),
            ("\U0001f3e2  Environment", self.colors['warning'], [
                ('Internet Access',  'internet_access',      'combo'),
                ('Travel Time',      'travel_time',          'combo'),
                ('Extra Activities', 'extra_activities',     'combo'),
            ]),
            ("\U0001f4dd  Test Scores", self.colors['danger'], [
                ('Math Score',       'math_score',           'number'),
                ('Science Score',    'science_score',        'number'),
                ('English Score',    'english_score',        'number'),
            ]),
        ]

        self.input_vars = {}
        row_idx = 0
        row_colors = [self.colors['white'], '#f8fafc']

        for group_title, group_color, fields in field_groups:
            # Group header bar
            grp_hdr = tk.Frame(scrollable_frame, bg=group_color, height=32)
            grp_hdr.pack(fill='x')
            grp_hdr.pack_propagate(False)
            tk.Label(grp_hdr, text=group_title,
                     font=('Segoe UI', 9, 'bold'), fg='white',
                     bg=group_color).pack(side='left', padx=12, pady=5)

            for label_text, key, field_type in fields:
                bg = row_colors[row_idx % 2]
                row_frame = tk.Frame(scrollable_frame, bg=bg)
                row_frame.pack(fill='x')

                # Label with icon indicator
                tk.Frame(row_frame, bg=group_color, width=3).pack(side='left', fill='y')
                tk.Label(row_frame, text=label_text, font=('Segoe UI', 9, 'bold'),
                         bg=bg, fg=self.colors['text_soft'],
                         width=20, anchor='w').pack(side='left', padx=(10, 0), pady=8)

                if field_type == 'combo':
                    var = tk.StringVar()
                    options_map = {
                        'gender':           (['male', 'female', 'other'], 'male'),
                        'school_type':      (['public', 'private'], 'public'),
                        'parent_education': (['high school', 'phd', 'graduate', 'diploma',
                                             'post graduate', 'masters', 'no formal'], 'graduate'),
                        'study_method':     (['textbook', 'group study', 'coaching',
                                             'mixed', 'online videos', 'notes'], 'textbook'),
                        'internet_access':  (['0', '1'], '1'),
                        'extra_activities': (['0', '1'], '1'),
                        'travel_time':      (['<15 min', '15-30 min', '30-60 min', '>60 min'], '<15 min'),
                    }
                    values, default = options_map.get(key, ([], ''))
                    var.set(default)
                    cb = ttk.Combobox(row_frame, textvariable=var, values=values,
                                      font=('Segoe UI', 9), state='readonly', width=26)
                    cb.pack(side='left', padx=10, pady=8)
                    self.input_vars[key] = var
                else:
                    defaults = {'age': 16, 'study_hours': 5, 'attendance_percentage': 85,
                                'math_score': 70, 'science_score': 70, 'english_score': 70,
                                'name': 'Student'}
                    var = tk.DoubleVar() if field_type == 'number' else tk.StringVar()
                    var.set(defaults.get(key, ''))
                    entry = tk.Entry(row_frame, textvariable=var, font=('Segoe UI', 9),
                                     bg=self.colors['white'], fg=self.colors['text'],
                                     relief='solid', bd=1, width=28,
                                     highlightthickness=1,
                                     highlightcolor=group_color,
                                     highlightbackground=self.colors['border'],
                                     insertbackground=group_color)
                    entry.pack(side='left', padx=10, pady=8)
                    self.input_vars[key] = var

                row_idx += 1

        # Predict button with full-width style
        btn_host = tk.Frame(scrollable_frame, bg=self.colors['white'])
        btn_host.pack(fill='x', pady=(8, 16), padx=12)

        predict_btn = HoverButton(btn_host,
                                  text="\U0001f52e   PREDICT PERFORMANCE   \u2192",
                                  command=self.predict_performance,
                                  font=('Segoe UI', 12, 'bold'),
                                  bg=self.colors['accent'], fg='white',
                                  hover_bg=self.colors['accent_hover'],
                                  padx=30, pady=12, cursor='hand2',
                                  relief='flat', bd=0,
                                  activeforeground='white',
                                  activebackground=self.colors['accent_hover'])
        predict_btn.pack(fill='x')

        # ---- Right: results ------------------------------------------ #
        right_wrap = tk.Frame(panes, bg=self.colors['light'])
        panes.add(right_wrap, minsize=420)

        res_shadow = tk.Frame(right_wrap, bg='#cbd5e1', padx=1, pady=1)
        res_shadow.pack(fill='both', expand=True)
        res_card = tk.Frame(res_shadow, bg=self.colors['white'])
        res_card.pack(fill='both', expand=True)

        # Results header
        res_hdr = tk.Frame(res_card, bg=self.colors['success'])
        res_hdr.pack(fill='x')
        tk.Frame(res_hdr, bg='#6ee7b7', height=3).pack(fill='x')

        res_hdr_inner = tk.Frame(res_hdr, bg=self.colors['success'])
        res_hdr_inner.pack(fill='x', padx=16, pady=12)
        tk.Label(res_hdr_inner, text="\U0001f4ca",
                 font=('Segoe UI', 20), bg=self.colors['success']).pack(side='left', padx=(0, 10))
        res_title_col = tk.Frame(res_hdr_inner, bg=self.colors['success'])
        res_title_col.pack(side='left', fill='y')
        tk.Label(res_title_col, text="Prediction Results",
                 font=('Segoe UI', 13, 'bold'), fg='white',
                 bg=self.colors['success']).pack(anchor='w')
        tk.Label(res_title_col, text="AI-powered analysis and recommendations",
                 font=('Segoe UI', 9), fg='#d1fae5',
                 bg=self.colors['success']).pack(anchor='w')

        # Placeholder before prediction
        self._result_placeholder = tk.Frame(res_card, bg=self.colors['white'])
        self._result_placeholder.pack(fill='both', expand=True)

        placeholder_inner = tk.Frame(self._result_placeholder, bg='#f8fafc')
        placeholder_inner.pack(fill='both', expand=True, padx=16, pady=16)

        tk.Label(placeholder_inner, text="\U0001f52e",
                 font=('Segoe UI', 48), bg='#f8fafc',
                 fg=self.colors['border']).pack(pady=(40, 8))
        tk.Label(placeholder_inner, text="No Prediction Yet",
                 font=('Segoe UI', 14, 'bold'), bg='#f8fafc',
                 fg=self.colors['muted']).pack()
        tk.Label(placeholder_inner,
                 text="Fill in the student information form on the left\nand click Predict to see AI analysis here.",
                 font=('Segoe UI', 9), bg='#f8fafc',
                 fg=self.colors['muted'], justify='center').pack(pady=(4, 40))

        # Result text (hidden until prediction)
        self.result_text = tk.Text(res_card, font=('Segoe UI', 10), wrap=tk.WORD,
                                   bg='#f8fafc', fg=self.colors['text'],
                                   relief='flat', padx=12, pady=8,
                                   selectbackground='#dbeafe',
                                   insertbackground=self.colors['accent'])

        # Action buttons bar
        self._divider(res_card)
        action_row = tk.Frame(res_card, bg=self.colors['white'])
        action_row.pack(fill='x', padx=10, pady=10)

        self.save_btn = HoverButton(action_row,
                                    text="\U0001f4be  Save to Database",
                                    command=self.save_prediction,
                                    state='disabled',
                                    font=('Segoe UI', 9, 'bold'),
                                    bg=self.colors['success'], fg='white',
                                    hover_bg=self.colors['success_hover'],
                                    padx=14, pady=7, cursor='hand2',
                                    relief='flat', bd=0,
                                    activeforeground='white',
                                    activebackground=self.colors['success_hover'])
        self.save_btn.pack(side='left', padx=(0, 8))

        HoverButton(action_row,
                    text="\U0001f4ce  Export",
                    command=self.export_results,
                    font=('Segoe UI', 9, 'bold'),
                    bg=self.colors['surface'], fg='white',
                    hover_bg='#475569',
                    padx=14, pady=7, cursor='hand2',
                    relief='flat', bd=0,
                    activeforeground='white', activebackground='#475569').pack(side='left', padx=(0, 8))

        HoverButton(action_row,
                    text="\U0001f504  Clear",
                    command=self.clear_form,
                    font=('Segoe UI', 9, 'bold'),
                    bg=self.colors['warning'], fg='white',
                    hover_bg=self.colors['warning_hover'],
                    padx=14, pady=7, cursor='hand2',
                    relief='flat', bd=0,
                    activeforeground='white',
                    activebackground=self.colors['warning_hover']).pack(side='left')

    # ------------------------------------------------------------------ #
    #  Training tab                                                        #
    # ------------------------------------------------------------------ #

    def create_training_tab(self):
        """Create training tab."""
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text="\U0001f916  Model Training")

        # Controls card
        ctrl = self._card(outer, pady=(20, 0))
        self._section_label(ctrl, "Training Controls")
        btn_row = tk.Frame(ctrl, bg=self.colors['white'])
        btn_row.pack(fill='x', padx=16, pady=(4, 20))

        buttons = [
            ("\U0001f4c1  Load Dataset", self.load_dataset,  'accent'),
            ("\U0001f3af  Train Model",  self.train_model,   'success'),
            ("\U0001f4be  Save Model",   self.save_model,    'surface'),
            ("\U0001f4c2  Load Model",   self.load_model,    'surface'),
        ]
        for text, cmd, col in buttons:
            self._accent_button(btn_row, text, cmd, col).pack(side='left', padx=(0, 10))

        # Progress bar
        progress_frame = tk.Frame(outer, bg=self.colors['light'])
        progress_frame.pack(fill='x', padx=20, pady=12)
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate',
                                        length=600,
                                        style='Accent.Horizontal.TProgressbar')
        self.progress.pack(side='left')

        # Results card
        res = self._card(outer, fill='both', expand=True, pady=(0, 20))
        self._section_label(res, "Training Results")

        text_host = tk.Frame(res, bg=self.colors['white'])
        text_host.pack(fill='both', expand=True, padx=16, pady=(4, 16))

        self.training_text = tk.Text(text_host, font=('Consolas', 9),
                                     wrap=tk.WORD,
                                     bg='#0f172a', fg='#e2e8f0',
                                     relief='flat', padx=12, pady=10,
                                     insertbackground='white',
                                     selectbackground='#1e3a5f')
        self.training_text.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(text_host, orient='vertical',
                            command=self.training_text.yview)
        vsb.pack(side='right', fill='y')
        self.training_text.configure(yscrollcommand=vsb.set)

    # ------------------------------------------------------------------ #
    #  Data tab                                                            #
    # ------------------------------------------------------------------ #

    def create_data_tab(self):
        """Create data management tab with pagination."""
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text="\U0001f4ca  Data Management")

        # Table card
        tbl_card = self._card(outer, fill='both', expand=True, pady=(20, 0))

        # Header row
        hdr = tk.Frame(tbl_card, bg=self.colors['white'])
        hdr.pack(fill='x', padx=16, pady=(10, 6))

        self.dataset_info_label = tk.Label(hdr, text="No dataset loaded",
                                           font=('Segoe UI', 10, 'bold'),
                                           bg=self.colors['white'],
                                           fg=self.colors['primary'])
        self.dataset_info_label.pack(side='left')

        # Export button (far right)
        HoverButton(hdr,
                    text="\U0001f4ce  Export View",
                    command=self.export_current_view,
                    font=('Segoe UI', 9, 'bold'),
                    bg=self.colors['success'], fg='white',
                    hover_bg=self.colors['success_hover'],
                    padx=12, pady=5, cursor='hand2',
                    relief='flat', bd=0,
                    activeforeground='white',
                    activebackground=self.colors['success_hover']).pack(side='right')

        # Pagination controls
        page_ctrl = tk.Frame(hdr, bg=self.colors['white'])
        page_ctrl.pack(side='right', padx=10)

        tk.Label(page_ctrl, text="Rows/page:", bg=self.colors['white'],
                 font=('Segoe UI', 9), fg=self.colors['muted']).pack(side='left', padx=(0, 4))

        self.rows_per_page = tk.StringVar(value="100")
        rows_combo = ttk.Combobox(page_ctrl, textvariable=self.rows_per_page,
                                  values=["50", "100", "500", "1000", "5000", "All"],
                                  width=7, state='readonly', font=('Segoe UI', 9))
        rows_combo.pack(side='left', padx=(0, 10))
        rows_combo.bind('<<ComboboxSelected>>', lambda e: self.refresh_data_preview())

        self.prev_btn = HoverButton(page_ctrl, text="◀",
                                    command=self.prev_page, state='disabled',
                                    font=('Segoe UI', 9, 'bold'),
                                    bg=self.colors['secondary'], fg='white',
                                    hover_bg=self.colors['surface'],
                                    padx=10, pady=4, cursor='hand2',
                                    relief='flat', bd=0,
                                    activeforeground='white',
                                    activebackground=self.colors['surface'])
        self.prev_btn.pack(side='left', padx=(0, 4))

        self.page_var = tk.StringVar(value="Page 1")
        tk.Label(page_ctrl, textvariable=self.page_var,
                 bg=self.colors['white'], font=('Segoe UI', 9),
                 fg=self.colors['muted'], width=14).pack(side='left')

        self.next_btn = HoverButton(page_ctrl, text="▶",
                                    command=self.next_page, state='disabled',
                                    font=('Segoe UI', 9, 'bold'),
                                    bg=self.colors['secondary'], fg='white',
                                    hover_bg=self.colors['surface'],
                                    padx=10, pady=4, cursor='hand2',
                                    relief='flat', bd=0,
                                    activeforeground='white',
                                    activebackground=self.colors['surface'])
        self.next_btn.pack(side='left', padx=(4, 0))

        self._divider(tbl_card)

        # Treeview
        tree_host = tk.Frame(tbl_card, bg=self.colors['white'])
        tree_host.pack(fill='both', expand=True, padx=16, pady=(4, 0))

        self.data_tree = ttk.Treeview(tree_host, style='Custom.Treeview', height=20)
        self.data_tree.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(tree_host, orient='vertical', command=self.data_tree.yview)
        vsb.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(tbl_card, orient='horizontal', command=self.data_tree.xview)
        hsb.pack(fill='x', padx=16, pady=(0, 10))
        self.data_tree.configure(xscrollcommand=hsb.set)

        # Stats card
        stats_card = self._card(outer, pady=(12, 20))
        self._section_label(stats_card, "Dataset Statistics")

        stats_host = tk.Frame(stats_card, bg=self.colors['white'])
        stats_host.pack(fill='both', expand=True, padx=16, pady=(4, 16))

        self.stats_text = tk.Text(stats_host, height=7, font=('Consolas', 9),
                                  wrap=tk.WORD,
                                  bg='#0f172a', fg='#94a3b8',
                                  relief='flat', padx=12, pady=8,
                                  insertbackground='white')
        self.stats_text.pack(side='left', fill='both', expand=True)

        vsb2 = ttk.Scrollbar(stats_host, orient='vertical', command=self.stats_text.yview)
        vsb2.pack(side='right', fill='y')
        self.stats_text.configure(yscrollcommand=vsb2.set)

        self.current_page = 0
        self.total_pages = 0
        self.displayed_data = None

    # ------------------------------------------------------------------ #
    #  Reports tab                                                         #
    # ------------------------------------------------------------------ #

    def create_reports_tab(self):
        """Create reports tab with rich styled content area."""
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text="\U0001f4c8  Reports")

        btn_card = self._card(outer, pady=(20, 0))
        self._section_label(btn_card, "Report Controls")
        btn_row = tk.Frame(btn_card, bg=self.colors['white'])
        btn_row.pack(fill='x', padx=16, pady=(4, 20))

        report_buttons = [
            ("\U0001f4dc  Prediction History", self.show_prediction_history, 'accent'),
            ("\U0001f465  Student Records",     self.show_student_records,    'surface'),
            ("\U0001f4ca  Model Performance",   self.show_model_performance,  'success'),
            ("\U0001f4c8  Visualizations",      self.show_visualizations,     'warning'),
        ]
        for text, cmd, col in report_buttons:
            self._accent_button(btn_row, text, cmd, col).pack(side='left', padx=(0, 10))

        # Report content area — a scrollable host that can hold either
        # a Text widget or any other widget tree (tables, cards, etc.)
        rep_card = self._card(outer, fill='both', expand=True, pady=(12, 20))
        self._section_label(rep_card, "Report Viewer")
        self._divider(rep_card)

        self._report_host = tk.Frame(rep_card, bg=self.colors['white'])
        self._report_host.pack(fill='both', expand=True, padx=16, pady=(4, 16))

        # Placeholder text (replaced on each report click)
        self.report_display = tk.Text(self._report_host, font=('Consolas', 9),
                                      wrap=tk.WORD,
                                      bg='#0f172a', fg='#e2e8f0',
                                      relief='flat', padx=12, pady=10,
                                      insertbackground='white',
                                      selectbackground='#1e3a5f')
        self.report_display.pack(fill='both', expand=True)
        self.report_display.insert('1.0',
            "Select a report above to view results.\n\n"
            "  \U0001f4dc  Prediction History  —  View past predictions\n"
            "  \U0001f465  Student Records     —  Browse student data table\n"
            "  \U0001f4ca  Model Performance   —  View training metrics\n"
            "  \U0001f4c8  Visualizations      —  Interactive charts\n")

    def _clear_report_host(self):
        """Remove all children from the report host frame."""
        for w in self._report_host.winfo_children():
            w.destroy()
        self.report_display = None

    # ------------------------------------------------------------------ #
    #  Status bar                                                          #
    # ------------------------------------------------------------------ #

    def create_status_bar(self):
        """Create status bar."""
        self.status_bar = tk.Frame(self.root, bg=self.colors['secondary'], height=30)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)

        # Accent left strip
        tk.Frame(self.status_bar, bg=self.colors['accent'], width=4).pack(side='left', fill='y')

        self.status_indicator = tk.Frame(self.status_bar, bg=self.colors['success'],
                                         width=8, height=8)
        self.status_indicator.pack(side='left', padx=(8, 4), pady=11)

        self.status_label = tk.Label(self.status_bar, text="System Ready",
                                     font=('Segoe UI', 9), fg='#e2e8f0',
                                     bg=self.colors['secondary'])
        self.status_label.pack(side='left', pady=5)

        # Right: clock + version
        tk.Label(self.status_bar, text="Student Performance AI  |",
                 font=('Segoe UI', 8), fg='#64748b',
                 bg=self.colors['secondary']).pack(side='right', padx=(0, 6), pady=5)

        self.clock_label = tk.Label(self.status_bar, text="",
                                    font=('Segoe UI', 9), fg='#94a3b8',
                                    bg=self.colors['secondary'])
        self.clock_label.pack(side='right', padx=(0, 14), pady=5)
        self.update_clock()

    def update_clock(self):
        """Update clock display."""
        current_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.clock_label.config(text=current_time)
        self.root.after(1000, self.update_clock)

    def update_status(self, message, status_type='info'):
        """Update status bar."""
        color_map = {
            'info':    self.colors['accent'],
            'success': self.colors['success'],
            'warning': self.colors['warning'],
            'error':   self.colors['danger'],
        }
        self.status_indicator.configure(bg=color_map.get(status_type, self.colors['accent']))
        self.status_label.config(text=message)

    # ------------------------------------------------------------------ #
    #  Data helpers (unchanged logic)                                      #
    # ------------------------------------------------------------------ #

    def refresh_data_preview(self):
        """Refresh data preview with current pagination settings."""
        if self.current_data is not None:
            rows, cols = self.current_data.shape
            self.dataset_info_label.config(
                text=f"\U0001f4ca  Dataset: {rows:,} rows \u00d7 {cols} columns")

            rows_per_page_str = self.rows_per_page.get()
            rows_per_page = rows if rows_per_page_str == "All" else int(rows_per_page_str)

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
            self.next_btn.config(
                state='normal' if self.current_page < self.total_pages - 1 else 'disabled')
            self.update_status(
                f"Showing rows {start_idx+1:,}\u2013{end_idx:,} of {rows:,}", 'info')

    def update_treeview(self):
        """Update treeview with current displayed data."""
        if self.displayed_data is not None and not self.displayed_data.empty:
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            columns = list(self.displayed_data.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=110, minwidth=60)
            for idx, row in self.displayed_data.iterrows():
                values = [str(row[col])[:50] for col in columns]
                self.data_tree.insert('', 'end', values=values)
            self.update_statistics()

    def update_statistics(self):
        """Update statistics display."""
        if self.current_data is not None:
            self.stats_text.config(state='normal')
            self.stats_text.delete(1.0, tk.END)
            rows, cols = self.current_data.shape
            self.stats_text.insert(tk.END, f"DATASET OVERVIEW\n{'=' * 50}\n\n")
            self.stats_text.insert(tk.END, f"Total Rows    : {rows:,}\n")
            self.stats_text.insert(tk.END, f"Total Columns : {cols}\n\n")

            missing = self.current_data.isnull().sum()
            if missing.sum() > 0:
                self.stats_text.insert(tk.END, "Missing Values:\n")
                for col, count in missing[missing > 0].items():
                    self.stats_text.insert(
                        tk.END, f"  {col}: {count:,} ({count/rows*100:.1f}%)\n")

            numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.stats_text.insert(tk.END, f"\nNumeric Columns: {len(numeric_cols)}\n")
                for col in numeric_cols[:5]:
                    self.stats_text.insert(
                        tk.END,
                        f"  {col}: mean={self.current_data[col].mean():.2f}"
                        f"  std={self.current_data[col].std():.2f}\n")

    def prev_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_data_preview()

    def next_page(self):
        """Go to next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.refresh_data_preview()

    def export_current_view(self):
        """Export current view to CSV."""
        if self.displayed_data is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".csv")
            if filename:
                self.displayed_data.to_csv(filename, index=False)
                self.update_status(f"Exported {len(self.displayed_data)} rows", 'success')
                messagebox.showinfo("Success", f"Exported {len(self.displayed_data)} rows")

    # ------------------------------------------------------------------ #
    #  Dataset / model actions (logic unchanged)                          #
    # ------------------------------------------------------------------ #

    def load_dataset(self):
        """Load dataset from CSV."""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            try:
                self.current_data = pd.read_csv(filename)
                self.current_page = 0
                self.refresh_data_preview()
                self.update_status(
                    f"Dataset loaded: {self.current_data.shape[0]:,} rows", 'success')
                if 'Dataset Status' in self.status_cards:
                    self.status_cards['Dataset Status'].config(
                        text=f"Loaded ({self.current_data.shape[0]:,} rows)")
                messagebox.showinfo(
                    "Success",
                    f"Dataset loaded successfully!\nShape: {self.current_data.shape}")
            except Exception as e:
                self.update_status(f"Failed to load dataset: {e}", 'error')
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        """Train the model."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        self.update_status("Starting model training...", 'info')
        self.progress.start()
        self.training_text.config(state='normal')
        self.training_text.delete(1.0, tk.END)
        self.training_text.insert(tk.END, "Starting model training...\n\n")
        thread = threading.Thread(target=self._training_worker, daemon=True)
        thread.start()

    def _training_worker(self):
        """Training worker thread."""
        try:
            self._update_training_text("Preprocessing data...\n")
            result = self.preprocessor.prepare_data(self.current_data)
            if isinstance(result, tuple) and len(result) == 2:
                X, y = result
            else:
                self._update_training_text("Error: No target column found!\n")
                self._stop_progress()
                return

            self._update_training_text(f"Data shape: {X.shape}\n")
            self._update_training_text("Training models...\n\n")

            results, X_test, y_test = self.model_trainer.train_and_evaluate(X, y)

            self._update_training_text("Model Performance Results\n" + "=" * 50 + "\n\n")
            for name, metrics in results.items():
                self._update_training_text(
                    f"  {name}:\n"
                    f"    RMSE    : {metrics['rmse']:.4f}\n"
                    f"    R2 Score: {metrics['r2']:.4f}\n\n")

            self._update_training_text("=" * 50 + "\n")
            self._update_training_text(f"Best Model  : {self.model_trainer.best_model_name}\n")
            self._update_training_text(
                f"Best R2     : {self.model_trainer.model_metrics['r2']:.4f}\n")

            self.is_model_trained = True
            self._update_training_text("\nModel training completed successfully!\n")
            self.update_status("Model training completed!", 'success')

            if 'Model Status' in self.status_cards:
                self.status_cards['Model Status'].config(text="Trained")
                self.status_cards['Prediction Ready'].config(text="Yes")

            self.root.after(
                0, lambda: messagebox.showinfo(
                    "Success", "Model training completed successfully!"))
        except Exception as e:
            self._update_training_text(f"\nTraining failed: {e}\n")
            self.update_status(f"Training failed: {e}", 'error')
        finally:
            self._stop_progress()

    def _update_training_text(self, text):
        """Update training text."""
        self.root.after(0, lambda: self.training_text.insert(tk.END, text))
        self.root.after(0, lambda: self.training_text.see(tk.END))

    def _stop_progress(self):
        """Stop progress bar."""
        self.root.after(0, self.progress.stop)

    # ------------------------------------------------------------------ #
    #  Prediction (logic fully unchanged)                                  #
    # ------------------------------------------------------------------ #

    def predict_performance(self):
        """Make prediction with detailed property analysis and improvement suggestions."""
        if not self.is_model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        try:
            student_name = self.input_vars['name'].get()
            if not student_name or student_name == "Enter student name":
                student_name = "Student"

            student_data = {}
            student_data['age'] = float(self.input_vars['age'].get()) if self.input_vars['age'].get() else 16
            student_data['study_hours'] = float(self.input_vars['study_hours'].get()) if self.input_vars['study_hours'].get() else 5
            student_data['attendance_percentage'] = float(self.input_vars['attendance_percentage'].get()) if self.input_vars['attendance_percentage'].get() else 85
            student_data['math_score'] = float(self.input_vars['math_score'].get()) if self.input_vars['math_score'].get() else 70
            student_data['science_score'] = float(self.input_vars['science_score'].get()) if self.input_vars['science_score'].get() else 70
            student_data['english_score'] = float(self.input_vars['english_score'].get()) if self.input_vars['english_score'].get() else 70
            student_data['internet_access'] = int(float(self.input_vars['internet_access'].get())) if self.input_vars['internet_access'].get() else 1
            student_data['extra_activities'] = int(float(self.input_vars['extra_activities'].get())) if self.input_vars['extra_activities'].get() else 1

            travel_time_value = self.input_vars['travel_time'].get()
            travel_time_map = {'<15 min': 0, '15-30 min': 1, '30-60 min': 2, '>60 min': 3}
            student_data['travel_time'] = travel_time_map.get(travel_time_value, 0)

            student_data['gender'] = self.input_vars['gender'].get() if self.input_vars['gender'].get() else "male"
            student_data['school_type'] = self.input_vars['school_type'].get() if self.input_vars['school_type'].get() else "public"
            student_data['parent_education'] = self.input_vars['parent_education'].get() if self.input_vars['parent_education'].get() else "graduate"
            student_data['study_method'] = self.input_vars['study_method'].get() if self.input_vars['study_method'].get() else "textbook"

            math_score = student_data['math_score']
            science_score = student_data['science_score']
            english_score = student_data['english_score']
            study_hours = student_data['study_hours']
            attendance = student_data['attendance_percentage']
            travel_time_val = student_data['travel_time']
            extra_activities = student_data['extra_activities']
            internet_access = student_data['internet_access']

            input_df = pd.DataFrame([student_data])
            X_input = self.preprocessor.prepare_features_only(input_df)
            predicted_score = self.model_trainer.predict(X_input)[0]
            predicted_score = max(0, min(100, predicted_score))

            travel_time_text = {0: "Less than 15 minutes", 1: "15-30 minutes",
                                 2: "30-60 minutes", 3: "More than 60 minutes"}.get(travel_time_val, "Unknown")

            if predicted_score >= 85:
                overall_category, overall_icon, overall_color = "EXCELLENT", "\U0001f31f", self.colors['success']
            elif predicted_score >= 70:
                overall_category, overall_icon, overall_color = "GOOD", "\U0001f44d", self.colors['accent']
            elif predicted_score >= 50:
                overall_category, overall_icon, overall_color = "AVERAGE", "\U0001f4da", self.colors['warning']
            else:
                overall_category, overall_icon, overall_color = "NEEDS IMPROVEMENT", "\u26a0\ufe0f", self.colors['danger']

            strengths, weaknesses = [], []

            for subj_score, label, icon in [(math_score, 'Mathematics', '\U0001f4d0'),
                                             (science_score, 'Science', '\U0001f52c'),
                                             (english_score, 'English', '\U0001f4dd')]:
                if subj_score >= 75:
                    strengths.append(f"{icon} {label}: {subj_score:.0f}/100 - Strong performance!")
                elif subj_score >= 60:
                    strengths.append(f"{icon} {label}: {subj_score:.0f}/100 - Satisfactory")
                else:
                    weaknesses.append(f"{icon} {label}: {subj_score:.0f}/100 - Needs significant improvement")

            if study_hours >= 6:
                strengths.append(f"\U0001f4da Study Hours: {study_hours:.1f} hours/day - Excellent dedication!")
            elif study_hours >= 4:
                strengths.append(f"\U0001f4da Study Hours: {study_hours:.1f} hours/day - Good consistency")
            else:
                weaknesses.append(f"\U0001f4da Study Hours: {study_hours:.1f} hours/day - Below recommended (aim for 5-6 hours)")

            if attendance >= 90:
                strengths.append(f"\U0001f4c8 Attendance: {attendance:.0f}% - Outstanding!")
            elif attendance >= 75:
                strengths.append(f"\U0001f4c8 Attendance: {attendance:.0f}% - Good")
            else:
                weaknesses.append(f"\U0001f4c8 Attendance: {attendance:.0f}% - Low attendance affects learning")

            if travel_time_val <= 1:
                strengths.append(f"\U0001f68c Travel Time: {travel_time_text} - Convenient")
            elif travel_time_val == 2:
                weaknesses.append(f"\U0001f68c Travel Time: {travel_time_text} - Consider using travel time for review")
            else:
                weaknesses.append(f"\U0001f68c Travel Time: {travel_time_text} - Long commute may affect study time")

            if extra_activities == 1:
                strengths.append("\u2b50 Extra Activities: Participating - Good for holistic development")
            else:
                weaknesses.append("\u2b50 Extra Activities: Not participating - Consider joining activities")

            if internet_access == 1:
                strengths.append("\U0001f310 Internet Access: Available - Good for online learning resources")
            else:
                weaknesses.append("\U0001f310 Internet Access: Limited - Consider library resources")

            improvement_tips = []
            if predicted_score < 70:
                improvement_tips.append("\U0001f3af Set daily study goals and track progress")
            if math_score < 60:
                improvement_tips.append("\U0001f4d0 Math: Practice daily, focus on weak topics, use online tutorials")
            if science_score < 60:
                improvement_tips.append("\U0001f52c Science: Create concept maps, watch educational videos, join study groups")
            if english_score < 60:
                improvement_tips.append("\U0001f4dd English: Read regularly, practice writing, expand vocabulary")
            if study_hours < 5:
                improvement_tips.append("\u23f0 Increase study time gradually - aim for 5-6 hours daily")
            if attendance < 80:
                improvement_tips.append("\U0001f4c5 Improve attendance - regular classes are crucial for success")
            if travel_time_val >= 2:
                improvement_tips.append("\U0001f3a7 Use travel time productively - listen to educational podcasts")
            if extra_activities == 0:
                improvement_tips.append("\U0001f91d Join extracurricular activities to develop soft skills")

            # Render results — hide placeholder, show result text
            if hasattr(self, '_result_placeholder') and self._result_placeholder.winfo_exists():
                self._result_placeholder.pack_forget()
            if not self.result_text.winfo_manager():
                self.result_text.pack(fill='both', expand=True, padx=10, pady=(0, 4),
                                      before=self.save_btn.master)
            self.result_text.delete(1.0, tk.END)

            self.result_text.tag_configure('title',    font=('Segoe UI', 14, 'bold'), foreground=self.colors['primary'])
            self.result_text.tag_configure('header',   font=('Segoe UI', 11, 'bold'), foreground=self.colors['secondary'])
            self.result_text.tag_configure('score_big',font=('Segoe UI', 28, 'bold'), foreground=self.colors['accent'])
            self.result_text.tag_configure('category', font=('Segoe UI', 13, 'bold'), foreground=overall_color)
            self.result_text.tag_configure('strength', font=('Segoe UI', 9),          foreground=self.colors['success'])
            self.result_text.tag_configure('weakness', font=('Segoe UI', 9),          foreground=self.colors['danger'])
            self.result_text.tag_configure('tip',      font=('Segoe UI', 9),          foreground=self.colors['text_soft'])
            self.result_text.tag_configure('divider',  font=('Segoe UI', 9),          foreground=self.colors['border'])
            self.result_text.tag_configure('meta',     font=('Segoe UI', 9),          foreground=self.colors['muted'])

            self.result_text.insert(tk.END, "=" * 56 + "\n", 'divider')
            self.result_text.insert(tk.END, "\U0001f393 STUDENT PERFORMANCE ANALYSIS\n", 'title')
            self.result_text.insert(tk.END, "=" * 56 + "\n\n", 'divider')

            self.result_text.insert(tk.END, f"\U0001f464 Student : {student_name}\n", 'meta')
            self.result_text.insert(tk.END, f"\U0001f4c5 Date    : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n\n", 'meta')

            self.result_text.insert(tk.END, "\u2500" * 56 + "\n", 'divider')
            self.result_text.insert(tk.END, "\U0001f3af  PREDICTED OVERALL SCORE\n", 'header')
            self.result_text.insert(tk.END, f"{predicted_score:.1f}", 'score_big')
            self.result_text.insert(tk.END, " / 100\n\n", 'header')
            self.result_text.insert(tk.END, f"Category: {overall_icon}  {overall_category}\n", 'category')
            self.result_text.insert(tk.END, "\u2500" * 56 + "\n\n", 'divider')

            self.result_text.insert(tk.END, "\U0001f4cb  SUBJECT-WISE BREAKDOWN\n", 'header')
            self.result_text.insert(tk.END, "\u2500" * 40 + "\n", 'divider')
            for subj_name, subj_score, icon in [("Mathematics", math_score, "\U0001f4d0"),
                                                  ("Science",     science_score, "\U0001f52c"),
                                                  ("English",     english_score, "\U0001f4dd")]:
                bar_len = int(subj_score / 10)
                bar = "\u2588" * bar_len + "\u2591" * (10 - bar_len)
                self.result_text.insert(tk.END, f"{icon} {subj_name:<14} {subj_score:5.0f}/100  [{bar}]\n")
            self.result_text.insert(tk.END, "\n")

            self.result_text.insert(tk.END, "\U0001f4da  STUDY HABITS\n", 'header')
            self.result_text.insert(tk.END, "\u2500" * 40 + "\n", 'divider')
            h_bar_len = min(int(study_hours / 8 * 10), 10)
            h_bar = "\u2588" * h_bar_len + "\u2591" * (10 - h_bar_len)
            self.result_text.insert(tk.END, f"\u23f0 Study Hours  : {study_hours:.1f} hrs/day  [{h_bar}]  (target 6+)\n")
            a_bar_len = min(int(attendance / 10), 10)
            a_bar = "\u2588" * a_bar_len + "\u2591" * (10 - a_bar_len)
            self.result_text.insert(tk.END, f"\U0001f4c8 Attendance   : {attendance:.0f}%       [{a_bar}]  (target 90%+)\n")
            self.result_text.insert(tk.END, f"\U0001f68c Travel Time  : {travel_time_text}\n")
            self.result_text.insert(tk.END, f"\u2b50 Extra Acts   : {'Yes \u2705' if extra_activities == 1 else 'No \u274c'}\n")
            self.result_text.insert(tk.END, f"\U0001f310 Internet     : {'Available \u2705' if internet_access == 1 else 'Limited \u274c'}\n\n")

            if strengths:
                self.result_text.insert(tk.END, "\u2705  STRENGTHS\n", 'header')
                self.result_text.insert(tk.END, "\u2500" * 40 + "\n", 'divider')
                for s in strengths[:5]:
                    self.result_text.insert(tk.END, f"  {s}\n", 'strength')
                self.result_text.insert(tk.END, "\n")

            if weaknesses:
                self.result_text.insert(tk.END, "\u26a0\ufe0f  AREAS FOR IMPROVEMENT\n", 'header')
                self.result_text.insert(tk.END, "\u2500" * 40 + "\n", 'divider')
                for w in weaknesses[:5]:
                    self.result_text.insert(tk.END, f"  {w}\n", 'weakness')
                self.result_text.insert(tk.END, "\n")

            self.result_text.insert(tk.END, "\U0001f4a1  RECOMMENDATIONS\n", 'header')
            self.result_text.insert(tk.END, "\u2500" * 40 + "\n", 'divider')
            if improvement_tips:
                for i, tip in enumerate(improvement_tips[:6], 1):
                    self.result_text.insert(tk.END, f"  {i}. {tip}\n", 'tip')
            else:
                self.result_text.insert(tk.END, "  \U0001f389 Excellent work! Keep maintaining your good habits!\n", 'strength')

            self.result_text.insert(tk.END, "\n" + "=" * 56 + "\n", 'divider')
            self.result_text.insert(tk.END, "\U0001f3af Focus on weak areas and maintain your strengths!\n", 'header')
            self.result_text.insert(tk.END, "=" * 56, 'divider')

            self.last_prediction = {
                'name': student_name, 'score': predicted_score,
                'category': overall_category,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'math_score': math_score, 'science_score': science_score,
                'english_score': english_score, 'study_hours': study_hours,
                'attendance': attendance, 'travel_time': travel_time_text,
                'extra_activities': extra_activities, 'internet_access': internet_access,
                'data': student_data,
            }

            self.save_btn.config(state='normal')
            self.update_status("Prediction completed successfully!", 'success')

        except Exception as e:
            self.update_status(f"Prediction failed: {e}", 'error')
            messagebox.showerror(
                "Error", f"Prediction failed: {e}\n\nPlease make sure the model is trained properly.")
            logger.error(f"Prediction error: {traceback.format_exc()}")

    def clear_form(self):
        """Clear all input fields."""
        defaults = {
            'name': "Student", 'age': 16, 'gender': 'male',
            'school_type': 'public', 'parent_education': 'graduate',
            'study_hours': 5, 'attendance_percentage': 85,
            'internet_access': '1', 'travel_time': '<15 min',
            'extra_activities': '1', 'study_method': 'textbook',
            'math_score': 70, 'science_score': 70, 'english_score': 70,
        }
        for key, val in defaults.items():
            self.input_vars[key].set(val)
        self.result_text.delete(1.0, tk.END)
        # Hide result text, show placeholder
        self.result_text.pack_forget()
        if hasattr(self, '_result_placeholder') and self._result_placeholder.winfo_exists():
            self._result_placeholder.pack(fill='both', expand=True)
        self.update_status("Form cleared", 'info')

    def save_prediction(self):
        """Save prediction."""
        if self.last_prediction:
            messagebox.showinfo("Success", "Prediction saved!")
            self.save_btn.config(state='disabled')

    def save_model(self):
        """Save model."""
        if self.is_model_trained:
            self.model_trainer.save_model()
            self.preprocessor.save_preprocessors()
            self.update_status("Model saved!", 'success')
            messagebox.showinfo("Success", "Model saved successfully!")

    def load_model(self):
        """Load model."""
        filename = filedialog.askopenfilename(filetypes=[("PKL files", "*.pkl")])
        if filename:
            try:
                import joblib
                model_data = joblib.load(filename)
                self.model_trainer.best_model = model_data['model']
                self.model_trainer.best_model_name = model_data.get('model_name', 'Loaded')
                self.model_trainer.model_metrics = model_data.get('metrics', {})
                self.preprocessor.load_preprocessors()
                self.is_model_trained = True
                self.update_status("Model loaded!", 'success')
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")

    def export_results(self):
        """Export results."""
        if self.last_prediction:
            filename = filedialog.asksaveasfilename(defaultextension=".csv")
            if filename:
                df = pd.DataFrame([self.last_prediction])
                df.to_csv(filename, index=False)
                self.update_status("Results exported!", 'success')
                messagebox.showinfo("Success", "Results exported!")

    # ------------------------------------------------------------------ #
    #  Reports                                                             #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Report: Prediction History (card-style)                             #
    # ------------------------------------------------------------------ #

    def show_prediction_history(self):
        """Show prediction history as styled cards."""
        self._clear_report_host()

        if not (hasattr(self, 'last_prediction') and self.last_prediction):
            self._show_report_placeholder("No predictions yet. Run a prediction first.")
            return

        p = self.last_prediction

        # Scrollable canvas
        canvas = tk.Canvas(self._report_host, bg=self.colors['white'],
                           highlightthickness=0)
        vsb = ttk.Scrollbar(self._report_host, orient='vertical',
                            command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        scroll_frame = tk.Frame(canvas, bg=self.colors['white'])
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        scroll_frame.bind('<Configure>',
                          lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        # ---- Header banner ---- #
        hdr = tk.Frame(scroll_frame, bg=self.colors['accent'], height=56)
        hdr.pack(fill='x', padx=8, pady=(8, 0))
        hdr.pack_propagate(False)
        tk.Label(hdr, text="\U0001f4dc  Latest Prediction Result",
                 font=('Segoe UI', 13, 'bold'), fg='white',
                 bg=self.colors['accent']).pack(side='left', padx=16, pady=12)
        tk.Label(hdr, text=p.get('date', ''),
                 font=('Segoe UI', 9), fg='#dbeafe',
                 bg=self.colors['accent']).pack(side='right', padx=16, pady=12)

        # ---- Score hero card ---- #
        hero_shadow = tk.Frame(scroll_frame, bg='#cbd5e1', padx=1, pady=1)
        hero_shadow.pack(fill='x', padx=8, pady=12)
        hero = tk.Frame(hero_shadow, bg=self.colors['white'])
        hero.pack(fill='both', expand=True)

        score_color = self.colors['success'] if p['score'] >= 70 else (
            self.colors['warning'] if p['score'] >= 50 else self.colors['danger'])

        tk.Frame(hero, bg=score_color, height=5).pack(fill='x')

        score_row = tk.Frame(hero, bg=self.colors['white'])
        score_row.pack(fill='x', padx=20, pady=16)

        tk.Label(score_row, text=f"{p['score']:.1f}",
                 font=('Segoe UI', 42, 'bold'), fg=score_color,
                 bg=self.colors['white']).pack(side='left')
        tk.Label(score_row, text="/ 100",
                 font=('Segoe UI', 16), fg=self.colors['muted'],
                 bg=self.colors['white']).pack(side='left', padx=(4, 20), pady=(18, 0))

        cat_frame = tk.Frame(score_row, bg=self.colors['white'])
        cat_frame.pack(side='left', padx=20, pady=(10, 0))
        tk.Label(cat_frame, text=p['category'],
                 font=('Segoe UI', 12, 'bold'), fg='white',
                 bg=score_color, padx=14, pady=6).pack()

        tk.Label(score_row, text=f"Student: {p.get('name', 'N/A')}",
                 font=('Segoe UI', 11), fg=self.colors['text_soft'],
                 bg=self.colors['white']).pack(side='right', padx=10)

        # ---- Subject score cards ---- #
        subj_row = tk.Frame(scroll_frame, bg=self.colors['white'])
        subj_row.pack(fill='x', padx=8, pady=(0, 8))

        subjects = [
            ("\U0001f4d0 Mathematics", p.get('math_score', 0)),
            ("\U0001f52c Science",     p.get('science_score', 0)),
            ("\U0001f4dd English",     p.get('english_score', 0)),
        ]
        for i, (subj_name, subj_score) in enumerate(subjects):
            tile_shadow = tk.Frame(subj_row, bg='#cbd5e1', padx=1, pady=1)
            tile_shadow.grid(row=0, column=i, padx=6, pady=4, sticky='nsew')
            subj_row.grid_columnconfigure(i, weight=1)

            tile = tk.Frame(tile_shadow, bg=self.colors['white'])
            tile.pack(fill='both', expand=True)

            bar_color = self.colors['success'] if subj_score >= 75 else (
                self.colors['accent'] if subj_score >= 60 else self.colors['danger'])
            tk.Frame(tile, bg=bar_color, height=4).pack(fill='x')

            tk.Label(tile, text=subj_name, font=('Segoe UI', 9),
                     bg=self.colors['white'], fg=self.colors['muted']).pack(pady=(10, 2))
            tk.Label(tile, text=f"{subj_score:.0f}",
                     font=('Segoe UI', 24, 'bold'), fg=bar_color,
                     bg=self.colors['white']).pack()
            tk.Label(tile, text="/ 100", font=('Segoe UI', 9),
                     bg=self.colors['white'], fg=self.colors['muted']).pack(pady=(0, 4))

            # Visual bar
            bar_host = tk.Frame(tile, bg=self.colors['border'], height=8)
            bar_host.pack(fill='x', padx=12, pady=(0, 12))
            bar_host.pack_propagate(False)
            fill_w = max(subj_score, 1)
            tk.Frame(bar_host, bg=bar_color, width=fill_w).pack(side='left', fill='y')

        # ---- Detail info grid ---- #
        detail_shadow = tk.Frame(scroll_frame, bg='#cbd5e1', padx=1, pady=1)
        detail_shadow.pack(fill='x', padx=8, pady=(4, 8))
        detail = tk.Frame(detail_shadow, bg=self.colors['white'])
        detail.pack(fill='both', expand=True)

        tk.Frame(detail, bg=self.colors['secondary'], height=3).pack(fill='x')

        info_items = [
            ("\u23f0 Study Hours",    f"{p.get('study_hours', 0):.1f} hrs/day"),
            ("\U0001f4c8 Attendance",  f"{p.get('attendance', 0):.0f}%"),
            ("\U0001f68c Travel Time", p.get('travel_time', 'N/A')),
            ("\u2b50 Extra Activities", "Yes" if p.get('extra_activities') == 1 else "No"),
            ("\U0001f310 Internet",     "Available" if p.get('internet_access') == 1 else "Limited"),
        ]
        for i, (label, value) in enumerate(info_items):
            row_bg = self.colors['white'] if i % 2 == 0 else '#f8fafc'
            r = tk.Frame(detail, bg=row_bg)
            r.pack(fill='x')
            tk.Label(r, text=label, font=('Segoe UI', 9, 'bold'),
                     bg=row_bg, fg=self.colors['text_soft'],
                     width=20, anchor='w').pack(side='left', padx=(16, 0), pady=8)
            tk.Label(r, text=value, font=('Segoe UI', 10),
                     bg=row_bg, fg=self.colors['primary'],
                     anchor='w').pack(side='left', padx=10, pady=8)

    # ------------------------------------------------------------------ #
    #  Report: Student Records (styled Treeview table)                     #
    # ------------------------------------------------------------------ #

    def show_student_records(self):
        """Show student records as a styled, sortable table."""
        self._clear_report_host()

        if self.current_data is None:
            self._show_report_placeholder("No dataset loaded. Load a CSV first.")
            return

        df = self.current_data.head(50)

        # ---- Summary banner ---- #
        banner_shadow = tk.Frame(self._report_host, bg='#cbd5e1', padx=1, pady=1)
        banner_shadow.pack(fill='x', pady=(0, 8))
        banner = tk.Frame(banner_shadow, bg=self.colors['secondary'])
        banner.pack(fill='both', expand=True)

        summary_items = [
            (f"{len(self.current_data):,}", "Total Students"),
            (f"{len(self.current_data.columns)}", "Columns"),
            (f"{self.current_data.select_dtypes(include=[np.number]).shape[1]}", "Numeric"),
            (f"{self.current_data.select_dtypes(exclude=[np.number]).shape[1]}", "Categorical"),
        ]
        for val, label in summary_items:
            col = tk.Frame(banner, bg=self.colors['secondary'])
            col.pack(side='left', expand=True, padx=16, pady=12)
            tk.Label(col, text=val, font=('Segoe UI', 18, 'bold'),
                     fg='white', bg=self.colors['secondary']).pack()
            tk.Label(col, text=label, font=('Segoe UI', 9),
                     fg='#94a3b8', bg=self.colors['secondary']).pack()

        # ---- Table ---- #
        tbl_shadow = tk.Frame(self._report_host, bg='#cbd5e1', padx=1, pady=1)
        tbl_shadow.pack(fill='both', expand=True)

        tbl = tk.Frame(tbl_shadow, bg=self.colors['white'])
        tbl.pack(fill='both', expand=True)

        # Style for report treeview
        style = ttk.Style()
        style.configure('Report.Treeview',
                        font=('Segoe UI', 9),
                        rowheight=28,
                        background=self.colors['white'],
                        fieldbackground=self.colors['white'],
                        foreground=self.colors['text'])
        style.configure('Report.Treeview.Heading',
                        font=('Segoe UI', 9, 'bold'),
                        background=self.colors['secondary'],
                        foreground='white',
                        padding=[8, 6])
        style.map('Report.Treeview',
                  background=[('selected', '#dbeafe')],
                  foreground=[('selected', self.colors['accent'])])

        tree_host = tk.Frame(tbl, bg=self.colors['white'])
        tree_host.pack(fill='both', expand=True, padx=2, pady=2)

        tree = ttk.Treeview(tree_host, style='Report.Treeview', height=18)
        tree.pack(side='left', fill='both', expand=True)

        vsb = ttk.Scrollbar(tree_host, orient='vertical', command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(tbl, orient='horizontal', command=tree.xview)
        hsb.pack(fill='x', padx=2, pady=(0, 2))
        tree.configure(xscrollcommand=hsb.set)

        columns = list(df.columns)
        tree['columns'] = columns
        tree['show'] = 'headings'

        for col in columns:
            tree.heading(col, text=col.upper().replace('_', ' '))
            tree.column(col, width=110, minwidth=60)

        # Alternating row tags
        tree.tag_configure('even', background=self.colors['white'])
        tree.tag_configure('odd',  background='#f8fafc')

        for idx, (_, row) in enumerate(df.iterrows()):
            values = [str(row[c])[:60] for c in columns]
            tag = 'even' if idx % 2 == 0 else 'odd'
            tree.insert('', 'end', values=values, tags=(tag,))

        # Footer
        footer = tk.Frame(tbl, bg=self.colors['light'])
        footer.pack(fill='x')
        tk.Label(footer,
                 text=f"Showing {len(df)} of {len(self.current_data):,} records  |  "
                      f"Scroll horizontally for more columns",
                 font=('Segoe UI', 8), fg=self.colors['muted'],
                 bg=self.colors['light']).pack(side='left', padx=12, pady=6)

    # ------------------------------------------------------------------ #
    #  Report: Model Performance (metric cards + visual bars)              #
    # ------------------------------------------------------------------ #

    def show_model_performance(self):
        """Show model performance with styled metric cards and visual bars."""
        self._clear_report_host()

        if not self.model_trainer.best_model_name:
            self._show_report_placeholder("No model trained yet. Train a model first.")
            return

        # Scrollable canvas
        canvas = tk.Canvas(self._report_host, bg=self.colors['white'],
                           highlightthickness=0)
        vsb = ttk.Scrollbar(self._report_host, orient='vertical',
                            command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        scroll_frame = tk.Frame(canvas, bg=self.colors['white'])
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')
        scroll_frame.bind('<Configure>',
                          lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        # ---- Best model hero ---- #
        hero_shadow = tk.Frame(scroll_frame, bg='#cbd5e1', padx=1, pady=1)
        hero_shadow.pack(fill='x', padx=8, pady=(8, 12))
        hero = tk.Frame(hero_shadow, bg=self.colors['success'])
        hero.pack(fill='both', expand=True)

        tk.Label(hero, text="\U0001f3c6  Best Model",
                 font=('Segoe UI', 10), fg='#d1fae5',
                 bg=self.colors['success']).pack(pady=(12, 0), padx=16, anchor='w')
        tk.Label(hero, text=self.model_trainer.best_model_name,
                 font=('Segoe UI', 22, 'bold'), fg='white',
                 bg=self.colors['success']).pack(pady=(0, 12), padx=16, anchor='w')

        # ---- Metric cards row ---- #
        metrics_row = tk.Frame(scroll_frame, bg=self.colors['white'])
        metrics_row.pack(fill='x', padx=8, pady=(0, 12))

        r2 = self.model_trainer.model_metrics.get('r2', 0)
        rmse = self.model_trainer.model_metrics.get('rmse', 0)

        metric_cards = [
            ("R\u00b2 Score", f"{r2:.4f}", r2 * 100,
             self.colors['success'] if r2 >= 0.8 else (
                 self.colors['accent'] if r2 >= 0.6 else self.colors['warning'])),
            ("RMSE", f"{rmse:.4f}", max(0, 100 - rmse),
             self.colors['accent'] if rmse < 10 else (
                 self.colors['warning'] if rmse < 20 else self.colors['danger'])),
        ]

        for i, (name, value, pct, color) in enumerate(metric_cards):
            tile_shadow = tk.Frame(metrics_row, bg='#cbd5e1', padx=1, pady=1)
            tile_shadow.grid(row=0, column=i, padx=6, pady=4, sticky='nsew')
            metrics_row.grid_columnconfigure(i, weight=1)

            tile = tk.Frame(tile_shadow, bg=self.colors['white'])
            tile.pack(fill='both', expand=True)

            tk.Frame(tile, bg=color, height=4).pack(fill='x')

            tk.Label(tile, text=name, font=('Segoe UI', 9),
                     bg=self.colors['white'], fg=self.colors['muted']).pack(pady=(12, 2))
            tk.Label(tile, text=value, font=('Segoe UI', 24, 'bold'),
                     fg=color, bg=self.colors['white']).pack()

            # Visual progress bar
            bar_host = tk.Frame(tile, bg=self.colors['border'], height=10)
            bar_host.pack(fill='x', padx=14, pady=(8, 14))
            bar_host.pack_propagate(False)
            fill_pct = max(min(pct, 100), 0)
            tk.Frame(bar_host, bg=color, width=fill_pct).pack(side='left', fill='y')

        # ---- All models comparison table ---- #
        if hasattr(self.model_trainer, 'results') and self.model_trainer.results:
            tbl_shadow = tk.Frame(scroll_frame, bg='#cbd5e1', padx=1, pady=1)
            tbl_shadow.pack(fill='x', padx=8, pady=(4, 8))
            tbl = tk.Frame(tbl_shadow, bg=self.colors['white'])
            tbl.pack(fill='both', expand=True)

            tk.Frame(tbl, bg=self.colors['secondary'], height=3).pack(fill='x')

            tk.Label(tbl, text="All Models Comparison",
                     font=('Segoe UI', 11, 'bold'),
                     bg=self.colors['white'], fg=self.colors['primary']).pack(
                         anchor='w', padx=16, pady=(12, 4))

            style = ttk.Style()
            style.configure('Model.Treeview',
                            font=('Segoe UI', 9),
                            rowheight=30,
                            background=self.colors['white'],
                            fieldbackground=self.colors['white'],
                            foreground=self.colors['text'])
            style.configure('Model.Treeview.Heading',
                            font=('Segoe UI', 9, 'bold'),
                            background=self.colors['secondary'],
                            foreground='white',
                            padding=[8, 6])
            style.map('Model.Treeview',
                      background=[('selected', '#dbeafe')],
                      foreground=[('selected', self.colors['accent'])])

            tree_host = tk.Frame(tbl, bg=self.colors['white'])
            tree_host.pack(fill='both', expand=True, padx=12, pady=(4, 12))

            tree = ttk.Treeview(tree_host, style='Model.Treeview',
                                columns=['model', 'rmse', 'r2', 'rank'],
                                height=6, show='headings')
            tree.heading('model', text='MODEL')
            tree.heading('rmse',  text='RMSE')
            tree.heading('r2',    text='R\u00b2 SCORE')
            tree.heading('rank',  text='RANK')
            tree.column('model', width=200, minwidth=120)
            tree.column('rmse',  width=120, minwidth=80)
            tree.column('r2',    width=120, minwidth=80)
            tree.column('rank',  width=80,  minwidth=60)

            tree.tag_configure('best', background='#ecfdf5',
                               foreground=self.colors['success'])
            tree.tag_configure('even', background=self.colors['white'])
            tree.tag_configure('odd',  background='#f8fafc')

            sorted_results = sorted(
                self.model_trainer.results.items(),
                key=lambda x: x[1]['r2'], reverse=True)

            for rank, (name, m) in enumerate(sorted_results, 1):
                tag = 'best' if name == self.model_trainer.best_model_name else (
                    'even' if rank % 2 == 0 else 'odd')
                tree.insert('', 'end',
                            values=(name, f"{m['rmse']:.4f}",
                                    f"{m['r2']:.4f}", f"#{rank}"),
                            tags=(tag,))

            tree.pack(side='left', fill='both', expand=True)

    # ------------------------------------------------------------------ #
    #  Report: Visualizations (multi-chart dashboard)                      #
    # ------------------------------------------------------------------ #

    def show_visualizations(self):
        """Show visualizations in a multi-chart dashboard window."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data available!")
            return

        viz_window = tk.Toplevel(self.root)
        viz_window.title("Data Visualizations Dashboard")
        viz_window.geometry("1200x800")
        viz_window.configure(bg=self.colors['light'])

        # Header
        hdr = tk.Frame(viz_window, bg=self.colors['secondary'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=self.colors['accent'], width=5).pack(side='left', fill='y')
        tk.Label(hdr, text="\U0001f4c8  Data Visualizations Dashboard",
                 font=('Segoe UI', 13, 'bold'), fg='white',
                 bg=self.colors['secondary']).pack(side='left', padx=16, pady=12)
        tk.Label(hdr, text=f"{len(self.current_data):,} records",
                 font=('Segoe UI', 9), fg='#94a3b8',
                 bg=self.colors['secondary']).pack(side='right', padx=16, pady=12)

        # Chart selector buttons
        btn_bar = tk.Frame(viz_window, bg=self.colors['white'])
        btn_bar.pack(fill='x', padx=16, pady=(12, 0))

        chart_types = [
            ("Distribution",  self._viz_distribution),
            ("Correlation",   self._viz_correlation),
            ("Box Plot",      self._viz_boxplot),
            ("Pair Plot",     self._viz_pairplot),
        ]

        chart_frame = tk.Frame(viz_window, bg=self.colors['light'])
        chart_frame.pack(fill='both', expand=True, padx=16, pady=12)

        for i, (label, chart_fn) in enumerate(chart_types):
            btn = HoverButton(
                btn_bar, text=f"  {label}  ",
                command=lambda fn=chart_fn: fn(chart_frame, viz_window),
                font=('Segoe UI', 9, 'bold'),
                bg=self.colors['accent'] if i == 0 else self.colors['surface'],
                fg='white',
                hover_bg=self.colors['accent_hover'] if i == 0 else '#475569',
                padx=16, pady=6, cursor='hand2',
                relief='flat', bd=0,
                activeforeground='white',
                activebackground=self.colors['accent_hover'])
            btn.pack(side='left', padx=(0, 8))

        # Show first chart by default
        self._viz_distribution(chart_frame, viz_window)

    def _clear_chart_frame(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def _style_axes(self, ax, title):
        ax.set_facecolor('#f8fafc')
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color=self.colors['primary'], pad=12)
        ax.tick_params(colors=self.colors['muted'], labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color(self.colors['border'])

    def _viz_distribution(self, frame, window):
        """Distribution histogram chart."""
        self._clear_chart_frame(frame)
        if TARGET not in self.current_data.columns:
            self._show_chart_placeholder(frame, f"Target column '{TARGET}' not found")
            return

        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor('#f8fafc')
        self._style_axes(ax, 'Performance Score Distribution')

        self.current_data[TARGET].hist(bins=30, ax=ax,
                                       color=self.colors['accent'],
                                       edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Score', fontsize=10, color=self.colors['text_soft'])
        ax.set_ylabel('Frequency', fontsize=10, color=self.colors['text_soft'])

        # Add mean line
        mean_val = self.current_data[TARGET].mean()
        ax.axvline(mean_val, color=self.colors['danger'], linestyle='--',
                   linewidth=1.5, label=f'Mean: {mean_val:.1f}')
        ax.legend(fontsize=9, facecolor='white', edgecolor=self.colors['border'])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _viz_correlation(self, frame, window):
        """Correlation heatmap chart."""
        self._clear_chart_frame(frame)
        numeric = self.current_data.select_dtypes(include=[np.number])
        if numeric.empty:
            self._show_chart_placeholder(frame, "No numeric columns found")
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#f8fafc')

        corr = numeric.corr()
        sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0,
                    annot=True, fmt='.2f', annot_kws={'size': 8},
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=13,
                     fontweight='bold', color=self.colors['primary'], pad=12)
        ax.tick_params(colors=self.colors['muted'], labelsize=8)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _viz_boxplot(self, frame, window):
        """Box plot chart for numeric features."""
        self._clear_chart_frame(frame)
        numeric = self.current_data.select_dtypes(include=[np.number])
        if numeric.empty:
            self._show_chart_placeholder(frame, "No numeric columns found")
            return

        cols = numeric.columns[:8]  # limit to 8 for readability
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor('#f8fafc')
        self._style_axes(ax, 'Feature Distribution (Box Plot)')

        bp = ax.boxplot([numeric[c].dropna() for c in cols],
                        labels=cols, patch_artist=True,
                        medianprops={'color': self.colors['danger'], 'linewidth': 2})

        palette = [self.colors['accent'], self.colors['success'],
                   self.colors['warning'], '#8b5cf6',
                   '#ec4899', '#14b8a6', '#f97316', '#6366f1']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.7)

        ax.set_xlabel('Feature', fontsize=10, color=self.colors['text_soft'])
        ax.set_ylabel('Value', fontsize=10, color=self.colors['text_soft'])
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _viz_pairplot(self, frame, window):
        """Pair plot / scatter matrix for key features."""
        self._clear_chart_frame(frame)
        numeric = self.current_data.select_dtypes(include=[np.number])
        if numeric.empty:
            self._show_chart_placeholder(frame, "No numeric columns found")
            return

        cols = list(numeric.columns[:5])
        if TARGET in self.current_data.columns and TARGET not in cols:
            cols.append(TARGET)

        fig, axes = plt.subplots(len(cols), len(cols),
                                 figsize=(10, 10))
        fig.patch.set_facecolor('#f8fafc')

        for i, row_col in enumerate(cols):
            for j, col_col in enumerate(cols):
                ax = axes[i][j]
                ax.set_facecolor('#f8fafc')
                if i == j:
                    ax.hist(numeric[row_col].dropna(), bins=20,
                            color=self.colors['accent'], alpha=0.7,
                            edgecolor='white', linewidth=0.3)
                else:
                    ax.scatter(numeric[col_col], numeric[row_col],
                               s=8, alpha=0.4,
                               color=self.colors['accent'],
                               edgecolors='none')
                ax.tick_params(labelsize=6, colors=self.colors['muted'])
                if j == 0:
                    ax.set_ylabel(row_col[:10], fontsize=7,
                                  color=self.colors['text_soft'])
                if i == len(cols) - 1:
                    ax.set_xlabel(col_col[:10], fontsize=7,
                                  color=self.colors['text_soft'])
                for spine in ax.spines.values():
                    spine.set_color(self.colors['border'])

        fig.suptitle('Feature Pair Plot', fontsize=13, fontweight='bold',
                     color=self.colors['primary'], y=1.01)
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _show_chart_placeholder(self, parent, message):
        """Show a placeholder message in the chart area."""
        tk.Label(parent, text=message,
                 font=('Segoe UI', 12), fg=self.colors['muted'],
                 bg=self.colors['white']).pack(expand=True, pady=60)

    def _show_report_placeholder(self, message):
        """Show a placeholder message in the report area."""
        self._clear_report_host()
        self.report_display = tk.Text(self._report_host, font=('Segoe UI', 11),
                                      wrap=tk.WORD, bg='#f8fafc',
                                      fg=self.colors['muted'],
                                      relief='flat', padx=20, pady=20)
        self.report_display.pack(fill='both', expand=True)
        self.report_display.insert('1.0', message)
        self.report_display.config(state='disabled')

    def on_closing(self):
        """Handle closing."""
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                self.db_manager.close()
            except Exception:
                pass
        self.root.destroy()


# For backward compatibility
ProfessionalGUI = StudentPerformanceGUI
