"""
Self-reporting module for user studies to collect subjective cognitive state reports.

This module provides a mechanism for periodically prompting users to report their
fatigue and frustration levels during programming tasks.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import datetime
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SelfReportManager:
    """
    Manages the periodic self-reporting of cognitive states during user studies.
    
    This class handles the scheduling and display of prompt windows that ask
    users to rate their current fatigue and frustration levels.
    """
    
    def __init__(self, prompt_interval=1200, results_dir='results/user_study'):
        """
        Initialize the self-report manager.
        
        Parameters:
        -----------
        prompt_interval : int
            Time between prompts in seconds (default: 1200 = 20 minutes)
        results_dir : str
            Directory to save self-report results
        """
        self.prompt_interval = prompt_interval
        self.results_dir = results_dir
        self.reports = []
        self.active = False
        self.thread = None
        self.participant_id = None
        self.task_id = None
        self.start_time = None
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info("Self-report manager initialized")
    
    def start_session(self, participant_id, task_id):
        """
        Start a self-reporting session.
        
        Parameters:
        -----------
        participant_id : str
            Identifier for the participant
        task_id : str
            Identifier for the task (e.g., "debugging", "feature_implementation")
        
        Returns:
        --------
        bool
            True if session started successfully, False otherwise
        """
        if self.active:
            logger.warning("Attempt to start session when one is already active")
            return False
        
        self.participant_id = participant_id
        self.task_id = task_id
        self.start_time = datetime.datetime.now()
        self.reports = []
        self.active = True
        
        # Start the prompt thread
        self.thread = threading.Thread(target=self._prompt_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started self-report session for participant {participant_id}, task {task_id}")
        return True
    
    def stop_session(self):
        """
        Stop the self-reporting session and save the collected data.
        
        Returns:
        --------
        str
            Path to the saved report file
        """
        if not self.active:
            logger.warning("Attempt to stop session when none is active")
            return None
        
        self.active = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # Save the reports
        report_file = self._save_reports()
        
        logger.info(f"Stopped self-report session, saved to {report_file}")
        return report_file
    
    def collect_baseline(self):
        """
        Collect baseline cognitive state measurements at the start of a session.
        
        Returns:
        --------
        dict
            Baseline cognitive state data
        """
        # Display the self-report prompt immediately
        baseline_data = self._show_prompt(is_baseline=True)
        
        if baseline_data:
            baseline_data['timestamp'] = datetime.datetime.now().isoformat()
            baseline_data['type'] = 'baseline'
            self.reports.append(baseline_data)
            logger.info(f"Collected baseline cognitive state: fatigue={baseline_data['fatigue']}, frustration={baseline_data['frustration']}")
            return baseline_data
        
        logger.warning("Failed to collect baseline cognitive state")
        return None
    
    def _prompt_loop(self):
        """
        Main loop for periodically showing prompts.
        """
        # Wait for the first interval
        next_prompt_time = time.time() + self.prompt_interval
        
        while self.active:
            current_time = time.time()
            
            if current_time >= next_prompt_time:
                # Time to show a prompt
                report_data = self._show_prompt()
                
                if report_data:
                    report_data['timestamp'] = datetime.datetime.now().isoformat()
                    report_data['type'] = 'periodic'
                    report_data['minutes_elapsed'] = int((time.time() - self.start_time.timestamp()) / 60)
                    self.reports.append(report_data)
                    logger.info(f"Collected cognitive state report: fatigue={report_data['fatigue']}, frustration={report_data['frustration']}")
                
                # Schedule next prompt
                next_prompt_time = time.time() + self.prompt_interval
            
            # Sleep to avoid busy waiting
            time.sleep(1.0)
    
    def _show_prompt(self, is_baseline=False):
        """
        Display a prompt window for self-reporting cognitive states.
        
        Parameters:
        -----------
        is_baseline : bool
            Whether this is a baseline measurement
        
        Returns:
        --------
        dict
            Self-report data or None if user canceled
        """
        # Use threading event to wait for user response
        response_event = threading.Event()
        response_data = {'completed': False}
        
        # Define function to run GUI in main thread
        def show_gui():
            # Create a new top-level window
            root = tk.Tk()
            root.title("Cognitive State Self-Report")
            root.geometry("600x700")
            root.attributes('-topmost', True)
            
            # Configure style
            style = ttk.Style()
            style.configure('TFrame', background='#f0f0f0')
            style.configure('TLabel', background='#f0f0f0', font=('Arial', 11))
            style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
            style.configure('Scale.TLabel', font=('Arial', 10))
            
            # Create main frame
            main_frame = ttk.Frame(root, padding="20 20 20 20", style='TFrame')
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add header
            if is_baseline:
                header_text = "Baseline Cognitive State Assessment"
            else:
                header_text = "Cognitive State Self-Report"
            
            header = ttk.Label(main_frame, text=header_text, style='Header.TLabel')
            header.pack(pady=(0, 20))
            
            # Instructions
            instructions = ttk.Label(main_frame, 
                                    text="Please rate your current cognitive state based on the scales below.\n"
                                          "Your responses will help us understand how our system is performing.",
                                    wraplength=500, justify='center')
            instructions.pack(pady=(0, 30))
            
            # ---- Fatigue Scale (Stanford Sleepiness Scale) ----
            fatigue_frame = ttk.Frame(main_frame)
            fatigue_frame.pack(fill=tk.X, pady=(0, 30))
            
            fatigue_header = ttk.Label(fatigue_frame, text="Fatigue Level (Stanford Sleepiness Scale)", 
                                      style='Header.TLabel')
            fatigue_header.pack(anchor='w')
            
            # Define SSS options
            sss_options = [
                "1 - Feeling active, vital, alert, wide awake",
                "2 - Functioning at high level, but not at peak; able to concentrate",
                "3 - Relaxed, awake, not at full alertness, responsive",
                "4 - A little foggy, let down",
                "5 - Foggy, beginning to lose track, difficulty staying awake",
                "6 - Sleepy, prefer to lie down, woozy",
                "7 - Almost in reverie, sleep onset soon, lost struggle to remain awake"
            ]
            
            # Create variable and set default
            fatigue_var = tk.IntVar(value=1)
            
            # Create radio buttons for SSS
            for i, option in enumerate(sss_options, 1):
                ttk.Radiobutton(fatigue_frame, text=option, variable=fatigue_var, value=i).pack(
                    anchor='w', pady=3)
            
            # ---- Frustration Scale ----
            frustration_frame = ttk.Frame(main_frame)
            frustration_frame.pack(fill=tk.X, pady=(0, 30))
            
            frustration_header = ttk.Label(frustration_frame, text="Frustration Level", 
                                          style='Header.TLabel')
            frustration_header.pack(anchor='w')
            
            # Create frustration scale
            frustration_var = tk.IntVar(value=1)
            frustration_scale_frame = ttk.Frame(frustration_frame)
            frustration_scale_frame.pack(fill=tk.X, pady=(10, 0))
            
            # Labels for min/max
            ttk.Label(frustration_scale_frame, text="1 - Not at all frustrated", 
                     style='Scale.TLabel').pack(side=tk.LEFT)
            ttk.Label(frustration_scale_frame, text="7 - Extremely frustrated", 
                     style='Scale.TLabel').pack(side=tk.RIGHT)
            
            # Scale widget
            frustration_scale = ttk.Scale(frustration_frame, from_=1, to=7, orient=tk.HORIZONTAL,
                                         variable=frustration_var, length=500)
            frustration_scale.pack(fill=tk.X, pady=(5, 0))
            
            # Scale tick labels
            ticks_frame = ttk.Frame(frustration_frame)
            ticks_frame.pack(fill=tk.X)
            for i in range(1, 8):
                ttk.Label(ticks_frame, text=str(i), style='Scale.TLabel').place(
                    relx=(i-1)/6, y=0, anchor='n')
            
            # Optional notes
            notes_frame = ttk.Frame(main_frame)
            notes_frame.pack(fill=tk.X, pady=(0, 20))
            
            notes_label = ttk.Label(notes_frame, text="Optional: Any comments about your current state?")
            notes_label.pack(anchor='w')
            
            notes_text = tk.Text(notes_frame, height=4, width=50, wrap=tk.WORD, 
                               font=('Arial', 10))
            notes_text.pack(fill=tk.X, pady=(5, 0))
            
            # Submit button
            def on_submit():
                response_data['fatigue'] = fatigue_var.get()
                response_data['frustration'] = frustration_var.get()
                response_data['notes'] = notes_text.get("1.0", tk.END).strip()
                response_data['completed'] = True
                response_event.set()
                root.destroy()
            
            submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
            submit_button.pack(pady=20)
            
            # Handle window close
            def on_closing():
                response_event.set()
                root.destroy()
            
            root.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Start the main loop
            root.mainloop()
        
        # Show GUI in main thread
        threading.Thread(target=show_gui).start()
        
        # Wait for response with timeout
        response_event.wait(timeout=300)  # 5 minutes timeout
        
        if response_data['completed']:
            return response_data
        else:
            logger.warning("User did not complete self-report prompt")
            return None
    
    def _save_reports(self):
        """
        Save collected self-reports to a JSON file.
        
        Returns:
        --------
        str
            Path to the saved file
        """
        if not self.reports:
            logger.warning("No reports to save")
            return None
        
        # Create filename based on participant, task, and timestamp
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"{self.participant_id}_{self.task_id}_{timestamp}_self_reports.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Create data structure
        data = {
            'participant_id': self.participant_id,
            'task_id': self.task_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'reports': self.reports
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath


class StanfordSleepinessScale:
    """
    Stanford Sleepiness Scale (SSS) definitions for reference.
    """
    
    LEVELS = {
        1: "Feeling active, vital, alert, or wide awake",
        2: "Functioning at high levels, but not at peak; able to concentrate",
        3: "Awake, but relaxed; responsive but not fully alert",
        4: "Somewhat foggy, let down",
        5: "Foggy; losing interest in remaining awake; slowed down",
        6: "Sleepy, woozy, fighting sleep; prefer to lie down",
        7: "No longer fighting sleep, sleep onset soon; having dream-like thoughts"
    }
    
    @classmethod
    def get_description(cls, level):
        """Get the description for a given SSS level."""
        return cls.LEVELS.get(level, "Unknown level")
    
    @classmethod
    def is_fatigued(cls, level):
        """
        Determine if a given SSS level indicates fatigue.
        
        Parameters:
        -----------
        level : int
            SSS level (1-7)
            
        Returns:
        --------
        bool
            True if the level indicates fatigue (≥ 4), False otherwise
        """
        return level >= 4
