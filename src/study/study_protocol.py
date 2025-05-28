"""
Study protocol implementation for cognitive state detection evaluation.

This module implements the specific study protocol described in the thesis:
- Warm-up/Baseline Task (15 mins)
- Fatigue-Inducing Task (75 mins)
- Frustration-Inducing Task (45 mins)

It coordinates task timing, questionnaires, and adaptive interventions.
"""

import logging
import time
import json
import os
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Enum representing different task types in the study protocol."""
    WARM_UP = "warm_up"
    FATIGUE = "fatigue"
    FRUSTRATION = "frustration"

class StudyProtocolManager:
    """
    Manages the study protocol for cognitive state detection evaluation.
    
    This class coordinates the three study tasks, questionnaires,
    and adaptive interventions as described in the study protocol.
    """
    
    def __init__(self, 
                 study_dir='results/user_study',
                 warm_up_duration=15*60,    # 15 minutes in seconds
                 fatigue_duration=75*60,    # 75 minutes in seconds  
                 frustration_duration=45*60 # 45 minutes in seconds
                ):
        """
        Initialize the study protocol manager.
        
        Parameters:
        -----------
        study_dir : str
            Directory to save study results
        warm_up_duration : int
            Duration of warm-up task in seconds
        fatigue_duration : int
            Duration of fatigue-inducing task in seconds
        frustration_duration : int
            Duration of frustration-inducing task in seconds
        """
        self.study_dir = study_dir
        self.warm_up_duration = warm_up_duration
        self.fatigue_duration = fatigue_duration
        self.frustration_duration = frustration_duration
        
        self.participant_id = None
        self.participant_data = {}
        self.current_task = None
        self.task_start_time = None
        self.questionnaire_responses = {
            'pre_study': None,
            'in_task': [],
            'post_study': None
        }
        
        # Create study directory if it doesn't exist
        os.makedirs(study_dir, exist_ok=True)
        
        logger.info("Study Protocol Manager initialized")
    
    def start_study(self, participant_id):
        """
        Start the study for a participant.
        
        Parameters:
        -----------
        participant_id : str
            Identifier for the participant
            
        Returns:
        --------
        bool
            True if study started successfully, False otherwise
        """
        self.participant_id = participant_id
        self.participant_data = {
            'participant_id': participant_id,
            'study_start_time': datetime.datetime.now().isoformat(),
            'tasks': [],
            'questionnaires': {
                'pre_study': None,
                'in_task': [],
                'post_study': None
            }
        }
        
        logger.info(f"Starting study for participant {participant_id}")
        
        # Show pre-study questionnaire
        self.show_pre_study_questionnaire()
        
        return True
    
    def start_task(self, task_type):
        """
        Start a specific task for the current participant.
        
        Parameters:
        -----------
        task_type : TaskType
            Type of task to start
            
        Returns:
        --------
        bool
            True if task started successfully, False otherwise
        """
        if not self.participant_id:
            logger.error("Cannot start task: No participant ID set")
            return False
        
        self.current_task = task_type
        self.task_start_time = time.time()
        
        task_info = {
            'task_type': task_type.value,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'completed': False
        }
        
        self.participant_data['tasks'].append(task_info)
        
        logger.info(f"Starting {task_type.value} task for participant {self.participant_id}")
        
        # Show task instructions
        self.show_task_instructions(task_type)
        
        return True
    
    def end_task(self, completed=True):
        """
        End the current task.
        
        Parameters:
        -----------
        completed : bool
            Whether the task was completed successfully
            
        Returns:
        --------
        bool
            True if task ended successfully, False otherwise
        """
        if not self.current_task:
            logger.error("Cannot end task: No task currently active")
            return False
        
        # Find the current task in the participant data
        for task in self.participant_data['tasks']:
            if task['task_type'] == self.current_task.value and task['end_time'] is None:
                task['end_time'] = datetime.datetime.now().isoformat()
                task['completed'] = completed
                task['duration_seconds'] = time.time() - self.task_start_time
                break
        
        logger.info(f"Ending {self.current_task.value} task for participant {self.participant_id}")
        
        # If this was the final task, show post-study questionnaire
        if self.current_task == TaskType.FRUSTRATION:
            self.show_post_study_questionnaire()
        
        self.current_task = None
        self.task_start_time = None
        
        return True
    
    def show_pre_study_questionnaire(self):
        """
        Show the pre-study questionnaire.
        
        Returns:
        --------
        dict
            Dictionary containing questionnaire responses
        """
        logger.info("Showing pre-study questionnaire")
        
        # In a real implementation, this would show a GUI questionnaire
        # For simulation, we create a dummy response
        responses = {
            'demographics': {
                'age': 26,
                'gender': 'Female',
                'programming_experience_years': 4,
                'education_level': 'Graduate'
            },
            'baseline': {
                'fatigue_level': 2,  # 1-7 Likert scale
                'frustration_level': 1,  # 1-7 Likert scale
                'stress_level': 2  # 1-7 Likert scale
            }
        }
        
        self.questionnaire_responses['pre_study'] = responses
        self.participant_data['questionnaires']['pre_study'] = responses
        
        return responses
    
    def show_in_task_questionnaire(self):
        """
        Show an in-task questionnaire.
        
        Returns:
        --------
        dict
            Dictionary containing questionnaire responses
        """
        if not self.current_task:
            logger.error("Cannot show in-task questionnaire: No task currently active")
            return None
        
        logger.info(f"Showing in-task questionnaire for {self.current_task.value} task")
        
        # In a real implementation, this would show a GUI questionnaire
        # For simulation, we create a dummy response
        responses = {
            'timestamp': datetime.datetime.now().isoformat(),
            'task_type': self.current_task.value,
            'fatigue_level': 4 if self.current_task == TaskType.FATIGUE else 3,  # 1-7 Likert scale
            'frustration_level': 5 if self.current_task == TaskType.FRUSTRATION else 2,  # 1-7 Likert scale
            'notes': ''
        }
        
        self.questionnaire_responses['in_task'].append(responses)
        self.participant_data['questionnaires']['in_task'].append(responses)
        
        return responses
    
    def show_post_study_questionnaire(self):
        """
        Show the post-study questionnaire.
        
        Returns:
        --------
        dict
            Dictionary containing questionnaire responses
        """
        logger.info("Showing post-study questionnaire")
        
        # In a real implementation, this would show a GUI questionnaire
        # For simulation, we create a dummy response
        responses = {
            'nasa_tlx': {
                'mental_demand': 65,
                'physical_demand': 30,
                'temporal_demand': 70,
                'performance': 75,
                'effort': 80,
                'frustration': 60
            },
            'system_usability': {
                'overall_score': 72,  # SUS score (0-100)
                'comfort': 4,  # 1-5 scale
                'privacy_concerns': 3,  # 1-5 scale
                'perceived_accuracy': 4  # 1-5 scale
            },
            'open_ended': {
                'feedback': 'The break recommendations were helpful during long coding sessions.',
                'improvement_suggestions': 'The system could be more proactive in suggesting solutions when frustration is detected.'
            }
        }
        
        self.questionnaire_responses['post_study'] = responses
        self.participant_data['questionnaires']['post_study'] = responses
        
        # Save all participant data
        self.save_participant_data()
        
        return responses
    
    def show_task_instructions(self, task_type):
        """
        Show instructions for a specific task.
        
        Parameters:
        -----------
        task_type : TaskType
            Type of task to show instructions for
        """
        instructions = {
            TaskType.WARM_UP: """
                Warm-up/Baseline Task (15 minutes):
                - Simple code refactoring task in Python
                - Familiarize yourself with the setup
                - This helps establish baseline eye metrics
            """,
            TaskType.FATIGUE: """
                Fatigue-Inducing Task (75 minutes):
                - Code comprehension and debugging task
                - Moderately complex Java module (approx. 2000 LOC)
                - Understand its functionality and fix three seeded bugs
                - Bugs are of increasing difficulty
                - You may receive break recommendations during this task
            """,
            TaskType.FRUSTRATION: """
                Frustration-Inducing Task (45 minutes):
                - Challenging algorithmic problem in Python
                - Solve a dynamic programming problem
                - Note: There is a deliberately misleading hint in the problem statement
            """
        }
        
        logger.info(f"Showing instructions for {task_type.value} task")
        logger.info(instructions[task_type])
        
        # In a real implementation, this would show a GUI with instructions
        # For simulation, we just log the instructions
        
    def save_participant_data(self):
        """
        Save all participant data to a JSON file.
        
        Returns:
        --------
        str
            Path to the saved file
        """
        if not self.participant_id:
            logger.error("Cannot save participant data: No participant ID set")
            return None
        
        # Add study end time
        self.participant_data['study_end_time'] = datetime.datetime.now().isoformat()
        
        # Create filename
        filename = f"{self.participant_id}_study_data.json"
        filepath = os.path.join(self.study_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.participant_data, f, indent=2)
        
        logger.info(f"Saved participant data to {filepath}")
        
        return filepath
    
    def get_current_task_info(self):
        """
        Get information about the current task.
        
        Returns:
        --------
        dict
            Dictionary containing current task information
        """
        if not self.current_task or not self.task_start_time:
            return None
        
        elapsed_time = time.time() - self.task_start_time
        remaining_time = None
        
        if self.current_task == TaskType.WARM_UP:
            remaining_time = max(0, self.warm_up_duration - elapsed_time)
        elif self.current_task == TaskType.FATIGUE:
            remaining_time = max(0, self.fatigue_duration - elapsed_time)
        elif self.current_task == TaskType.FRUSTRATION:
            remaining_time = max(0, self.frustration_duration - elapsed_time)
        
        return {
            'task_type': self.current_task.value,
            'elapsed_time_seconds': elapsed_time,
            'remaining_time_seconds': remaining_time,
            'participant_id': self.participant_id
        }
