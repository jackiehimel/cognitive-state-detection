"""
Data collection module for user studies to correlate system predictions with self-reports.

This module handles the collection and storage of system predictions and user self-reports
during user studies for later analysis.
"""

import os
import json
import datetime
import pandas as pd
import numpy as np
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudyDataCollector:
    """
    Collects and correlates system predictions with user self-reports during studies.
    
    This class stores system predictions continuously and correlates them with
    periodic user self-reports for later analysis.
    """
    
    def __init__(self, window_size=5, results_dir='results/user_study'):
        """
        Initialize the study data collector.
        
        Parameters:
        -----------
        window_size : int
            Size of each prediction window in seconds
        results_dir : str
            Directory to save study results
        """
        self.window_size = window_size
        self.results_dir = results_dir
        self.predictions = []
        self.features = []
        self.self_reports = []
        self.participant_id = None
        self.task_id = None
        self.start_time = None
        self.recording = False
        
        # Buffer for recent predictions (for real-time analysis)
        self.recent_predictions = deque(maxlen=int(1200/window_size))  # Last 20 minutes
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info("Study data collector initialized")
    
    def start_session(self, participant_id, task_id):
        """
        Start a data collection session.
        
        Parameters:
        -----------
        participant_id : str
            Identifier for the participant
        task_id : str
            Identifier for the task
            
        Returns:
        --------
        bool
            True if session started successfully, False otherwise
        """
        if self.recording:
            logger.warning("Attempt to start session when one is already active")
            return False
        
        self.participant_id = participant_id
        self.task_id = task_id
        self.start_time = datetime.datetime.now()
        self.predictions = []
        self.features = []
        self.self_reports = []
        self.recent_predictions.clear()
        self.recording = True
        
        logger.info(f"Started data collection for participant {participant_id}, task {task_id}")
        return True
    
    def stop_session(self):
        """
        Stop the data collection session and save the collected data.
        
        Returns:
        --------
        tuple
            Paths to the saved prediction and feature files
        """
        if not self.recording:
            logger.warning("Attempt to stop session when none is active")
            return None, None
        
        self.recording = False
        
        # Save the data
        prediction_file = self._save_predictions()
        feature_file = self._save_features()
        
        logger.info(f"Stopped data collection, saved predictions to {prediction_file} and features to {feature_file}")
        return prediction_file, feature_file
    
    def record_prediction(self, prediction, timestamp=None):
        """
        Record a system prediction.
        
        Parameters:
        -----------
        prediction : dict
            Dictionary containing system prediction
            {
                'fatigue_probability': float,
                'frustration_probability': float,
                'detected_state': str
            }
        timestamp : datetime, optional
            Timestamp for the prediction, defaults to current time
            
        Returns:
        --------
        bool
            True if prediction was recorded successfully, False otherwise
        """
        if not self.recording:
            logger.warning("Attempt to record prediction when not recording")
            return False
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Create prediction record
        prediction_record = {
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': (timestamp - self.start_time).total_seconds(),
            'fatigue_probability': prediction.get('fatigue_probability', 0.0),
            'frustration_probability': prediction.get('frustration_probability', 0.0),
            'detected_state': prediction.get('detected_state', 'neutral')
        }
        
        # Add to predictions list and recent buffer
        self.predictions.append(prediction_record)
        self.recent_predictions.append(prediction_record)
        
        return True
    
    def record_features(self, features, timestamp=None):
        """
        Record extracted eye features.
        
        Parameters:
        -----------
        features : dict
            Dictionary containing extracted eye features
            {
                'perclos': float,
                'blink_rate': float,
                'blink_duration': float,
                'pupil_diameter': float,
                'pupil_variance': float,
                'gaze_dispersion': float,
                ...
            }
        timestamp : datetime, optional
            Timestamp for the features, defaults to current time
            
        Returns:
        --------
        bool
            True if features were recorded successfully, False otherwise
        """
        if not self.recording:
            logger.warning("Attempt to record features when not recording")
            return False
        
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        # Create feature record
        feature_record = {
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': (timestamp - self.start_time).total_seconds()
        }
        
        # Add all features to the record
        feature_record.update(features)
        
        # Add to features list
        self.features.append(feature_record)
        
        return True
    
    def record_self_report(self, self_report):
        """
        Record a user self-report.
        
        Parameters:
        -----------
        self_report : dict
            Dictionary containing self-report data
            {
                'timestamp': str (ISO format),
                'fatigue': int (1-7),
                'frustration': int (1-7),
                'notes': str,
                'type': str ('baseline' or 'periodic'),
                'minutes_elapsed': int (optional)
            }
            
        Returns:
        --------
        bool
            True if self-report was recorded successfully, False otherwise
        """
        if not self.recording:
            logger.warning("Attempt to record self-report when not recording")
            return False
        
        # Make a copy to avoid modifying the original
        report = dict(self_report)
        
        # Add participant and task info
        report['participant_id'] = self.participant_id
        report['task_id'] = self.task_id
        
        # Calculate fatigue and frustration binary flags
        report['is_fatigued'] = 1 if report.get('fatigue', 0) >= 4 else 0
        report['is_frustrated'] = 1 if report.get('frustration', 0) >= 5 else 0
        
        # Add to self-reports list
        self.self_reports.append(report)
        
        # Get system predictions for the same time period
        self._correlate_predictions_with_report(report)
        
        logger.info(f"Recorded self-report: fatigue={report['fatigue']}, frustration={report['frustration']}")
        return True
    
    def _correlate_predictions_with_report(self, report):
        """
        Correlate system predictions with a user self-report.
        
        Parameters:
        -----------
        report : dict
            Dictionary containing self-report data
        """
        # If this is a baseline report, don't try to correlate
        if report.get('type') == 'baseline':
            return
        
        # Parse the timestamp
        try:
            report_time = datetime.datetime.fromisoformat(report['timestamp'])
        except (ValueError, KeyError):
            logger.error("Invalid timestamp in self-report")
            return
        
        # Get system predictions for the 20-minute window before the report
        window_start = report_time - datetime.timedelta(minutes=20)
        
        # Filter predictions in the time window
        window_predictions = []
        for pred in self.predictions:
            try:
                pred_time = datetime.datetime.fromisoformat(pred['timestamp'])
                if window_start <= pred_time <= report_time:
                    window_predictions.append(pred)
            except (ValueError, KeyError):
                continue
        
        if not window_predictions:
            logger.warning("No predictions found in 20-minute window before self-report")
            return
        
        # Calculate system's aggregate prediction for the window
        fatigue_probs = [p.get('fatigue_probability', 0) for p in window_predictions]
        frustration_probs = [p.get('frustration_probability', 0) for p in window_predictions]
        
        avg_fatigue_prob = np.mean(fatigue_probs) if fatigue_probs else 0
        avg_frustration_prob = np.mean(frustration_probs) if frustration_probs else 0
        
        # Calculate binary system prediction (if >60% of windows detected the state)
        system_fatigue = 1 if (np.array(fatigue_probs) > 0.5).mean() > 0.6 else 0
        system_frustration = 1 if (np.array(frustration_probs) > 0.5).mean() > 0.6 else 0
        
        # Count predictions by state
        state_counts = {}
        for p in window_predictions:
            state = p.get('detected_state', 'neutral')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Add correlation data to the report
        report['window_predictions_count'] = len(window_predictions)
        report['system_avg_fatigue_prob'] = avg_fatigue_prob
        report['system_avg_frustration_prob'] = avg_frustration_prob
        report['system_fatigue_detection'] = system_fatigue
        report['system_frustration_detection'] = system_frustration
        report['system_state_counts'] = state_counts
        
        # Add agreement metrics
        report['fatigue_agreement'] = 1 if (report['is_fatigued'] == system_fatigue) else 0
        report['frustration_agreement'] = 1 if (report['is_frustrated'] == system_frustration) else 0
        
        logger.info(f"Correlated self-report with {len(window_predictions)} system predictions")
    
    def get_session_summary(self):
        """
        Get a summary of the current session.
        
        Returns:
        --------
        dict
            Summary statistics for the session
        """
        if not self.predictions or not self.self_reports:
            return {"status": "No data collected yet"}
        
        # Calculate basic statistics
        total_duration = 0
        if self.recording and self.start_time:
            total_duration = (datetime.datetime.now() - self.start_time).total_seconds() / 60
        elif self.predictions:
            try:
                first_time = datetime.datetime.fromisoformat(self.predictions[0]['timestamp'])
                last_time = datetime.datetime.fromisoformat(self.predictions[-1]['timestamp'])
                total_duration = (last_time - first_time).total_seconds() / 60
            except (ValueError, KeyError, IndexError):
                pass
        
        # Count predictions by state
        state_counts = {}
        for p in self.predictions:
            state = p.get('detected_state', 'neutral')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Summarize self-reports
        fatigue_levels = [r.get('fatigue', 0) for r in self.self_reports]
        frustration_levels = [r.get('frustration', 0) for r in self.self_reports]
        
        avg_fatigue = np.mean(fatigue_levels) if fatigue_levels else 0
        avg_frustration = np.mean(frustration_levels) if frustration_levels else 0
        
        # Calculate agreement rates
        fatigue_agreements = [r.get('fatigue_agreement', None) for r in self.self_reports 
                             if r.get('fatigue_agreement') is not None]
        frustration_agreements = [r.get('frustration_agreement', None) for r in self.self_reports 
                                 if r.get('frustration_agreement') is not None]
        
        fatigue_agreement_rate = np.mean(fatigue_agreements) if fatigue_agreements else None
        frustration_agreement_rate = np.mean(frustration_agreements) if frustration_agreements else None
        
        return {
            "participant_id": self.participant_id,
            "task_id": self.task_id,
            "total_duration_minutes": total_duration,
            "total_predictions": len(self.predictions),
            "total_self_reports": len(self.self_reports),
            "state_distribution": state_counts,
            "avg_reported_fatigue": avg_fatigue,
            "avg_reported_frustration": avg_frustration,
            "fatigue_agreement_rate": fatigue_agreement_rate,
            "frustration_agreement_rate": frustration_agreement_rate
        }
    
    def _save_predictions(self):
        """
        Save collected predictions to a JSON file.
        
        Returns:
        --------
        str
            Path to the saved file
        """
        if not self.predictions:
            logger.warning("No predictions to save")
            return None
        
        # Create filename
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"{self.participant_id}_{self.task_id}_{timestamp}_predictions.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Create data structure
        data = {
            'participant_id': self.participant_id,
            'task_id': self.task_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'window_size_seconds': self.window_size,
            'predictions': self.predictions,
            'self_reports': self.self_reports
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _save_features(self):
        """
        Save collected features to a JSON file.
        
        Returns:
        --------
        str
            Path to the saved file
        """
        if not self.features:
            logger.warning("No features to save")
            return None
        
        # Create filename
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"{self.participant_id}_{self.task_id}_{timestamp}_features.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Create data structure
        data = {
            'participant_id': self.participant_id,
            'task_id': self.task_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'features': self.features
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def export_to_csv(self, output_dir=None):
        """
        Export collected data to CSV files for analysis.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save CSV files, defaults to results_dir
            
        Returns:
        --------
        tuple
            Paths to the saved CSV files (predictions, features, self_reports)
        """
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create base filename
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S') if self.start_time else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{self.participant_id}_{self.task_id}_{timestamp}"
        
        # Export predictions
        predictions_file = None
        if self.predictions:
            predictions_df = pd.DataFrame(self.predictions)
            predictions_file = os.path.join(output_dir, f"{base_filename}_predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)
        
        # Export features
        features_file = None
        if self.features:
            features_df = pd.DataFrame(self.features)
            features_file = os.path.join(output_dir, f"{base_filename}_features.csv")
            features_df.to_csv(features_file, index=False)
        
        # Export self-reports
        self_reports_file = None
        if self.self_reports:
            self_reports_df = pd.DataFrame(self.self_reports)
            self_reports_file = os.path.join(output_dir, f"{base_filename}_self_reports.csv")
            self_reports_df.to_csv(self_reports_file, index=False)
        
        logger.info(f"Exported data to CSV files in {output_dir}")
        return predictions_file, features_file, self_reports_file
