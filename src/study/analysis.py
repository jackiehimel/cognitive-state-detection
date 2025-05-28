"""
Statistical analysis module for user studies.

This module provides tools for analyzing the correlation between eye metrics
and subjective cognitive states as reported by users.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudyAnalysis:
    """
    Analyzes user study data to calculate correlations between eye metrics
    and subjective cognitive states.
    """
    
    def __init__(self, results_dir='results/user_study'):
        """
        Initialize the study analysis.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing study results
        """
        self.results_dir = results_dir
        self.feature_data = None
        self.prediction_data = None
        self.self_report_data = None
        self.correlation_results = None
        
        logger.info("Study analysis initialized")
    
    def load_participant_data(self, participant_id, task_id=None, timestamp=None):
        """
        Load data for a specific participant.
        
        Parameters:
        -----------
        participant_id : str
            Identifier for the participant
        task_id : str, optional
            Identifier for the task
        timestamp : str, optional
            Timestamp for the specific session
            
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        # Determine file pattern based on parameters
        pattern = f"{participant_id}"
        if task_id:
            pattern += f"_{task_id}"
        if timestamp:
            pattern += f"_{timestamp}"
        
        # Find matching files
        feature_files = []
        prediction_files = []
        self_report_files = []
        
        for file in os.listdir(self.results_dir):
            if not file.startswith(pattern):
                continue
                
            if file.endswith("_features.json"):
                feature_files.append(file)
            elif file.endswith("_predictions.json"):
                prediction_files.append(file)
            elif file.endswith("_self_reports.json"):
                self_report_files.append(file)
        
        if not feature_files or not prediction_files or not self_report_files:
            logger.warning(f"Could not find all required files for {pattern}")
            return False
        
        # Sort by timestamp (newest first)
        feature_files.sort(reverse=True)
        prediction_files.sort(reverse=True)
        self_report_files.sort(reverse=True)
        
        # Load the files
        try:
            with open(os.path.join(self.results_dir, feature_files[0]), 'r') as f:
                self.feature_data = json.load(f)
            
            with open(os.path.join(self.results_dir, prediction_files[0]), 'r') as f:
                self.prediction_data = json.load(f)
            
            with open(os.path.join(self.results_dir, self_report_files[0]), 'r') as f:
                self.self_report_data = json.load(f)
                
            logger.info(f"Loaded data for participant {participant_id}")
            return True
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def load_all_participants(self):
        """
        Load data for all participants.
        
        Returns:
        --------
        tuple
            (features_df, predictions_df, self_reports_df)
        """
        # Lists to store data from all participants
        all_features = []
        all_predictions = []
        all_self_reports = []
        
        # Find all unique participant IDs
        participant_ids = set()
        for file in os.listdir(self.results_dir):
            if file.endswith("_features.json") or file.endswith("_predictions.json") or file.endswith("_self_reports.json"):
                parts = file.split('_')
                if len(parts) >= 3:
                    participant_ids.add(parts[0])
        
        # Load data for each participant
        for participant_id in participant_ids:
            success = self.load_participant_data(participant_id)
            if success:
                # Add participant data to lists
                if self.feature_data and 'features' in self.feature_data:
                    for feature in self.feature_data['features']:
                        feature['participant_id'] = participant_id
                        all_features.append(feature)
                
                if self.prediction_data and 'predictions' in self.prediction_data:
                    for prediction in self.prediction_data['predictions']:
                        prediction['participant_id'] = participant_id
                        all_predictions.append(prediction)
                
                if self.self_report_data and 'reports' in self.self_report_data:
                    for report in self.self_report_data['reports']:
                        report['participant_id'] = participant_id
                        all_self_reports.append(report)
        
        # Convert to dataframes
        features_df = pd.DataFrame(all_features) if all_features else None
        predictions_df = pd.DataFrame(all_predictions) if all_predictions else None
        self_reports_df = pd.DataFrame(all_self_reports) if all_self_reports else None
        
        logger.info(f"Loaded data for {len(participant_ids)} participants")
        return features_df, predictions_df, self_reports_df
    
    def calculate_eye_metric_correlations(self, output_dir=None):
        """
        Calculate Spearman's rank correlation between eye metrics and self-reported cognitive states.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save correlation results, defaults to results_dir
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing correlation results
        """
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all participant data
        features_df, _, self_reports_df = self.load_all_participants()
        
        if features_df is None or self_reports_df is None:
            logger.error("Cannot calculate correlations: Missing data")
            return None
        
        # Prepare self-reports dataframe
        self_reports_df['timestamp'] = pd.to_datetime(self_reports_df['timestamp'])
        
        # Define eye metrics to analyze
        eye_metrics = [
            'perclos',
            'blink_rate',
            'blink_duration',
            'pupil_diameter',
            'pupil_variance',
            'gaze_scanpath_area',
            'avg_fixation_duration',
            'saccade_amplitude'
        ]
        
        # Filter available metrics
        available_metrics = [m for m in eye_metrics if m in features_df.columns]
        
        if not available_metrics:
            logger.error("No eye metrics found in feature data")
            return None
        
        # Initialize results dataframe
        results = []
        
        # For each participant
        for participant_id in features_df['participant_id'].unique():
            participant_features = features_df[features_df['participant_id'] == participant_id].copy()
            participant_reports = self_reports_df[self_reports_df['participant_id'] == participant_id].copy()
            
            if participant_features.empty or participant_reports.empty:
                continue
            
            # Convert timestamps to datetime
            participant_features['timestamp'] = pd.to_datetime(participant_features['timestamp'])
            
            # For each self-report
            for _, report in participant_reports.iterrows():
                # Skip baseline reports
                if report.get('type') == 'baseline':
                    continue
                
                report_time = pd.to_datetime(report['timestamp'])
                
                # Get features in the 20-minute window before the report
                window_start = report_time - pd.Timedelta(minutes=20)
                window_features = participant_features[
                    (participant_features['timestamp'] >= window_start) & 
                    (participant_features['timestamp'] <= report_time)
                ]
                
                if window_features.empty:
                    continue
                
                # Calculate average metrics for the window
                avg_metrics = {}
                for metric in available_metrics:
                    if metric in window_features.columns:
                        avg_metrics[metric] = window_features[metric].mean()
                
                # Add to results
                result = {
                    'participant_id': participant_id,
                    'fatigue_score': report.get('fatigue', 0),
                    'frustration_score': report.get('frustration', 0),
                    'timestamp': report_time
                }
                result.update(avg_metrics)
                results.append(result)
        
        if not results:
            logger.error("No matching data found for correlation analysis")
            return None
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        # Calculate correlations for each metric
        correlation_results = []
        
        for metric in available_metrics:
            if metric not in results_df.columns:
                continue
                
            # Correlation with fatigue
            fatigue_corr, fatigue_p = stats.spearmanr(
                results_df[metric], 
                results_df['fatigue_score'],
                nan_policy='omit'
            )
            
            # Correlation with frustration
            frustration_corr, frustration_p = stats.spearmanr(
                results_df[metric], 
                results_df['frustration_score'],
                nan_policy='omit'
            )
            
            correlation_results.append({
                'eye_metric': metric,
                'correlation_with_fatigue': fatigue_corr,
                'fatigue_p_value': fatigue_p,
                'correlation_with_frustration': frustration_corr,
                'frustration_p_value': frustration_p
            })
        
        # Convert to dataframe
        correlation_df = pd.DataFrame(correlation_results)
        self.correlation_results = correlation_df
        
        # Save results
        csv_path = os.path.join(output_dir, 'eye_metric_correlations.csv')
        correlation_df.to_csv(csv_path, index=False)
        
        # Create correlation table visualization
        self._create_correlation_table(correlation_df, output_dir)
        
        logger.info(f"Calculated correlations between eye metrics and cognitive states, saved to {csv_path}")
        return correlation_df
    
    def _create_correlation_table(self, correlation_df, output_dir):
        """
        Create a visualization of the correlation table (Table X.3).
        
        Parameters:
        -----------
        correlation_df : pd.DataFrame
            DataFrame containing correlation results
        output_dir : str
            Directory to save the visualization
        """
        # Prepare data for table
        table_data = []
        for _, row in correlation_df.iterrows():
            fatigue_sig = '***' if row['fatigue_p_value'] < 0.001 else ('**' if row['fatigue_p_value'] < 0.01 else ('*' if row['fatigue_p_value'] < 0.05 else ''))
            frustration_sig = '***' if row['frustration_p_value'] < 0.001 else ('**' if row['frustration_p_value'] < 0.01 else ('*' if row['frustration_p_value'] < 0.05 else ''))
            
            table_data.append([
                row['eye_metric'],
                f"{row['correlation_with_fatigue']:.2f}{fatigue_sig}",
                f"{row['fatigue_p_value']:.3f}",
                f"{row['correlation_with_frustration']:.2f}{frustration_sig}",
                f"{row['frustration_p_value']:.3f}"
            ])
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, len(correlation_df) * 0.5 + 2))
        
        # Hide axes
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=['Eye Metric', 'Correlation with Fatigue (ρ)', 'p-value', 'Correlation with Frustration (ρ)', 'p-value'],
            loc='center',
            cellLoc='center'
        )
        
        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title('Table X.3: Mean Spearman\'s ρ correlations between eye metrics and subjective cognitive state scores', 
                 fontsize=12, pad=20)
        
        # Add footnote
        plt.figtext(0.1, 0.01, "* p < 0.05, ** p < 0.01, *** p < 0.001", ha="left", fontsize=10)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eye_metric_correlations_table.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_detection_performance(self, output_dir=None):
        """
        Calculate system performance in detecting self-reported cognitive states.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save performance results, defaults to results_dir
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all participant data
        _, predictions_df, self_reports_df = self.load_all_participants()
        
        if predictions_df is None or self_reports_df is None:
            logger.error("Cannot calculate performance: Missing data")
            return None
        
        # Filter self-reports with system predictions
        valid_reports = self_reports_df[
            self_reports_df['system_fatigue_detection'].notna() & 
            self_reports_df['system_frustration_detection'].notna()
        ]
        
        if valid_reports.empty:
            logger.error("No valid self-reports with system predictions found")
            return None
        
        # Calculate performance metrics
        # For fatigue
        fatigue_true_pos = valid_reports[(valid_reports['is_fatigued'] == 1) & 
                                        (valid_reports['system_fatigue_detection'] == 1)].shape[0]
        fatigue_true_neg = valid_reports[(valid_reports['is_fatigued'] == 0) & 
                                        (valid_reports['system_fatigue_detection'] == 0)].shape[0]
        fatigue_false_pos = valid_reports[(valid_reports['is_fatigued'] == 0) & 
                                         (valid_reports['system_fatigue_detection'] == 1)].shape[0]
        fatigue_false_neg = valid_reports[(valid_reports['is_fatigued'] == 1) & 
                                         (valid_reports['system_fatigue_detection'] == 0)].shape[0]
        
        # For frustration
        frustration_true_pos = valid_reports[(valid_reports['is_frustrated'] == 1) & 
                                           (valid_reports['system_frustration_detection'] == 1)].shape[0]
        frustration_true_neg = valid_reports[(valid_reports['is_frustrated'] == 0) & 
                                           (valid_reports['system_frustration_detection'] == 0)].shape[0]
        frustration_false_pos = valid_reports[(valid_reports['is_frustrated'] == 0) & 
                                            (valid_reports['system_frustration_detection'] == 1)].shape[0]
        frustration_false_neg = valid_reports[(valid_reports['is_frustrated'] == 1) & 
                                            (valid_reports['system_frustration_detection'] == 0)].shape[0]
        
        # Calculate metrics
        fatigue_accuracy = (fatigue_true_pos + fatigue_true_neg) / valid_reports.shape[0]
        fatigue_precision = fatigue_true_pos / (fatigue_true_pos + fatigue_false_pos) if (fatigue_true_pos + fatigue_false_pos) > 0 else 0
        fatigue_recall = fatigue_true_pos / (fatigue_true_pos + fatigue_false_neg) if (fatigue_true_pos + fatigue_false_neg) > 0 else 0
        fatigue_f1 = 2 * (fatigue_precision * fatigue_recall) / (fatigue_precision + fatigue_recall) if (fatigue_precision + fatigue_recall) > 0 else 0
        
        frustration_accuracy = (frustration_true_pos + frustration_true_neg) / valid_reports.shape[0]
        frustration_precision = frustration_true_pos / (frustration_true_pos + frustration_false_pos) if (frustration_true_pos + frustration_false_pos) > 0 else 0
        frustration_recall = frustration_true_pos / (frustration_true_pos + frustration_false_neg) if (frustration_true_pos + frustration_false_neg) > 0 else 0
        frustration_f1 = 2 * (frustration_precision * frustration_recall) / (frustration_precision + frustration_recall) if (frustration_precision + frustration_recall) > 0 else 0
        
        # Compile results
        performance = {
            'fatigue': {
                'accuracy': fatigue_accuracy,
                'precision': fatigue_precision,
                'recall': fatigue_recall,
                'f1_score': fatigue_f1,
                'confusion_matrix': {
                    'true_negatives': fatigue_true_neg,
                    'false_positives': fatigue_false_pos,
                    'false_negatives': fatigue_false_neg,
                    'true_positives': fatigue_true_pos
                }
            },
            'frustration': {
                'accuracy': frustration_accuracy,
                'precision': frustration_precision,
                'recall': frustration_recall,
                'f1_score': frustration_f1,
                'confusion_matrix': {
                    'true_negatives': frustration_true_neg,
                    'false_positives': frustration_false_pos,
                    'false_negatives': frustration_false_neg,
                    'true_positives': frustration_true_pos
                }
            }
        }
        
        # Calculate per-participant metrics for standard deviation
        participant_metrics = {}
        for participant_id in valid_reports['participant_id'].unique():
            participant_reports = valid_reports[valid_reports['participant_id'] == participant_id]
            
            # Skip if too few reports
            if participant_reports.shape[0] < 3:
                continue
                
            # Calculate fatigue metrics
            p_fatigue_true_pos = participant_reports[(participant_reports['is_fatigued'] == 1) & 
                                                   (participant_reports['system_fatigue_detection'] == 1)].shape[0]
            p_fatigue_true_neg = participant_reports[(participant_reports['is_fatigued'] == 0) & 
                                                   (participant_reports['system_fatigue_detection'] == 0)].shape[0]
            p_fatigue_false_pos = participant_reports[(participant_reports['is_fatigued'] == 0) & 
                                                    (participant_reports['system_fatigue_detection'] == 1)].shape[0]
            p_fatigue_false_neg = participant_reports[(participant_reports['is_fatigued'] == 1) & 
                                                    (participant_reports['system_fatigue_detection'] == 0)].shape[0]
            
            p_fatigue_accuracy = (p_fatigue_true_pos + p_fatigue_true_neg) / participant_reports.shape[0]
            p_fatigue_precision = p_fatigue_true_pos / (p_fatigue_true_pos + p_fatigue_false_pos) if (p_fatigue_true_pos + p_fatigue_false_pos) > 0 else 0
            p_fatigue_recall = p_fatigue_true_pos / (p_fatigue_true_pos + p_fatigue_false_neg) if (p_fatigue_true_pos + p_fatigue_false_neg) > 0 else 0
            p_fatigue_f1 = 2 * (p_fatigue_precision * p_fatigue_recall) / (p_fatigue_precision + p_fatigue_recall) if (p_fatigue_precision + p_fatigue_recall) > 0 else 0
            
            # Calculate frustration metrics
            p_frustration_true_pos = participant_reports[(participant_reports['is_frustrated'] == 1) & 
                                                      (participant_reports['system_frustration_detection'] == 1)].shape[0]
            p_frustration_true_neg = participant_reports[(participant_reports['is_frustrated'] == 0) & 
                                                      (participant_reports['system_frustration_detection'] == 0)].shape[0]
            p_frustration_false_pos = participant_reports[(participant_reports['is_frustrated'] == 0) & 
                                                       (participant_reports['system_frustration_detection'] == 1)].shape[0]
            p_frustration_false_neg = participant_reports[(participant_reports['is_frustrated'] == 1) & 
                                                       (participant_reports['system_frustration_detection'] == 0)].shape[0]
            
            p_frustration_accuracy = (p_frustration_true_pos + p_frustration_true_neg) / participant_reports.shape[0]
            p_frustration_precision = p_frustration_true_pos / (p_frustration_true_pos + p_frustration_false_pos) if (p_frustration_true_pos + p_frustration_false_pos) > 0 else 0
            p_frustration_recall = p_frustration_true_pos / (p_frustration_true_pos + p_frustration_false_neg) if (p_frustration_true_pos + p_frustration_false_neg) > 0 else 0
            p_frustration_f1 = 2 * (p_frustration_precision * p_frustration_recall) / (p_frustration_precision + p_frustration_recall) if (p_frustration_precision + p_frustration_recall) > 0 else 0
            
            participant_metrics[participant_id] = {
                'fatigue': {
                    'accuracy': p_fatigue_accuracy,
                    'precision': p_fatigue_precision,
                    'recall': p_fatigue_recall,
                    'f1_score': p_fatigue_f1
                },
                'frustration': {
                    'accuracy': p_frustration_accuracy,
                    'precision': p_frustration_precision,
                    'recall': p_frustration_recall,
                    'f1_score': p_frustration_f1
                }
            }
        
        # Calculate standard deviations
        if participant_metrics:
            fatigue_accuracies = [m['fatigue']['accuracy'] for m in participant_metrics.values()]
            fatigue_precisions = [m['fatigue']['precision'] for m in participant_metrics.values()]
            fatigue_recalls = [m['fatigue']['recall'] for m in participant_metrics.values()]
            fatigue_f1s = [m['fatigue']['f1_score'] for m in participant_metrics.values()]
            
            frustration_accuracies = [m['frustration']['accuracy'] for m in participant_metrics.values()]
            frustration_precisions = [m['frustration']['precision'] for m in participant_metrics.values()]
            frustration_recalls = [m['frustration']['recall'] for m in participant_metrics.values()]
            frustration_f1s = [m['frustration']['f1_score'] for m in participant_metrics.values()]
            
            performance['fatigue']['accuracy_std'] = np.std(fatigue_accuracies)
            performance['fatigue']['precision_std'] = np.std(fatigue_precisions)
            performance['fatigue']['recall_std'] = np.std(fatigue_recalls)
            performance['fatigue']['f1_score_std'] = np.std(fatigue_f1s)
            
            performance['frustration']['accuracy_std'] = np.std(frustration_accuracies)
            performance['frustration']['precision_std'] = np.std(frustration_precisions)
            performance['frustration']['recall_std'] = np.std(frustration_recalls)
            performance['frustration']['f1_score_std'] = np.std(frustration_f1s)
        
        # Save results
        json_path = os.path.join(output_dir, 'detection_performance.json')
        with open(json_path, 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info(f"Calculated detection performance metrics, saved to {json_path}")
        return performance
