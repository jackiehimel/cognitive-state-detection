"""
Test script to verify functionality of the user study components.

This script tests the self-reporting, data collection, and analysis modules
to ensure they work correctly together.
"""

import os
import sys
import time
import datetime
import threading
import json
import numpy as np
import logging
import tkinter as tk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import study components
from src.study.self_reporting import SelfReportManager
from src.study.data_collection import StudyDataCollector
from src.study.analysis import StudyAnalysis

# Directory for test results
TEST_RESULTS_DIR = 'results/test_study'
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def simulate_eye_metrics():
    """
    Simulate eye metrics data for testing.
    
    Returns:
    --------
    dict
        Dictionary of simulated eye metrics
    """
    # Generate random eye metrics
    perclos = np.random.uniform(0.1, 0.4)
    blink_rate = np.random.uniform(10, 25)
    blink_duration = np.random.uniform(0.1, 0.5)
    pupil_diameter = np.random.uniform(0.3, 0.7)
    pupil_variance = np.random.uniform(0.01, 0.1)
    gaze_scanpath_area = np.random.uniform(0.1, 0.6)
    avg_fixation_duration = np.random.uniform(0.2, 0.6)
    saccade_amplitude = np.random.uniform(1.0, 5.0)
    
    return {
        'perclos': perclos,
        'blink_rate': blink_rate,
        'blink_duration': blink_duration,
        'pupil_diameter': pupil_diameter,
        'pupil_variance': pupil_variance,
        'gaze_scanpath_area': gaze_scanpath_area,
        'avg_fixation_duration': avg_fixation_duration,
        'saccade_amplitude': saccade_amplitude
    }

def simulate_cognitive_state(features):
    """
    Simulate cognitive state prediction based on features.
    
    Parameters:
    -----------
    features : dict
        Dictionary of eye metrics
        
    Returns:
    --------
    dict
        Dictionary containing prediction results
    """
    # Simple thresholds for simulation
    fatigue_prob = min(1.0, max(0.0, features['perclos'] * 2.0 + 
                                     features['blink_rate'] / 40.0 +
                                     features['blink_duration'] * 0.5 - 0.3))
    
    frustration_prob = min(1.0, max(0.0, features['pupil_diameter'] * 0.8 + 
                                         features['pupil_variance'] * 3.0 +
                                         features['gaze_scanpath_area'] * 0.6 - 0.2))
    
    # Determine detected state
    if fatigue_prob > 0.6:
        detected_state = 'fatigue'
    elif frustration_prob > 0.6:
        detected_state = 'frustration'
    else:
        detected_state = 'neutral'
    
    return {
        'fatigue_probability': fatigue_prob,
        'frustration_probability': frustration_prob,
        'detected_state': detected_state
    }

def test_self_reporting(reduced_interval=True):
    """
    Test the self-reporting functionality.
    
    Parameters:
    -----------
    reduced_interval : bool
        If True, use a shorter interval for testing
        
    Returns:
    --------
    bool
        True if test passed, False otherwise
    """
    logger.info("Testing self-reporting functionality")
    
    # Initialize self-report manager with shorter interval for testing
    interval = 10 if reduced_interval else 1200  # 10 seconds instead of 20 minutes
    manager = SelfReportManager(prompt_interval=interval, results_dir=TEST_RESULTS_DIR)
    
    # Start a session
    success = manager.start_session('test_participant', 'test_task')
    if not success:
        logger.error("Failed to start self-report session")
        return False
    
    # Collect baseline (will show prompt)
    logger.info("Collecting baseline cognitive state (prompt will appear)")
    baseline = manager.collect_baseline()
    if baseline is None:
        logger.error("Failed to collect baseline")
        return False
    
    if reduced_interval:
        # Wait for prompt interval plus buffer
        logger.info(f"Waiting for {interval} seconds for prompt to appear...")
        time.sleep(interval + 2)
    else:
        # For manual testing, just wait a bit and then stop
        time.sleep(5)
    
    # Stop session
    report_file = manager.stop_session()
    if report_file is None:
        logger.error("Failed to stop session and save reports")
        return False
    
    logger.info(f"Self-reporting test completed, reports saved to {report_file}")
    return True

def test_data_collection():
    """
    Test the data collection functionality.
    
    Returns:
    --------
    bool
        True if test passed, False otherwise
    """
    logger.info("Testing data collection functionality")
    
    # Initialize data collector
    collector = StudyDataCollector(window_size=5, results_dir=TEST_RESULTS_DIR)
    
    # Start a session
    success = collector.start_session('test_participant', 'test_task')
    if not success:
        logger.error("Failed to start data collection session")
        return False
    
    # Simulate recording data for 30 seconds
    logger.info("Simulating data collection for 30 seconds")
    start_time = time.time()
    
    while time.time() - start_time < 30:
        # Generate simulated data every 5 seconds
        features = simulate_eye_metrics()
        prediction = simulate_cognitive_state(features)
        
        # Record data
        collector.record_features(features)
        collector.record_prediction(prediction)
        
        # Print current state
        logger.info(f"Recorded state: {prediction['detected_state']} "
                  f"(Fatigue: {prediction['fatigue_probability']:.2f}, "
                  f"Frustration: {prediction['frustration_probability']:.2f})")
        
        # Simulate self-report at 15 seconds
        if 14 < time.time() - start_time < 16:
            self_report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'fatigue': 3,
                'frustration': 2,
                'notes': 'Test self-report',
                'type': 'periodic',
                'minutes_elapsed': 0
            }
            collector.record_self_report(self_report)
            logger.info("Recorded simulated self-report")
        
        time.sleep(5)
    
    # Get session summary
    summary = collector.get_session_summary()
    logger.info(f"Session summary: {summary}")
    
    # Export to CSV
    csv_files = collector.export_to_csv()
    logger.info(f"Exported data to CSV: {csv_files}")
    
    # Stop session
    prediction_file, feature_file = collector.stop_session()
    if prediction_file is None or feature_file is None:
        logger.error("Failed to stop session and save data")
        return False
    
    logger.info(f"Data collection test completed, data saved to {prediction_file} and {feature_file}")
    return True

def test_analysis():
    """
    Test the analysis functionality.
    
    Returns:
    --------
    bool
        True if test passed, False otherwise
    """
    logger.info("Testing analysis functionality")
    
    # First, generate some test data if it doesn't exist
    if not os.path.exists(os.path.join(TEST_RESULTS_DIR, 'test_participant_test_task_features.json')):
        logger.info("Generating test data for analysis")
        
        # Create simulated feature data
        feature_data = {
            'participant_id': 'test_participant',
            'task_id': 'test_task',
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
            'features': []
        }
        
        # Create simulated prediction data
        prediction_data = {
            'participant_id': 'test_participant',
            'task_id': 'test_task',
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
            'window_size_seconds': 5,
            'predictions': [],
            'self_reports': []
        }
        
        # Generate data for 2 hours with 6 self-reports
        start_time = datetime.datetime.now()
        for i in range(720):  # 2 hours * 60 minutes * 60 seconds / 10 seconds = 720
            timestamp = start_time + datetime.timedelta(seconds=i*10)
            
            # Add feature data
            features = simulate_eye_metrics()
            feature_record = {
                'timestamp': timestamp.isoformat(),
                'elapsed_seconds': i*10
            }
            feature_record.update(features)
            feature_data['features'].append(feature_record)
            
            # Add prediction data
            prediction = simulate_cognitive_state(features)
            prediction_record = {
                'timestamp': timestamp.isoformat(),
                'elapsed_seconds': i*10,
                'fatigue_probability': prediction['fatigue_probability'],
                'frustration_probability': prediction['frustration_probability'],
                'detected_state': prediction['detected_state']
            }
            prediction_data['predictions'].append(prediction_record)
            
            # Add self-reports every 20 minutes
            if i % 120 == 0 and i > 0:  # Every 20 minutes
                # Create self-report with random values
                fatigue = np.random.randint(1, 8)
                frustration = np.random.randint(1, 8)
                
                self_report = {
                    'timestamp': timestamp.isoformat(),
                    'fatigue': fatigue,
                    'frustration': frustration,
                    'notes': f'Self-report at {i//60} minutes',
                    'type': 'periodic',
                    'minutes_elapsed': i//6,
                    'participant_id': 'test_participant',
                    'task_id': 'test_task',
                    'is_fatigued': 1 if fatigue >= 4 else 0,
                    'is_frustrated': 1 if frustration >= 5 else 0
                }
                
                # Add system predictions for correlation
                fatigue_probs = [p['fatigue_probability'] for p in prediction_data['predictions'][-120:]]
                frustration_probs = [p['frustration_probability'] for p in prediction_data['predictions'][-120:]]
                
                avg_fatigue_prob = np.mean(fatigue_probs) if fatigue_probs else 0
                avg_frustration_prob = np.mean(frustration_probs) if frustration_probs else 0
                
                system_fatigue = 1 if (np.array(fatigue_probs) > 0.5).mean() > 0.6 else 0
                system_frustration = 1 if (np.array(frustration_probs) > 0.5).mean() > 0.6 else 0
                
                state_counts = {
                    'neutral': sum(1 for p in prediction_data['predictions'][-120:] if p['detected_state'] == 'neutral'),
                    'fatigue': sum(1 for p in prediction_data['predictions'][-120:] if p['detected_state'] == 'fatigue'),
                    'frustration': sum(1 for p in prediction_data['predictions'][-120:] if p['detected_state'] == 'frustration')
                }
                
                self_report['window_predictions_count'] = 120
                self_report['system_avg_fatigue_prob'] = avg_fatigue_prob
                self_report['system_avg_frustration_prob'] = avg_frustration_prob
                self_report['system_fatigue_detection'] = system_fatigue
                self_report['system_frustration_detection'] = system_frustration
                self_report['system_state_counts'] = state_counts
                self_report['fatigue_agreement'] = 1 if (self_report['is_fatigued'] == system_fatigue) else 0
                self_report['frustration_agreement'] = 1 if (self_report['is_frustrated'] == system_frustration) else 0
                
                prediction_data['self_reports'].append(self_report)
        
        # Save simulated data
        with open(os.path.join(TEST_RESULTS_DIR, 'test_participant_test_task_20250526_000000_features.json'), 'w') as f:
            json.dump(feature_data, f, indent=2)
            
        with open(os.path.join(TEST_RESULTS_DIR, 'test_participant_test_task_20250526_000000_predictions.json'), 'w') as f:
            json.dump(prediction_data, f, indent=2)
            
        with open(os.path.join(TEST_RESULTS_DIR, 'test_participant_test_task_20250526_000000_self_reports.json'), 'w') as f:
            json.dump({'reports': prediction_data['self_reports']}, f, indent=2)
    
    # Initialize analysis module
    analyzer = StudyAnalysis(results_dir=TEST_RESULTS_DIR)
    
    # Test loading participant data
    success = analyzer.load_participant_data('test_participant', 'test_task')
    if not success:
        logger.error("Failed to load participant data")
        return False
    
    # Test correlation analysis
    correlations = analyzer.calculate_eye_metric_correlations()
    if correlations is None:
        logger.error("Failed to calculate correlations")
        return False
    
    logger.info(f"Correlation results: {correlations}")
    
    # Test performance analysis
    performance = analyzer.calculate_detection_performance()
    if performance is None:
        logger.error("Failed to calculate detection performance")
        return False
    
    logger.info(f"Performance results: {performance}")
    
    logger.info("Analysis test completed")
    return True

def run_all_tests():
    """
    Run all tests.
    
    Returns:
    --------
    bool
        True if all tests passed, False otherwise
    """
    logger.info("Starting functionality tests for user study components")
    
    # Test self-reporting (in a separate thread to handle GUI)
    logger.info("Testing self-reporting (GUI will appear)...")
    self_reporting_thread = threading.Thread(target=test_self_reporting)
    self_reporting_thread.start()
    self_reporting_thread.join(timeout=60)  # Wait up to 60 seconds
    
    # Test data collection
    logger.info("Testing data collection...")
    data_collection_success = test_data_collection()
    if not data_collection_success:
        logger.error("Data collection test failed")
        return False
    
    # Test analysis
    logger.info("Testing analysis...")
    analysis_success = test_analysis()
    if not analysis_success:
        logger.error("Analysis test failed")
        return False
    
    logger.info("All tests completed successfully!")
    return True

if __name__ == "__main__":
    run_all_tests()
