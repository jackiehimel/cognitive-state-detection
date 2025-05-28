"""
Simple test script for the user study components.
This will test each component individually with clear console output.
"""

import os
import sys
import json
import datetime
import numpy as np
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Starting simple test of user study components...")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Create test directory
TEST_DIR = 'results/simple_test'
os.makedirs(TEST_DIR, exist_ok=True)
print(f"Created test directory: {TEST_DIR}")

# Test imports
print("\n--- Testing imports ---")
try:
    from src.study.self_reporting import SelfReportManager, StanfordSleepinessScale
    print("✓ Successfully imported self_reporting module")
except Exception as e:
    print(f"✗ Error importing self_reporting module: {str(e)}")

try:
    from src.study.data_collection import StudyDataCollector
    print("✓ Successfully imported data_collection module")
except Exception as e:
    print(f"✗ Error importing data_collection module: {str(e)}")

try:
    from src.study.analysis import StudyAnalysis
    print("✓ Successfully imported analysis module")
except Exception as e:
    print(f"✗ Error importing analysis module: {str(e)}")

# Test data generation
print("\n--- Testing data generation ---")
try:
    # Generate some test data
    features = {
        'perclos': random.uniform(0.1, 0.4),
        'blink_rate': random.uniform(10, 25),
        'blink_duration': random.uniform(0.1, 0.5),
        'pupil_diameter': random.uniform(0.3, 0.7),
        'pupil_variance': random.uniform(0.01, 0.1),
        'gaze_scanpath_area': random.uniform(0.1, 0.6),
        'avg_fixation_duration': random.uniform(0.2, 0.6),
        'saccade_amplitude': random.uniform(1.0, 5.0)
    }
    
    # Calculate "predictions" from features
    fatigue_prob = min(1.0, max(0.0, features['perclos'] * 2.0 + 
                              features['blink_rate'] / 40.0 +
                              features['blink_duration'] * 0.5 - 0.3))
    
    frustration_prob = min(1.0, max(0.0, features['pupil_diameter'] * 0.8 + 
                                  features['pupil_variance'] * 3.0 +
                                  features['gaze_scanpath_area'] * 0.6 - 0.2))
    
    prediction = {
        'fatigue_probability': fatigue_prob,
        'frustration_probability': frustration_prob,
        'detected_state': 'fatigue' if fatigue_prob > 0.6 else 
                         ('frustration' if frustration_prob > 0.6 else 'neutral')
    }
    
    print(f"Generated test features: {features}")
    print(f"Generated test prediction: {prediction}")
except Exception as e:
    print(f"✗ Error generating test data: {str(e)}")

# Test data collector initialization
print("\n--- Testing data collector initialization ---")
try:
    collector = StudyDataCollector(window_size=5, results_dir=TEST_DIR)
    print("✓ Successfully initialized StudyDataCollector")
    
    # Test session start
    success = collector.start_session('test_participant', 'test_task')
    print(f"{'✓' if success else '✗'} Start session result: {success}")
    
    # Test recording data
    if success:
        # Record features
        feature_success = collector.record_features(features)
        print(f"{'✓' if feature_success else '✗'} Record features result: {feature_success}")
        
        # Record prediction
        prediction_success = collector.record_prediction(prediction)
        print(f"{'✓' if prediction_success else '✗'} Record prediction result: {prediction_success}")
        
        # Record self-report
        self_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'fatigue': 3,
            'frustration': 2,
            'notes': 'Test self-report',
            'type': 'periodic',
            'minutes_elapsed': 0
        }
        report_success = collector.record_self_report(self_report)
        print(f"{'✓' if report_success else '✗'} Record self-report result: {report_success}")
        
        # Get summary
        summary = collector.get_session_summary()
        print(f"Session summary: {summary}")
        
        # Export to CSV
        csv_files = collector.export_to_csv()
        print(f"CSV export result: {csv_files}")
        
        # Stop session
        pred_file, feat_file = collector.stop_session()
        print(f"Stop session result: prediction_file={pred_file}, feature_file={feat_file}")
except Exception as e:
    print(f"✗ Error testing data collector: {str(e)}")

# Generate sample data for analysis
print("\n--- Generating test data for analysis ---")
try:
    # Create sample files that analysis can read
    feature_data = {
        'participant_id': 'test_participant',
        'task_id': 'test_task',
        'start_time': datetime.datetime.now().isoformat(),
        'end_time': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
        'features': []
    }
    
    prediction_data = {
        'participant_id': 'test_participant',
        'task_id': 'test_task',
        'start_time': datetime.datetime.now().isoformat(),
        'end_time': (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat(),
        'window_size_seconds': 5,
        'predictions': [],
        'self_reports': []
    }
    
    # Generate 10 data points
    for i in range(10):
        # Generate features
        timestamp = datetime.datetime.now() + datetime.timedelta(minutes=i*10)
        features = {
            'perclos': random.uniform(0.1, 0.4),
            'blink_rate': random.uniform(10, 25),
            'blink_duration': random.uniform(0.1, 0.5),
            'pupil_diameter': random.uniform(0.3, 0.7),
            'pupil_variance': random.uniform(0.01, 0.1),
            'gaze_scanpath_area': random.uniform(0.1, 0.6),
            'avg_fixation_duration': random.uniform(0.2, 0.6),
            'saccade_amplitude': random.uniform(1.0, 5.0)
        }
        
        # Add feature record
        feature_record = {
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': i*600
        }
        feature_record.update(features)
        feature_data['features'].append(feature_record)
        
        # Generate prediction
        fatigue_prob = min(1.0, max(0.0, features['perclos'] * 2.0 + 
                                  features['blink_rate'] / 40.0 +
                                  features['blink_duration'] * 0.5 - 0.3))
        
        frustration_prob = min(1.0, max(0.0, features['pupil_diameter'] * 0.8 + 
                                      features['pupil_variance'] * 3.0 +
                                      features['gaze_scanpath_area'] * 0.6 - 0.2))
        
        detected_state = 'fatigue' if fatigue_prob > 0.6 else ('frustration' if frustration_prob > 0.6 else 'neutral')
        
        # Add prediction record
        prediction_record = {
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': i*600,
            'fatigue_probability': fatigue_prob,
            'frustration_probability': frustration_prob,
            'detected_state': detected_state
        }
        prediction_data['predictions'].append(prediction_record)
        
        # Add self-report every other data point
        if i % 2 == 0 and i > 0:
            fatigue = random.randint(1, 7)
            frustration = random.randint(1, 7)
            
            self_report = {
                'timestamp': timestamp.isoformat(),
                'fatigue': fatigue,
                'frustration': frustration,
                'notes': f'Self-report at {i*10} minutes',
                'type': 'periodic',
                'minutes_elapsed': i*10,
                'participant_id': 'test_participant',
                'task_id': 'test_task',
                'is_fatigued': 1 if fatigue >= 4 else 0,
                'is_frustrated': 1 if frustration >= 5 else 0,
                'system_fatigue_detection': 1 if fatigue_prob > 0.5 else 0,
                'system_frustration_detection': 1 if frustration_prob > 0.5 else 0,
                'fatigue_agreement': 1 if (1 if fatigue >= 4 else 0) == (1 if fatigue_prob > 0.5 else 0) else 0,
                'frustration_agreement': 1 if (1 if frustration >= 5 else 0) == (1 if frustration_prob > 0.5 else 0) else 0
            }
            prediction_data['self_reports'].append(self_report)
    
    # Save files
    with open(os.path.join(TEST_DIR, 'test_participant_test_task_20250526_000000_features.json'), 'w') as f:
        json.dump(feature_data, f, indent=2)
    
    with open(os.path.join(TEST_DIR, 'test_participant_test_task_20250526_000000_predictions.json'), 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    with open(os.path.join(TEST_DIR, 'test_participant_test_task_20250526_000000_self_reports.json'), 'w') as f:
        json.dump({'reports': prediction_data['self_reports']}, f, indent=2)
    
    print("✓ Successfully generated test data for analysis")
except Exception as e:
    print(f"✗ Error generating test data for analysis: {str(e)}")

# Test analysis module
print("\n--- Testing analysis module ---")
try:
    analyzer = StudyAnalysis(results_dir=TEST_DIR)
    print("✓ Successfully initialized StudyAnalysis")
    
    # Test loading data
    load_success = analyzer.load_participant_data('test_participant', 'test_task')
    print(f"{'✓' if load_success else '✗'} Load participant data result: {load_success}")
    
    if load_success:
        # Test correlation calculation
        correlations = analyzer.calculate_eye_metric_correlations(output_dir=TEST_DIR)
        print(f"Correlation results: {correlations}")
        
        # Test performance calculation
        performance = analyzer.calculate_detection_performance(output_dir=TEST_DIR)
        print(f"Performance results: {performance}")
except Exception as e:
    print(f"✗ Error testing analysis module: {str(e)}")

print("\n--- Simple test completed ---")
