"""
Main script for running the cognitive state detection pipeline.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# Local imports
from models.cognitive_state_detection import CognitiveStateDetector
from features.extraction import EyeFeatureExtractor
from features.calibration import AdaptiveCalibration
from ui.adaptive_interface import UIAdaptationManager
from study.data_collection import StudyDataCollector
from study.self_reporting import SelfReportManager
from study.analysis import StudyAnalysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cognitive_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cognitive State Detection Pipeline')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing datasets')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip dataset preprocessing')
    parser.add_argument('--skip-feature-extraction', action='store_true',
                        help='Skip feature extraction')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip model evaluation')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--load-models', action='store_true',
                        help='Load pre-trained models instead of training new ones')
    parser.add_argument('--webcam', action='store_true',
                        help='Run cognitive state detection on webcam feed')
    parser.add_argument('--webcam-device', type=int, default=0,
                        help='Webcam device index')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run calibration before starting')
    parser.add_argument('--calibration-file', type=str, default='results/calibration.json',
                        help='File to save/load calibration data')
    parser.add_argument('--enable-continuous-calibration', action='store_true',
                        help='Enable WebGazer.js-style continuous calibration during long sessions')
    parser.add_argument('--user-study', action='store_true',
                        help='Enable user study components (self-reporting, data collection)')
    parser.add_argument('--participant-id', type=str, default=None,
                        help='Participant ID for user study')
    parser.add_argument('--task-id', type=str, default=None,
                        help='Task ID for user study (e.g., debugging, feature_implementation)')
    parser.add_argument('--prompt-interval', type=int, default=1200,
                        help='Interval between self-report prompts in seconds (default: 1200 = 20 minutes)')
    parser.add_argument('--adaptive-interface', action='store_true',
                        help='Enable adaptive IDE interface based on cognitive state')
    return parser.parse_args()


def preprocess_datasets(data_dir):
    """
    Preprocess all datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing datasets
        
    Returns:
    --------
    dict
        Dictionary containing preprocessed datasets
    """
    logger.info("Preprocessing datasets...")
    # Implementation omitted for clarity
    return {
        'train': {'fatigue': [], 'frustration': [], 'neutral': []},
        'validation': {'fatigue': [], 'frustration': [], 'neutral': []}
    }


def extract_features(datasets):
    """
    Extract features from datasets.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing preprocessed datasets
        
    Returns:
    --------
    tuple
        (features_train, features_val) - Training and validation features
    """
    logger.info("Extracting features...")
    # Implementation omitted for clarity
    return {}, {}


def generate_visualizations(results, feature_importances, ablation_results, output_dir='results'):
    """
    Generate visualizations for thesis.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing validation results
    feature_importances : dict
        Dictionary containing feature importance information
    ablation_results : dict
        Dictionary containing ablation study results
    output_dir : str
        Directory to save output visualizations
        
    Returns:
    --------
    dict
        Dictionary containing paths to generated visualizations
    """
    logger.info("Generating visualizations...")
    vis_paths = {}
    
    # Implementation omitted for clarity
    
    return vis_paths


def run_webcam_detection(detector, args):
    """
    Run cognitive state detection on webcam feed.
    
    Parameters:
    -----------
    detector : CognitiveStateDetector
        Trained cognitive state detector
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting webcam detection...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.webcam_device)
    if not cap.isOpened():
        logger.error(f"Error: Could not open webcam {args.webcam_device}")
        return
    
    # Create feature extractor
    extractor = EyeFeatureExtractor()
    
    # Create calibration object
    calibration = AdaptiveCalibration(calibration_file=args.calibration_file)
    
    # Load calibration data if available
    if os.path.exists(args.calibration_file):
        calibration.load_calibration()
    
    # Run calibration if requested
    if args.calibrate or not calibration.is_calibrated:
        logger.info("Starting calibration procedure...")
        calibration.run_calibration(cap)
        calibration.save_calibration()
    
    # Create data collector for user study if enabled
    data_collector = None
    self_report_prompt = None
    if args.user_study:
        if not args.participant_id or not args.task_id:
            logger.error("Participant ID and Task ID are required for user study")
            return
        
        # Create data collector
        data_collector = StudyDataCollector(
            participant_id=args.participant_id,
            task_id=args.task_id,
            results_dir='results/user_study'
        )
        
        # Create self-report prompt
        self_report_prompt = SelfReportManager(
            prompt_interval=args.prompt_interval,
            results_dir='results/user_study'
        )
    
    # Create UI adaptation manager if enabled
    ui_adaptation_manager = None
    if args.adaptive_interface:
        ui_adaptation_manager = UIAdaptationManager()
    
    # Start detection loop
    frame_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    try:
        logger.info("Starting detection loop...")
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Could not read frame from webcam")
                break
            
            # Process frame
            features = extractor.process_frame(frame)
            
            # Skip if no face detected
            if not features:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Cognitive State Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Make cognitive state prediction
            prediction = detector.predict(features)
            
            # Make gaze prediction if calibrated
            gaze_pos, gaze_confidence = None, 0
            if calibration.is_calibrated:
                gaze_pos, gaze_confidence = calibration.predict_gaze(features)
                
            # Record data for user study if enabled
            if args.user_study and data_collector:
                # Record extracted features
                feature_data = {
                    'perclos': features.get('perclos', 0),
                    'blink_rate': features.get('blink_rate', 0),
                    'blink_duration': features.get('blink_duration', 0),
                    'pupil_diameter': features.get('pupil_diameter', 0),
                    'pupil_variance': features.get('pupil_variance', 0),
                    'gaze_scanpath_area': features.get('gaze_scanpath_area', 0),
                    'avg_fixation_duration': features.get('avg_fixation_duration', 0),
                    'saccade_amplitude': features.get('saccade_amplitude', 0)
                }
                data_collector.record_features(feature_data)
                
                # Record prediction
                prediction_data = {
                    'fatigue_probability': prediction.get('fatigue_probability', 0),
                    'frustration_probability': prediction.get('frustration_probability', 0),
                    'neutral_probability': prediction.get('neutral_probability', 0),
                    'detected_state': prediction.get('cognitive_state', 'neutral'),
                    'confidence': prediction.get('confidence', 0)
                }
                data_collector.record_prediction(prediction_data)
            
            # Update UI adaptation based on cognitive state if enabled
            if args.adaptive_interface and ui_adaptation_manager:
                cognitive_state = prediction.get('cognitive_state', 'neutral')
                state_probabilities = {
                    'neutral': prediction.get('neutral_probability', 1.0 - prediction.get('fatigue_probability', 0) - prediction.get('frustration_probability', 0)),
                    'fatigue': prediction.get('fatigue_probability', 0),
                    'frustration': prediction.get('frustration_probability', 0)
                }
                
                logger.info(f"Adapting UI based on cognitive state: {cognitive_state} (confidence: {prediction.get('confidence', 0):.2f})")
                
                # Apply different UI adaptations based on cognitive state
                if cognitive_state == 'fatigue':
                    # Apply fatigue-specific adaptations
                    ui_adaptation_manager.update_cognitive_state(state_probabilities)
                    logger.info("Applied fatigue adaptations: Increased font size, reduced sidebar width, enhanced line spacing")
                    
                elif cognitive_state == 'frustration':
                    # Apply frustration-specific adaptations
                    ui_adaptation_manager.update_cognitive_state(state_probabilities)
                    logger.info("Applied frustration adaptations: Enhanced error highlighting, expanded help sidebar")
                    
                else:  # neutral state
                    # Reset to standard interface
                    ui_adaptation_manager.update_cognitive_state(state_probabilities)
                    logger.info("Applied neutral adaptations: Standard interface layout")
                
            # Update statistics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Draw information on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw cognitive state
            cognitive_state = prediction.get('cognitive_state', 'neutral')
            confidence = prediction.get('confidence', 0)
            
            if cognitive_state == 'fatigue':
                color = (0, 0, 255)  # Red for fatigue
            elif cognitive_state == 'frustration':
                color = (0, 165, 255)  # Orange for frustration
            else:
                color = (0, 255, 0)  # Green for neutral
                
            cv2.putText(frame, f"State: {cognitive_state.capitalize()} ({confidence:.2f})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw calibration status
            if calibration.is_calibrated:
                cv2.putText(frame, "Calibration: OK", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Calibration: Not calibrated", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw gaze position if available
            if gaze_pos:
                x, y = int(gaze_pos[0] * frame.shape[1]), int(gaze_pos[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(frame, f"Gaze confidence: {gaze_confidence:.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('Cognitive State Detection', frame)
            
            # Check for self-report prompt
            if args.user_study and self_report_prompt:
                current_time = time.time()
                if current_time - last_report_time >= args.prompt_interval:
                    report = self_report_prompt._show_prompt()
                    if report:
                        self_report_prompt.reports.append(report)
                        if data_collector:
                            data_collector.record_self_report(report)
                    last_report_time = current_time
            
            # Continuous calibration if enabled
            if args.enable_continuous_calibration and frame_count % 300 == 0:  # Every ~10 seconds at 30 FPS
                calibration.update_calibration(features, gaze_pos)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.exception(f"Error in webcam detection: {str(e)}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data for user study
        if args.user_study and data_collector:
            data_collector.save_data()
            
            # Run analysis
            analysis = StudyAnalysis('results/user_study')
            analysis.generate_reports(args.participant_id, args.task_id)


def main():
    """Main function to run the cognitive state detection pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run webcam detection if requested
        if args.webcam:
            # Create results directory for user study if needed
            if args.user_study:
                os.makedirs('results/user_study', exist_ok=True)
            
            # Create CognitiveStateDetector instance
            detector = CognitiveStateDetector(models_dir=os.path.join(args.output_dir, 'models'))
            
            # Load models for webcam detection
            success = detector.load_models()
            if not success:
                logger.error("Failed to load models required for webcam mode")
                return
                
            # Run webcam detection with calibration options
            run_webcam_detection(detector, args)
            return
        
        # Create CognitiveStateDetector instance for training/evaluation
        detector = CognitiveStateDetector(models_dir=os.path.join(args.output_dir, 'models'))
        
        # Preprocess datasets and extract features if training is enabled
        if not args.skip_training:
            # Preprocess datasets
            datasets = preprocess_datasets(args.data_dir)
            
            # Extract features
            features_train, features_val = extract_features(datasets)
            
            # Train and evaluate models
            train_results = detector.train_models(
                features_train,
                grid_search=True,
                cv=5
            )
            
            # Evaluate on validation data
            results = detector.evaluate_models(features_val)
            
            # Get feature importances
            feature_importances = None
            if hasattr(detector.fatigue_model, 'feature_importances_'):
                feature_importances = detector.fatigue_model.feature_importances_
                
            # Placeholder for ablation results
            ablation_results = {}
        else:
            logger.info("Skipping model training and evaluation")
            # Create dummy results for testing
            results = {
                'fatigue': {
                    'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87,
                    'f1': 0.85, 'auc': 0.92
                },
                'frustration': {
                    'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84,
                    'f1': 0.82, 'auc': 0.90
                }
            }
            feature_importances = {
                'perclos': 0.18, 'blink_rate': 0.15, 'blink_duration': 0.14,
                'pupil_diameter': 0.12, 'pupil_variance': 0.10,
                'gaze_scanpath_area': 0.09, 'avg_fixation_duration': 0.12,
                'saccade_amplitude': 0.10
            }
            ablation_results = {
                'all_features': results,
                'without_blink_patterns': {
                    'fatigue': {
                        'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82,
                        'f1': 0.80, 'auc': 0.88
                    },
                    'frustration': {
                        'accuracy': 0.78, 'precision': 0.76, 'recall': 0.80,
                        'f1': 0.78, 'auc': 0.85
                    }
                }
            }
        
        # Generate visualizations
        if not args.skip_visualization:
            paths = generate_visualizations(results, feature_importances, ablation_results, args.output_dir)
            
            # Print paths to generated visualizations
            logger.info("Generated visualizations:")
            for name, path in paths.items():
                logger.info(f"  {name}: {path}")
        else:
            logger.info("Skipping visualization generation")
        
        logger.info("Cognitive state detection pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error running cognitive state detection pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
