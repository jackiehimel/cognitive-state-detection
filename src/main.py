"""
Main script for running the cognitive state detection pipeline.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import cv2
import time
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.dataset_preprocessing import (
    preprocess_zju_dataset, preprocess_drowsiness_dataset,
    preprocess_cew_dataset, preprocess_cafe_dataset,
    preprocess_affectnet, preprocess_eyegaze_dataset,
    verify_dataset_availability
)
from src.features.extraction import EyeFeatureExtractor, extract_features_from_datasets
from src.features.calibration import AdaptiveCalibration, CalibrationUI, ContinuousCalibrationTracker
from src.models.cognitive_state_detection import (
    CognitiveStateDetector, prepare_feature_importance_analysis
)
from src.visualization.visualizations import generate_all_figures

# Configure logging
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
    logger.info("Starting dataset preprocessing")
    
    # Check dataset availability
    availability = verify_dataset_availability()
    
    datasets = {}
    
    # Process ZJU Eye-Blink Dataset
    if availability['ZJU Eye-Blink']:
        logger.info("Preprocessing ZJU Eye-Blink dataset")
        datasets['zju'] = preprocess_zju_dataset()
    else:
        logger.warning("ZJU Eye-Blink dataset not available, skipping preprocessing")
        
    # Process Real-Life Drowsiness Dataset
    if availability['Real-Life Drowsiness']:
        logger.info("Preprocessing Real-Life Drowsiness dataset")
        datasets['drowsiness'] = preprocess_drowsiness_dataset()
    else:
        logger.warning("Real-Life Drowsiness dataset not available, skipping preprocessing")
        
    # Process CEW Dataset
    if availability['CEW']:
        logger.info("Preprocessing CEW dataset")
        datasets['cew'] = preprocess_cew_dataset()
    else:
        logger.warning("CEW dataset not available, skipping preprocessing")
        
    # Process CAFE Dataset
    if availability['CAFE']:
        logger.info("Preprocessing CAFE dataset")
        datasets['cafe'] = preprocess_cafe_dataset()
    else:
        logger.warning("CAFE dataset not available, skipping preprocessing")
        
    # Process AffectNet
    if availability['AffectNet']:
        logger.info("Preprocessing AffectNet dataset")
        datasets['affectnet'] = preprocess_affectnet()
    else:
        logger.warning("AffectNet dataset not available, skipping preprocessing")
        
    # Process Eye Gaze Net
    if availability['Eye Gaze Net']:
        logger.info("Preprocessing Eye Gaze Net dataset")
        datasets['eyegaze'] = preprocess_eyegaze_dataset()
    else:
        logger.warning("Eye Gaze Net dataset not available, skipping preprocessing")
    
    logger.info(f"Dataset preprocessing complete. Processed {len(datasets)} datasets.")
    
    return datasets


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
    logger.info("Starting feature extraction")
    
    # Initialize feature extractor
    extractor = EyeFeatureExtractor()
    
    # Extract features from training data
    features_train = extract_features_from_datasets(datasets, extractor)
    
    # Extract features from validation data
    # Note: In a real implementation, extract_features_from_datasets would be called
    # with validation datasets. Here we'll just use a subset of training features.
    
    # Create a simplified validation set by sampling from training features
    features_val = {
        'blink_patterns': [],
        'eye_closure_patterns': [],
        'pupil_metrics': [],
        'gaze_patterns': [],
        'labels': []
    }
    
    # Sample a subset of each feature type for validation
    if features_train['blink_patterns']:
        n_samples = max(1, int(len(features_train['blink_patterns']) * 0.2))
        indices = np.random.choice(len(features_train['blink_patterns']), n_samples, replace=False)
        features_val['blink_patterns'] = [features_train['blink_patterns'][i] for i in indices]
        
    if features_train['eye_closure_patterns']:
        n_samples = max(1, int(len(features_train['eye_closure_patterns']) * 0.2))
        indices = np.random.choice(len(features_train['eye_closure_patterns']), n_samples, replace=False)
        features_val['eye_closure_patterns'] = [features_train['eye_closure_patterns'][i] for i in indices]
        
    if features_train['pupil_metrics']:
        n_samples = max(1, int(len(features_train['pupil_metrics']) * 0.2))
        indices = np.random.choice(len(features_train['pupil_metrics']), n_samples, replace=False)
        features_val['pupil_metrics'] = [features_train['pupil_metrics'][i] for i in indices]
        
    if features_train['gaze_patterns']:
        n_samples = max(1, int(len(features_train['gaze_patterns']) * 0.2))
        indices = np.random.choice(len(features_train['gaze_patterns']), n_samples, replace=False)
        features_val['gaze_patterns'] = [features_train['gaze_patterns'][i] for i in indices]
    
    # Sample corresponding labels
    for i, feature_type in enumerate(['blink_patterns', 'eye_closure_patterns', 'pupil_metrics', 'gaze_patterns']):
        if features_val[feature_type]:
            features_val['labels'].extend([features_train['labels'][i]] * len(features_val[feature_type]))
    
    logger.info(f"Feature extraction complete. "
               f"Training features: {sum(len(features_train[k]) for k in features_train if k != 'labels')}, "
               f"Validation features: {sum(len(features_val[k]) for k in features_val if k != 'labels')}")
    
    return features_train, features_val


def train_models(features_train, features_val, models_dir='models', load_models=False):
    """
    Train cognitive state detection models.
    
    Parameters:
    -----------
    features_train : dict
        Dictionary containing training features
    features_val : dict
        Dictionary containing validation features
    models_dir : str
        Directory to save or load models
    load_models : bool
        Whether to load pre-trained models instead of training new ones
        
    Returns:
    --------
    tuple
        (detector, results, feature_importances, ablation_results)
    """
    logger.info("Starting model training and evaluation")
    
    # Initialize cognitive state detector
    detector = CognitiveStateDetector(models_dir=models_dir)
    
    if load_models:
        # Load pre-trained models
        logger.info("Loading pre-trained models")
        success = detector.load_models()
        if not success:
            logger.warning("Failed to load models, fallback to training new models")
            load_models = False
    
    if not load_models:
        # Train models
        logger.info("Training new models")
        detector.train_models(features_train, grid_search=False)  # Set grid_search=True for hyperparameter tuning
    
    # Evaluate models
    logger.info("Evaluating models")
    results, fatigue_importances, frustration_importances = detector.evaluate_models(features_val)
    
    # Prepare feature importance analysis
    feature_names = [
        'PERCLOS', 'Blink Rate', 'Blink Duration', 'Pupil Size',
        'Pupil Variance', 'Gaze Fixation', 'Gaze Dispersion',
        'EAR Mean', 'EAR Std', 'EAR Min', 'EAR Max'
    ]
    feature_importances = prepare_feature_importance_analysis(
        fatigue_importances, frustration_importances, feature_names
    )
    
    # Perform ablation studies
    logger.info("Performing ablation studies")
    ablation_results = detector.perform_ablation_studies(features_train, features_val)
    
    logger.info("Model training and evaluation complete")
    
    return detector, results, feature_importances, ablation_results


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
    logger.info("Generating visualizations")
    
    # Generate all figures and tables
    paths = generate_all_figures(results, feature_importances, ablation_results, output_dir)
    
    logger.info(f"Visualizations generated and saved to {output_dir}")
    
    return paths


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
    logger.info("Starting webcam-based cognitive state detection")
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.webcam_device)
    if not cap.isOpened():
        logger.error(f"Failed to open webcam device {args.webcam_device}")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize feature extractor
    extractor = EyeFeatureExtractor()
    
    # Initialize adaptive calibration
    calibration = AdaptiveCalibration(calibration_file=args.calibration_file)
    
    # Run initial calibration if requested or if no calibration exists
    if args.calibrate or not os.path.exists(args.calibration_file):
        logger.info("Starting initial calibration")
        calibration_ui = CalibrationUI(calibration)
        calibration_ui.start_calibration(width, height)
        
        # Run calibration loop
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame during calibration")
                break
                
            # Extract features
            features = extractor.process_frame(frame)
            
            # Update calibration UI with current frame and features
            if not calibration_ui.update_frame(frame, features):
                break  # Calibration complete
    else:
        # Try to load existing calibration
        if not calibration.load_calibration():
            logger.warning("Failed to load calibration, but continuing without calibration")
    
    # Initialize continuous calibration tracker if enabled
    continuous_tracker = None
    if args.enable_continuous_calibration:
        logger.info("Continuous WebGazer.js-style calibration enabled")
        continuous_tracker = ContinuousCalibrationTracker(calibration)
    
    # Create window
    cv2.namedWindow('Cognitive State Detection', cv2.WINDOW_NORMAL)
    
    # Variables for continuous calibration
    mouse_position = None
    last_mouse_click = None
    calibration_reminder_time = time.time() + 300  # 5 minutes initial reminder
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
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
            
            # Visualize landmarks
            visualization = extractor.visualize_landmarks(frame)
            
            # Add prediction to visualization
            cv2.putText(visualization, f"State: {prediction['cognitive_state']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, f"Fatigue: {prediction['fatigue_probability']:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(visualization, f"Frustration: {prediction['frustration_probability']:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show eye metrics
            cv2.putText(visualization, f"PERCLOS: {features['perclos']:.2f}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(visualization, f"Blink rate: {features['blink_rate']:.2f} bpm", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show gaze information if available
            if gaze_pos:
                cv2.putText(visualization, f"Gaze confidence: {gaze_confidence:.2f}", (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.circle(visualization, gaze_pos, 10, (0, 255, 255), -1)
            
            # Display calibration status
            if calibration.needs_calibration():
                if time.time() > calibration_reminder_time:
                    cv2.putText(visualization, "Press 'c' to recalibrate", (width // 2 - 150, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Continuous calibration with mouse clicks if enabled
            if continuous_tracker and last_mouse_click:
                continuous_tracker.track_mouse_click(last_mouse_click, features)
                last_mouse_click = None  # Reset after processing
            
            # Display visualization
            cv2.imshow('Cognitive State Detection', visualization)
            
            # Handle keyboard/mouse input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q'
            if key == ord('q'):
                break
                
            # Recalibrate on 'c'
            elif key == ord('c'):
                logger.info("Starting recalibration")
                calibration_ui = CalibrationUI(calibration)
                calibration_ui.start_calibration(width, height)
                
                # Run calibration loop
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    features = extractor.process_frame(frame)
                    if not calibration_ui.update_frame(frame, features):
                        break
                        
                # Reset reminder timer after calibration
                calibration_reminder_time = time.time() + 300
            
            # Handle mouse events for continuous calibration
            if args.enable_continuous_calibration:
                # Simulating mouse events for this example
                # In a real IDE integration, these would come from actual mouse/keyboard events
                # Here we're using keyboard keys as proxies for mouse interaction
                if key == ord('m'):  # 'm' simulates a mouse click
                    mouse_x, mouse_y = width // 2, height // 2  # Center of screen for demo
                    last_mouse_click = (mouse_x, mouse_y)
                    logger.debug(f"Simulated mouse click at {mouse_x}, {mouse_y}")
                
    finally:
        # Save calibration before exiting
        if calibration.is_calibrated:
            calibration.save_calibration()
            
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    logger.info("Webcam-based cognitive state detection finished")


def main():
    """Main function to run the cognitive state detection pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Preprocess datasets
        if not args.skip_preprocessing:
            datasets = preprocess_datasets(args.data_dir)
        else:
            logger.info("Skipping dataset preprocessing")
            # Create dummy datasets for testing
            datasets = {
                'dummy': {
                    'train': {'videos': [], 'blinks': []},
                    'val': {'videos': [], 'blinks': []}
                }
            }
        
        # Extract features
        if not args.skip_feature_extraction:
            features_train, features_val = extract_features(datasets)
        else:
            logger.info("Skipping feature extraction")
            # Create dummy features for testing
            features_train = {
                'blink_patterns': [{'perclos': 0.1, 'blink_rate': 20}],
                'eye_closure_patterns': [{'ear_mean': 0.3, 'ear_std': 0.05}],
                'pupil_metrics': [{'pupil_size_mean': 5.0, 'pupil_size_std': 0.5}],
                'gaze_patterns': [{'gaze_x_mean': 0.1, 'gaze_y_mean': 0.2}],
                'labels': ['neutral', 'fatigue', 'frustration', 'neutral']
            }
            features_val = features_train
        
        # Train and evaluate models
        if not args.skip_training:
            detector, results, feature_importances, ablation_results = train_models(
                features_train, features_val, models_dir=os.path.join(args.output_dir, 'models'),
                load_models=args.load_models
            )
        else:
            logger.info("Skipping model training and evaluation")
            # Create dummy results for testing
            detector = CognitiveStateDetector(models_dir=os.path.join(args.output_dir, 'models'))
            
            if args.webcam:
                # If webcam mode is requested, try to load models
                success = detector.load_models()
                if not success:
                    logger.error("Failed to load models required for webcam mode")
                    return
            
            results = {
                'fatigue': {
                    'accuracy': 0.85, 'precision': 0.82, 'recall': 0.87,
                    'f1': 0.84, 'auc': 0.91, 'confusion_matrix': np.array([[85, 15], [13, 87]])
                },
                'frustration': {
                    'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82,
                    'f1': 0.80, 'auc': 0.88, 'confusion_matrix': np.array([[80, 20], [18, 82]])
                }
            }
            
            feature_importances = {
                'features': ['PERCLOS', 'Blink Rate', 'Blink Duration', 'Pupil Size', 'Pupil Variance', 'Gaze Fixation'],
                'fatigue': [0.25, 0.20, 0.15, 0.15, 0.15, 0.10],
                'frustration': [0.15, 0.15, 0.10, 0.20, 0.20, 0.20]
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
        
        # Run webcam detection if requested
        if args.webcam:
            # Ensure calibration directory exists
            calibration_dir = os.path.dirname(args.calibration_file)
            if calibration_dir:
                os.makedirs(calibration_dir, exist_ok=True)
                
            # Run webcam detection with calibration options
            run_webcam_detection(detector, args)
            return
        
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
