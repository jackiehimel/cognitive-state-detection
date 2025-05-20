"""
Script for generating thesis figures without running the full pipeline.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.visualizations import (
    create_system_architecture,
    create_results_figure,
    create_performance_table,
    create_perclos_threshold_analysis,
    create_gaze_pattern_visualization,
    generate_all_figures
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Thesis Figures')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save visualizations')
    return parser.parse_args()


def generate_dummy_data():
    """
    Generate dummy data for visualizations when no models are available.
    
    Returns:
    --------
    tuple
        (validation_results, feature_importances, ablation_results)
    """
    logger.info("Generating dummy data for visualizations")
    
    # Create dummy validation results
    validation_results = {
        'fatigue': {
            'accuracy': 0.85, 'precision': 0.82, 'recall': 0.87,
            'f1': 0.84, 'auc': 0.91, 'confusion_matrix': np.array([[85, 15], [13, 87]])
        },
        'frustration': {
            'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82,
            'f1': 0.80, 'auc': 0.88, 'confusion_matrix': np.array([[80, 20], [18, 82]])
        }
    }
    
    # Create dummy feature importances
    feature_importances = {
        'features': ['PERCLOS', 'Blink Rate', 'Blink Duration', 'Pupil Size', 
                    'Pupil Variance', 'Gaze Fixation', 'EAR Mean', 'EAR Std',
                    'Gaze Dispersion', 'Pupil Size Variance'],
        'fatigue': [0.25, 0.20, 0.15, 0.08, 0.07, 0.05, 0.06, 0.05, 0.04, 0.05],
        'frustration': [0.15, 0.12, 0.10, 0.18, 0.15, 0.12, 0.07, 0.04, 0.03, 0.04]
    }
    
    # Create dummy ablation results
    ablation_results = {
        'all_features': validation_results,
        'without_blink_patterns': {
            'fatigue': {
                'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82,
                'f1': 0.80, 'auc': 0.88
            },
            'frustration': {
                'accuracy': 0.78, 'precision': 0.76, 'recall': 0.80,
                'f1': 0.78, 'auc': 0.85
            }
        },
        'without_eye_closure_patterns': {
            'fatigue': {
                'accuracy': 0.75, 'precision': 0.73, 'recall': 0.77,
                'f1': 0.75, 'auc': 0.83
            },
            'frustration': {
                'accuracy': 0.79, 'precision': 0.77, 'recall': 0.81,
                'f1': 0.79, 'auc': 0.86
            }
        },
        'without_pupil_metrics': {
            'fatigue': {
                'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84,
                'f1': 0.82, 'auc': 0.89
            },
            'frustration': {
                'accuracy': 0.75, 'precision': 0.72, 'recall': 0.76,
                'f1': 0.74, 'auc': 0.82
            }
        },
        'without_gaze_patterns': {
            'fatigue': {
                'accuracy': 0.83, 'precision': 0.81, 'recall': 0.86,
                'f1': 0.83, 'auc': 0.90
            },
            'frustration': {
                'accuracy': 0.76, 'precision': 0.73, 'recall': 0.77,
                'f1': 0.75, 'auc': 0.83
            }
        }
    }
    
    return validation_results, feature_importances, ablation_results


def main():
    """Main function to generate thesis figures."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Generate dummy data for visualizations
        validation_results, feature_importances, ablation_results = generate_dummy_data()
        
        # Generate visualizations
        logger.info("Generating thesis figures")
        paths = generate_all_figures(validation_results, feature_importances, ablation_results, args.output_dir)
        
        # Print paths to generated visualizations
        logger.info("Generated visualizations:")
        for name, path in paths.items():
            logger.info(f"  {name}: {path}")
        
        logger.info(f"All thesis figures generated successfully in {args.output_dir}")
        
    except Exception as e:
        logger.exception(f"Error generating thesis figures: {str(e)}")
        raise


if __name__ == "__main__":
    main()
