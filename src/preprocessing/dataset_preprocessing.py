"""
Module for preprocessing various eye-tracking datasets for cognitive state detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define dataset paths - these should be configured based on actual location
ZJU_PATH = os.path.join('data', 'zju_eyeblink')
DROWSINESS_PATH = os.path.join('data', 'drowsiness_dataset')
CEW_PATH = os.path.join('data', 'cew_dataset')
CAFE_PATH = os.path.join('data', 'cafe_dataset')
AFFECTNET_PATH = os.path.join('data', 'affectnet')
EYEGAZE_PATH = os.path.join('data', 'eyegaze_net')
PROGRAMMER_PATH = os.path.join('data', 'programmer_productivity')


def preprocess_zju_dataset(dataset_path=ZJU_PATH):
    """
    Preprocess ZJU Eye-Blink dataset for blink detection training.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the ZJU Eye-Blink dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data splits
    """
    logger.info(f"Preprocessing ZJU Eye-Blink dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load annotation file (format may vary based on actual dataset structure)
        annotations_path = os.path.join(dataset_path, 'annotations.csv')
        if not os.path.exists(annotations_path):
            # Create a dummy annotation file for testing if not exists
            logger.warning(f"Annotation file not found. Creating dummy data for testing.")
            video_files = [f for f in os.listdir(dataset_path) if f.endswith('.avi') or f.endswith('.mp4')]
            if not video_files:
                logger.error("No video files found in the dataset.")
                return None
                
            # Create dummy annotations
            annotations = pd.DataFrame({
                'video_path': [os.path.join(dataset_path, f) for f in video_files],
                'blink_frames': [np.random.choice(range(100), size=5).tolist() for _ in video_files]
            })
        else:
            annotations = pd.read_csv(annotations_path)
        
        # Extract video paths and blink events
        video_paths = annotations['video_path'].tolist()
        blink_events = annotations['blink_frames'].tolist()
        
        # Split into training and validation sets
        train_videos, val_videos, train_blinks, val_blinks = train_test_split(
            video_paths, blink_events, test_size=0.2, random_state=42
        )
        
        logger.info(f"Successfully preprocessed ZJU dataset. "
                   f"Train samples: {len(train_videos)}, Val samples: {len(val_videos)}")
        
        return {
            'train': {'videos': train_videos, 'blinks': train_blinks},
            'val': {'videos': val_videos, 'blinks': val_blinks}
        }
    except Exception as e:
        logger.error(f"Error preprocessing ZJU dataset: {str(e)}")
        return None


def preprocess_drowsiness_dataset(dataset_path=DROWSINESS_PATH):
    """
    Preprocess Real-Life Drowsiness Dataset for fatigue detection.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the Drowsiness dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data splits for alert and drowsy states
    """
    logger.info(f"Preprocessing Real-Life Drowsiness dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load metadata file (format may vary based on actual dataset structure)
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata file for testing if not exists
            logger.warning(f"Metadata file not found. Creating dummy data for testing.")
            video_files = [f for f in os.listdir(dataset_path) if f.endswith('.avi') or f.endswith('.mp4')]
            if not video_files:
                logger.error("No video files found in the dataset.")
                return None
                
            # Create dummy metadata with random alert/drowsy labels
            states = np.random.choice(['alert', 'drowsy'], size=len(video_files))
            metadata = pd.DataFrame({
                'video_path': [os.path.join(dataset_path, f) for f in video_files],
                'state': states
            })
        else:
            metadata = pd.read_csv(metadata_path)
        
        # Extract fatigue-labeled segments
        alert_segments = metadata[metadata['state'] == 'alert']
        drowsy_segments = metadata[metadata['state'] == 'drowsy']
        
        # Create balanced dataset
        min_samples = min(len(alert_segments), len(drowsy_segments))
        balanced_alert = alert_segments.sample(min_samples, random_state=42) if len(alert_segments) >= min_samples else alert_segments
        balanced_drowsy = drowsy_segments.sample(min_samples, random_state=42) if len(drowsy_segments) >= min_samples else drowsy_segments
        
        # Split into training and validation
        train_alert, val_alert = train_test_split(balanced_alert, test_size=0.2)
        train_drowsy, val_drowsy = train_test_split(balanced_drowsy, test_size=0.2)
        
        logger.info(f"Successfully preprocessed Drowsiness dataset. "
                   f"Train alert: {len(train_alert)}, Train drowsy: {len(train_drowsy)}, "
                   f"Val alert: {len(val_alert)}, Val drowsy: {len(val_drowsy)}")
        
        return {
            'train': {
                'alert': train_alert['video_path'].tolist(),
                'drowsy': train_drowsy['video_path'].tolist()
            },
            'val': {
                'alert': val_alert['video_path'].tolist(),
                'drowsy': val_drowsy['video_path'].tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error preprocessing Drowsiness dataset: {str(e)}")
        return None


def preprocess_cew_dataset(dataset_path=CEW_PATH):
    """
    Preprocess CEW (Closed Eyes in the Wild) dataset for eye state classification.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the CEW dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data splits for open and closed eyes
    """
    logger.info(f"Preprocessing CEW dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Typically CEW dataset has closed and open eyes in separate folders
        closed_eyes_path = os.path.join(dataset_path, 'closed_eyes')
        open_eyes_path = os.path.join(dataset_path, 'open_eyes')
        
        if not all(os.path.exists(p) for p in [closed_eyes_path, open_eyes_path]):
            logger.warning("Standard CEW directory structure not found. Attempting to adapt.")
            
            # Try to find alternative structure or create dummy data for testing
            image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                logger.error("No image files found in the dataset.")
                return None
                
            # Create dummy labels (50% open, 50% closed)
            states = np.random.choice(['open', 'closed'], size=len(image_files))
            closed_files = [os.path.join(dataset_path, f) for f, s in zip(image_files, states) if s == 'closed']
            open_files = [os.path.join(dataset_path, f) for f, s in zip(image_files, states) if s == 'open']
        else:
            # Get files from the standard directory structure
            closed_files = [os.path.join(closed_eyes_path, f) for f in os.listdir(closed_eyes_path) 
                            if f.endswith('.jpg') or f.endswith('.png')]
            open_files = [os.path.join(open_eyes_path, f) for f in os.listdir(open_eyes_path) 
                          if f.endswith('.jpg') or f.endswith('.png')]
        
        # Split into training and validation sets
        train_closed, val_closed = train_test_split(closed_files, test_size=0.2, random_state=42)
        train_open, val_open = train_test_split(open_files, test_size=0.2, random_state=42)
        
        logger.info(f"Successfully preprocessed CEW dataset. "
                   f"Train closed: {len(train_closed)}, Train open: {len(train_open)}, "
                   f"Val closed: {len(val_closed)}, Val open: {len(val_open)}")
        
        return {
            'train': {
                'closed_eyes': train_closed,
                'open_eyes': train_open
            },
            'val': {
                'closed_eyes': val_closed,
                'open_eyes': val_open
            }
        }
    except Exception as e:
        logger.error(f"Error preprocessing CEW dataset: {str(e)}")
        return None


def preprocess_cafe_dataset(dataset_path=CAFE_PATH):
    """
    Preprocess CAFE (Columbia Affective Features) dataset for frustration detection.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the CAFE dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data for frustration-related emotions
    """
    logger.info(f"Preprocessing CAFE dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load metadata file (format may vary based on actual dataset structure)
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata file for testing if not exists
            logger.warning(f"Metadata file not found. Creating dummy data for testing.")
            image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                logger.error("No image files found in the dataset.")
                return None
                
            # Create dummy metadata with emotions
            emotions = np.random.choice(['neutral', 'anger', 'disgust', 'happy', 'sad'], size=len(image_files))
            metadata = pd.DataFrame({
                'image_path': [os.path.join(dataset_path, f) for f in image_files],
                'emotion': emotions
            })
        else:
            metadata = pd.read_csv(metadata_path)
        
        # For frustration detection, we focus on anger and disgust as proxy emotions
        frustration_proxies = metadata[metadata['emotion'].isin(['anger', 'disgust'])]
        neutral = metadata[metadata['emotion'] == 'neutral']
        
        # Balance the datasets
        min_samples = min(len(frustration_proxies), len(neutral))
        balanced_frustration = frustration_proxies.sample(min_samples, random_state=42) if len(frustration_proxies) >= min_samples else frustration_proxies
        balanced_neutral = neutral.sample(min_samples, random_state=42) if len(neutral) >= min_samples else neutral
        
        # Split into training and validation
        train_frust, val_frust = train_test_split(balanced_frustration, test_size=0.2)
        train_neutral, val_neutral = train_test_split(balanced_neutral, test_size=0.2)
        
        logger.info(f"Successfully preprocessed CAFE dataset. "
                   f"Train frustration: {len(train_frust)}, Train neutral: {len(train_neutral)}, "
                   f"Val frustration: {len(val_frust)}, Val neutral: {len(val_neutral)}")
        
        return {
            'train': {
                'frustration': train_frust['image_path'].tolist(),
                'neutral': train_neutral['image_path'].tolist()
            },
            'val': {
                'frustration': val_frust['image_path'].tolist(),
                'neutral': val_neutral['image_path'].tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error preprocessing CAFE dataset: {str(e)}")
        return None


def preprocess_affectnet(dataset_path=AFFECTNET_PATH):
    """
    Preprocess AffectNet dataset for enhanced emotion recognition including frustration.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the AffectNet dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data for frustration-related emotions
    """
    logger.info(f"Preprocessing AffectNet dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load metadata file (format may vary based on actual dataset structure)
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata file for testing if not exists
            logger.warning(f"Metadata file not found. Creating dummy data for testing.")
            image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                logger.error("No image files found in the dataset.")
                return None
                
            # Create dummy metadata with emotions
            emotions = np.random.choice(['neutral', 'anger', 'disgust', 'happy', 'sad', 'surprise', 'fear', 'contempt'], 
                                      size=len(image_files))
            metadata = pd.DataFrame({
                'image_path': [os.path.join(dataset_path, f) for f in image_files],
                'emotion': emotions
            })
        else:
            metadata = pd.read_csv(metadata_path)
        
        # Extract relevant emotion categories
        # For frustration, we'll use anger, disgust as proxies
        frustration_proxies = metadata[metadata['emotion'].isin(['anger', 'disgust'])]
        neutral = metadata[metadata['emotion'] == 'neutral']
        
        # Balance dataset
        min_samples = min(len(frustration_proxies), len(neutral))
        balanced_frustration = frustration_proxies.sample(min_samples, random_state=42) if len(frustration_proxies) >= min_samples else frustration_proxies
        balanced_neutral = neutral.sample(min_samples, random_state=42) if len(neutral) >= min_samples else neutral
        
        # Split into training and validation
        train_frust, val_frust = train_test_split(balanced_frustration, test_size=0.2)
        train_neutral, val_neutral = train_test_split(balanced_neutral, test_size=0.2)
        
        logger.info(f"Successfully preprocessed AffectNet dataset. "
                   f"Train frustration: {len(train_frust)}, Train neutral: {len(train_neutral)}, "
                   f"Val frustration: {len(val_frust)}, Val neutral: {len(val_neutral)}")
        
        return {
            'train': {
                'frustration': train_frust['image_path'].tolist(),
                'neutral': train_neutral['image_path'].tolist()
            },
            'val': {
                'frustration': val_frust['image_path'].tolist(),
                'neutral': val_neutral['image_path'].tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error preprocessing AffectNet dataset: {str(e)}")
        return None


def preprocess_eyegaze_dataset(dataset_path=EYEGAZE_PATH):
    """
    Preprocess Eye Gaze Net dataset for gaze direction estimation.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the Eye Gaze Net dataset
        
    Returns:
    --------
    dict
        Dictionary containing train and validation data with gaze annotations
    """
    logger.info(f"Preprocessing Eye Gaze Net dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load metadata file (format may vary based on actual dataset structure)
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata file for testing if not exists
            logger.warning(f"Metadata file not found. Creating dummy data for testing.")
            image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                logger.error("No image files found in the dataset.")
                return None
                
            # Create dummy gaze annotations (x, y coordinates)
            gaze_x = np.random.uniform(-1, 1, size=len(image_files))
            gaze_y = np.random.uniform(-1, 1, size=len(image_files))
            metadata = pd.DataFrame({
                'image_path': [os.path.join(dataset_path, f) for f in image_files],
                'gaze_x': gaze_x,
                'gaze_y': gaze_y
            })
        else:
            metadata = pd.read_csv(metadata_path)
        
        # Ensure necessary columns exist
        if not all(col in metadata.columns for col in ['image_path', 'gaze_x', 'gaze_y']):
            logger.error("Metadata file does not contain required gaze information.")
            return None
        
        # Split into training and validation
        train_data, val_data = train_test_split(metadata, test_size=0.2, random_state=42)
        
        logger.info(f"Successfully preprocessed Eye Gaze Net dataset. "
                   f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        return {
            'train': {
                'images': train_data['image_path'].tolist(),
                'gaze_x': train_data['gaze_x'].tolist(),
                'gaze_y': train_data['gaze_y'].tolist()
            },
            'val': {
                'images': val_data['image_path'].tolist(),
                'gaze_x': val_data['gaze_x'].tolist(),
                'gaze_y': val_data['gaze_y'].tolist()
            }
        }
    except Exception as e:
        logger.error(f"Error preprocessing Eye Gaze Net dataset: {str(e)}")
        return None


def preprocess_programmer_productivity(dataset_path=PROGRAMMER_PATH):
    """
    Preprocess Programmer Productivity dataset for cognitive state/productivity correlations.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the Programmer Productivity dataset
        
    Returns:
    --------
    dict
        Dictionary containing programmer session data with productivity metrics
    """
    logger.info(f"Preprocessing Programmer Productivity dataset at {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist")
        return None
        
    try:
        # Load metadata file (format may vary based on actual dataset structure)
        metadata_path = os.path.join(dataset_path, 'session_data.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata file for testing if not exists
            logger.warning(f"Session data file not found. Creating dummy data for testing.")
            
            # Create dummy session data
            n_sessions = 50
            session_ids = [f"session_{i}" for i in range(n_sessions)]
            productivity_scores = np.random.uniform(0, 10, size=n_sessions)
            cognitive_states = np.random.choice(['neutral', 'fatigue', 'frustration'], size=n_sessions)
            code_quality = np.random.uniform(0, 10, size=n_sessions)
            
            metadata = pd.DataFrame({
                'session_id': session_ids,
                'productivity_score': productivity_scores,
                'cognitive_state': cognitive_states,
                'code_quality': code_quality
            })
        else:
            metadata = pd.read_csv(metadata_path)
        
        # Split into training and validation
        train_data, val_data = train_test_split(metadata, test_size=0.2, random_state=42)
        
        logger.info(f"Successfully preprocessed Programmer Productivity dataset. "
                   f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Convert to dictionary format
        return {
            'train': train_data.to_dict('records'),
            'val': val_data.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Error preprocessing Programmer Productivity dataset: {str(e)}")
        return None


def verify_dataset_availability():
    """
    Check if all required datasets are available at their specified paths.
    
    Returns:
    --------
    dict
        Dictionary with dataset names as keys and availability status as values
    """
    datasets = {
        'ZJU Eye-Blink': ZJU_PATH,
        'Real-Life Drowsiness': DROWSINESS_PATH,
        'CEW': CEW_PATH,
        'CAFE': CAFE_PATH,
        'AffectNet': AFFECTNET_PATH,
        'Eye Gaze Net': EYEGAZE_PATH,
        'Programmer Productivity': PROGRAMMER_PATH
    }
    
    availability = {}
    for name, path in datasets.items():
        status = os.path.exists(path)
        availability[name] = status
        if status:
            logger.info(f"Dataset {name} is available at {path}")
        else:
            logger.warning(f"Dataset {name} is NOT available at {path}")
    
    return availability
