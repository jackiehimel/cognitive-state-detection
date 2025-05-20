"""
Module for adaptive eye tracking calibration, inspired by WebGazer.js.
Provides continuous calibration during long coding sessions.
"""

import numpy as np
import cv2
import time
import logging
import json
import os
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptiveCalibration:
    """
    Implements WebGazer.js-style continuous calibration for eye tracking.
    Adapts to user's eye characteristics over time to improve accuracy.
    """
    
    def __init__(
        self,
        calibration_file=None,
        buffer_size=30,
        calibration_points=9,
        learning_rate=0.1,
        recalibration_interval=300,  # 5 minutes in seconds
        min_calibrations_needed=5,
    ):
        """
        Initialize the adaptive calibration system.
        
        Parameters:
        -----------
        calibration_file : str, optional
            Path to save/load calibration data
        buffer_size : int
            Number of recent observations to keep for each calibration point
        calibration_points : int
            Number of calibration points (typically 9 for a 3x3 grid)
        learning_rate : float
            Rate at which the system adapts to new observations (0-1)
        recalibration_interval : int
            Time in seconds between suggested recalibrations
        min_calibrations_needed : int
            Minimum number of calibration points needed for accurate gaze estimation
        """
        # Calibration parameters
        self.buffer_size = buffer_size
        self.calibration_points = calibration_points
        self.learning_rate = learning_rate
        self.recalibration_interval = recalibration_interval
        self.min_calibrations_needed = min_calibrations_needed
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_file = calibration_file
        self.last_recalibration_time = time.time()
        
        # Calibration data
        # Maps screen coordinates to eye features
        self.calibration_data = {}
        
        # Screen dimensions will be set during calibration
        self.screen_width = None
        self.screen_height = None
        
        # Regression model parameters for gaze prediction
        # Will map eye features to screen coordinates
        self.gaze_model = {
            'weights': None,
            'bias': None,
            'error': float('inf')
        }
        
        # Calibration point history for adapting over time
        # Each entry is ((x, y), eye_features)
        self.observation_history = deque(maxlen=100)
        
        # Load existing calibration if available
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration()
        
        logger.info("AdaptiveCalibration initialized")
    
    def get_calibration_points(self, width, height):
        """
        Generate calibration point coordinates for a 3x3 grid.
        
        Parameters:
        -----------
        width : int
            Screen width
        height : int
            Screen height
            
        Returns:
        --------
        list
            List of (x, y) coordinates for calibration points
        """
        self.screen_width = width
        self.screen_height = height
        
        # Generate 3x3 grid of points
        x_coords = [width * 0.1, width * 0.5, width * 0.9]
        y_coords = [height * 0.1, height * 0.5, height * 0.9]
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append((int(x), int(y)))
        
        return points
    
    def add_calibration_point(self, screen_pos, eye_features):
        """
        Add a calibration point to the system.
        
        Parameters:
        -----------
        screen_pos : tuple
            (x, y) coordinates on screen
        eye_features : dict
            Dictionary of eye features extracted from the frame
            
        Returns:
        --------
        bool
            True if calibration point was added successfully
        """
        if not eye_features:
            logger.warning("No eye features detected for calibration point")
            return False
        
        # Convert eye features to a vector
        feature_vector = self._extract_feature_vector(eye_features)
        if feature_vector is None:
            return False
            
        # Add to calibration data
        pos_key = f"{screen_pos[0]}_{screen_pos[1]}"
        if pos_key not in self.calibration_data:
            self.calibration_data[pos_key] = []
            
        # Add new observation
        self.calibration_data[pos_key].append(feature_vector)
        
        # Keep only the most recent observations
        if len(self.calibration_data[pos_key]) > self.buffer_size:
            self.calibration_data[pos_key] = self.calibration_data[pos_key][-self.buffer_size:]
            
        # Add to history for continuous adaptation
        self.observation_history.append((screen_pos, feature_vector))
        
        # Update calibration status
        self._update_calibration_status()
        
        # Save calibration data if file is specified
        if self.calibration_file:
            self.save_calibration()
            
        # Update calibration model
        self._update_gaze_model()
            
        return True
    
    def predict_gaze(self, eye_features):
        """
        Predict gaze coordinates based on eye features.
        
        Parameters:
        -----------
        eye_features : dict
            Dictionary of eye features extracted from the frame
            
        Returns:
        --------
        tuple, float
            (x, y) predicted gaze coordinates and confidence score
        """
        if not self.is_calibrated:
            logger.warning("System not calibrated for gaze prediction")
            return None, 0.0
            
        # Extract feature vector
        feature_vector = self._extract_feature_vector(eye_features)
        if feature_vector is None:
            return None, 0.0
            
        # Apply gaze prediction model
        if self.gaze_model['weights'] is not None:
            # Add bias term
            feature_vector_with_bias = np.concatenate([[1], feature_vector])
            
            # Predict x, y coordinates
            gaze_x = np.dot(self.gaze_model['weights'][0], feature_vector_with_bias)
            gaze_y = np.dot(self.gaze_model['weights'][1], feature_vector_with_bias)
            
            # Clamp to screen dimensions
            gaze_x = max(0, min(self.screen_width, gaze_x))
            gaze_y = max(0, min(self.screen_height, gaze_y))
            
            # Calculate confidence based on model error
            # Lower error = higher confidence
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + self.gaze_model['error'])))
            
            return (int(gaze_x), int(gaze_y)), confidence
            
        return None, 0.0
    
    def adaptive_update(self, screen_pos, eye_features):
        """
        Update calibration based on user interaction points.
        Used for continuous calibration during normal use.
        
        Parameters:
        -----------
        screen_pos : tuple
            (x, y) coordinates on screen where user interacted
        eye_features : dict
            Dictionary of eye features extracted from the frame
            
        Returns:
        --------
        bool
            True if update was successful
        """
        # Only update if we have valid eye features
        if not eye_features:
            return False
            
        # Extract feature vector
        feature_vector = self._extract_feature_vector(eye_features)
        if feature_vector is None:
            return False
            
        # Add to observation history with lower weight (continuous calibration)
        self.observation_history.append((screen_pos, feature_vector))
        
        # Update gaze model incrementally
        self._update_gaze_model_incremental(screen_pos, feature_vector)
        
        # Track time since last explicit recalibration
        current_time = time.time()
        if current_time - self.last_recalibration_time > self.recalibration_interval:
            # It's time to suggest recalibration
            logger.info("Suggesting recalibration after %d seconds of continuous use",
                       self.recalibration_interval)
            # Reset timer even if user doesn't recalibrate immediately
            self.last_recalibration_time = current_time
            
        return True
    
    def needs_calibration(self):
        """
        Check if the system needs calibration.
        
        Returns:
        --------
        bool
            True if calibration is needed
        """
        # Not calibrated at all
        if not self.is_calibrated:
            return True
            
        # Check if it's time for a suggested recalibration
        if time.time() - self.last_recalibration_time > self.recalibration_interval:
            return True
            
        return False
    
    def reset_calibration(self):
        """Reset all calibration data and start fresh."""
        self.calibration_data = {}
        self.observation_history.clear()
        self.is_calibrated = False
        self.gaze_model = {
            'weights': None,
            'bias': None,
            'error': float('inf')
        }
        self.last_recalibration_time = time.time()
        logger.info("Calibration reset")
    
    def save_calibration(self):
        """Save calibration data to file."""
        if not self.calibration_file:
            logger.warning("No calibration file specified, cannot save")
            return False
            
        try:
            # Convert calibration data to serializable format
            serializable_data = {
                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'is_calibrated': self.is_calibrated,
                'last_recalibration_time': self.last_recalibration_time,
                'gaze_model': {
                    'weights': self.gaze_model['weights'].tolist() if self.gaze_model['weights'] is not None else None,
                    'bias': self.gaze_model['bias'] if self.gaze_model['bias'] is not None else None,
                    'error': float(self.gaze_model['error'])
                },
                'calibration_data': {k: [v.tolist() for v in vlist] for k, vlist in self.calibration_data.items()}
            }
            
            # Save to file
            with open(self.calibration_file, 'w') as f:
                json.dump(serializable_data, f)
                
            logger.info(f"Calibration saved to {self.calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {str(e)}")
            return False
    
    def load_calibration(self):
        """Load calibration data from file."""
        if not self.calibration_file or not os.path.exists(self.calibration_file):
            logger.warning("Calibration file not found, starting fresh")
            return False
            
        try:
            # Load from file
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
                
            # Restore calibration state
            self.screen_width = data['screen_width']
            self.screen_height = data['screen_height']
            self.is_calibrated = data['is_calibrated']
            self.last_recalibration_time = data['last_recalibration_time']
            
            # Restore gaze model
            if data['gaze_model']['weights'] is not None:
                self.gaze_model['weights'] = np.array(data['gaze_model']['weights'])
                self.gaze_model['bias'] = data['gaze_model']['bias']
                self.gaze_model['error'] = data['gaze_model']['error']
                
            # Restore calibration data
            self.calibration_data = {k: [np.array(v) for v in vlist] for k, vlist in data['calibration_data'].items()}
            
            logger.info(f"Calibration loaded from {self.calibration_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")
            self.reset_calibration()
            return False
    
    def _extract_feature_vector(self, eye_features):
        """
        Extract a feature vector from the eye features dictionary.
        
        Parameters:
        -----------
        eye_features : dict
            Dictionary of eye features
            
        Returns:
        --------
        numpy.ndarray
            Feature vector for gaze prediction
        """
        try:
            # Extract relevant features for gaze prediction
            # These features have been found to correlate with gaze direction
            features = [
                eye_features.get('eye_aspect_ratio', 0),
                eye_features.get('gaze_direction_x', 0),
                eye_features.get('gaze_direction_y', 0),
                eye_features.get('pupil_size', 0)
            ]
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting feature vector: {str(e)}")
            return None
    
    def _update_calibration_status(self):
        """Update the calibration status based on available data."""
        # Count calibration points with enough data
        valid_points = sum(1 for points in self.calibration_data.values() if len(points) >= 3)
        
        # Set calibration status
        self.is_calibrated = valid_points >= self.min_calibrations_needed
        
        if self.is_calibrated:
            logger.info(f"System calibrated with {valid_points} valid calibration points")
            self.last_recalibration_time = time.time()
    
    def _update_gaze_model(self):
        """Update the gaze prediction model based on all calibration data."""
        if not self.calibration_data:
            return
            
        try:
            # Collect all calibration data points
            X = []  # Feature vectors
            Y = []  # Screen coordinates
            
            for pos_key, feature_vectors in self.calibration_data.items():
                if not feature_vectors:
                    continue
                    
                # Parse screen coordinates from key
                x, y = map(int, pos_key.split('_'))
                
                # Add each feature vector with corresponding screen coordinates
                for feature_vector in feature_vectors:
                    X.append(feature_vector)
                    Y.append([x, y])
            
            if not X:
                return
                
            # Convert to numpy arrays
            X = np.array(X)
            Y = np.array(Y)
            
            # Add bias term
            X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
            
            # Fit linear regression model for x and y coordinates separately
            # Using pseudo-inverse method for linear regression
            weights_x = np.linalg.lstsq(X_with_bias, Y[:, 0], rcond=None)[0]
            weights_y = np.linalg.lstsq(X_with_bias, Y[:, 1], rcond=None)[0]
            
            # Store weights
            self.gaze_model['weights'] = np.vstack((weights_x, weights_y))
            self.gaze_model['bias'] = np.array([weights_x[0], weights_y[0]])
            
            # Calculate prediction error
            y_pred_x = np.dot(X_with_bias, weights_x)
            y_pred_y = np.dot(X_with_bias, weights_y)
            
            error_x = np.mean(np.abs(Y[:, 0] - y_pred_x))
            error_y = np.mean(np.abs(Y[:, 1] - y_pred_y))
            
            # Average error in pixels
            self.gaze_model['error'] = (error_x + error_y) / 2
            
            logger.info(f"Gaze model updated. Avg error: {self.gaze_model['error']:.2f} pixels")
            
        except Exception as e:
            logger.error(f"Error updating gaze model: {str(e)}")
    
    def _update_gaze_model_incremental(self, screen_pos, feature_vector):
        """
        Update the gaze model incrementally for continuous calibration.
        
        Parameters:
        -----------
        screen_pos : tuple
            (x, y) screen coordinates
        feature_vector : numpy.ndarray
            Extracted eye features
        """
        if not self.is_calibrated or self.gaze_model['weights'] is None:
            return
            
        try:
            # Add bias term
            feature_vector_with_bias = np.concatenate([[1], feature_vector])
            
            # Current prediction
            pred_x = np.dot(self.gaze_model['weights'][0], feature_vector_with_bias)
            pred_y = np.dot(self.gaze_model['weights'][1], feature_vector_with_bias)
            
            # Error
            error_x = screen_pos[0] - pred_x
            error_y = screen_pos[1] - pred_y
            
            # Update weights using learning rate
            self.gaze_model['weights'][0] += self.learning_rate * error_x * feature_vector_with_bias
            self.gaze_model['weights'][1] += self.learning_rate * error_y * feature_vector_with_bias
            
            # Update bias
            self.gaze_model['bias'][0] = self.gaze_model['weights'][0][0]
            self.gaze_model['bias'][1] = self.gaze_model['weights'][1][0]
            
            # Gradually reduce the learning rate over time
            # This makes the model more stable as it accumulates more data
            self.learning_rate = max(0.01, self.learning_rate * 0.999)
            
        except Exception as e:
            logger.error(f"Error in incremental update: {str(e)}")


class CalibrationUI:
    """
    Provides a user interface for eye tracker calibration.
    """
    
    def __init__(self, calibration, window_name="Eye Tracker Calibration"):
        """
        Initialize the calibration UI.
        
        Parameters:
        -----------
        calibration : AdaptiveCalibration
            Calibration system to use
        window_name : str
            Name of the calibration window
        """
        self.calibration = calibration
        self.window_name = window_name
        self.frame = None
        self.target_pos = None
        self.current_point_index = 0
        self.point_duration = 2.0  # seconds to focus on each point
        self.point_start_time = None
        self.calibration_complete = False
    
    def start_calibration(self, width=1280, height=720):
        """
        Start the calibration process.
        
        Parameters:
        -----------
        width : int
            Screen width
        height : int
            Screen height
            
        Returns:
        --------
        bool
            True if calibration was successfully completed
        """
        # Get calibration points
        self.calibration_points = self.calibration.get_calibration_points(width, height)
        self.current_point_index = 0
        self.calibration_complete = False
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
        
        # Start with the first point
        self.target_pos = self.calibration_points[0]
        self.point_start_time = time.time()
        
        return True
    
    def update_frame(self, frame, eye_features=None):
        """
        Update the calibration UI with a new frame.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Video frame
        eye_features : dict, optional
            Extracted eye features
            
        Returns:
        --------
        bool
            True if calibration is still in progress, False if complete
        """
        if self.calibration_complete:
            return False
            
        # Make a copy of the frame
        self.frame = frame.copy()
        height, width = self.frame.shape[:2]
        
        # Draw calibration target
        if self.target_pos:
            cv2.circle(self.frame, self.target_pos, 15, (0, 255, 0), -1)
            cv2.circle(self.frame, self.target_pos, 25, (0, 255, 0), 2)
            
        # Add progress text
        progress_text = f"Calibration point {self.current_point_index + 1}/{len(self.calibration_points)}"
        cv2.putText(self.frame, progress_text, (width // 2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        instructions = "Focus on the green circle"
        cv2.putText(self.frame, instructions, (width // 2 - 150, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow(self.window_name, self.frame)
        
        # Process calibration logic
        if self.target_pos and eye_features:
            current_time = time.time()
            elapsed = current_time - self.point_start_time
            
            # When point duration is reached, add calibration point and move to next
            if elapsed >= self.point_duration:
                # Add calibration point
                self.calibration.add_calibration_point(self.target_pos, eye_features)
                
                # Move to next point
                self.current_point_index += 1
                
                # If all points are done, finish calibration
                if self.current_point_index >= len(self.calibration_points):
                    self._finish_calibration()
                    return False
                    
                # Otherwise, move to next point
                self.target_pos = self.calibration_points[self.current_point_index]
                self.point_start_time = current_time
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key to exit
        if key == 27:
            self._finish_calibration()
            return False
        
        return True
    
    def _finish_calibration(self):
        """Clean up and finish calibration."""
        cv2.destroyWindow(self.window_name)
        self.calibration_complete = True
        
        # Update the model
        self.calibration._update_gaze_model()
        
        # Save calibration
        if self.calibration.calibration_file:
            self.calibration.save_calibration()
            
        logger.info("Calibration sequence complete")


class ContinuousCalibrationTracker:
    """
    Tracks user interaction points for continuous calibration during coding sessions.
    Inspired by WebGazer.js approach to continuous calibration.
    """
    
    def __init__(self, calibration, min_distance=100):
        """
        Initialize the continuous calibration tracker.
        
        Parameters:
        -----------
        calibration : AdaptiveCalibration
            Calibration system to use
        min_distance : int
            Minimum pixel distance between interaction points to consider them separate
        """
        self.calibration = calibration
        self.min_distance = min_distance
        self.last_point = None
        self.last_update_time = None
        self.min_time_between_updates =.5  # seconds
        
        logger.info("ContinuousCalibrationTracker initialized")
    
    def track_mouse_click(self, pos, eye_features):
        """
        Track a mouse click event for calibration.
        
        Parameters:
        -----------
        pos : tuple
            (x, y) screen coordinates of mouse click
        eye_features : dict
            Extracted eye features at the time of click
            
        Returns:
        --------
        bool
            True if calibration was updated
        """
        current_time = time.time()
        
        # Skip if not enough time has passed since last update
        if self.last_update_time and (current_time - self.last_update_time) < self.min_time_between_updates:
            return False
            
        # Skip if this point is too close to the last one
        if self.last_point and np.sqrt((pos[0] - self.last_point[0])**2 + (pos[1] - self.last_point[1])**2) < self.min_distance:
            return False
            
        # Update calibration
        if self.calibration.adaptive_update(pos, eye_features):
            self.last_point = pos
            self.last_update_time = current_time
            return True
            
        return False
    
    def track_keyboard_input(self, cursor_pos, eye_features):
        """
        Track keyboard input for calibration.
        
        Parameters:
        -----------
        cursor_pos : tuple
            (x, y) screen coordinates of text cursor
        eye_features : dict
            Extracted eye features at the time of keyboard input
            
        Returns:
        --------
        bool
            True if calibration was updated
        """
        # Use the same logic as mouse clicks
        return self.track_mouse_click(cursor_pos, eye_features)
    
    def reset(self):
        """Reset tracking state."""
        self.last_point = None
        self.last_update_time = None
