"""
Module for extracting eye-related features from videos and images for cognitive state detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices in MediaPipe Face Mesh
# Left eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Pupil landmarks
LEFT_PUPIL = [473, 474, 475, 476, 477]
RIGHT_PUPIL = [468, 469, 470, 471, 472]


class EyeFeatureExtractor:
    """
    Class for extracting eye-related features from video and image data.
    """
    
    def __init__(self, 
                 blink_threshold=0.3, 
                 min_blink_frames=3, 
                 temporal_window=30,
                 perclos_window=180):  # Default 6 seconds at 30 fps
        """
        Initialize the eye feature extractor.
        
        Parameters:
        -----------
        blink_threshold : float
            Threshold for eye aspect ratio to consider the eye as closed
        min_blink_frames : int
            Minimum number of consecutive frames with closed eyes to count as a blink
        temporal_window : int
            Number of frames to consider for temporal features
        perclos_window : int
            Number of frames to consider for PERCLOS calculation
        """
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.blink_threshold = blink_threshold
        self.min_blink_frames = min_blink_frames
        self.temporal_window = temporal_window
        self.perclos_window = perclos_window
        
        # Initialize buffers for temporal feature calculation
        self.eye_aspect_ratio_buffer = deque(maxlen=perclos_window)
        self.pupil_size_buffer = deque(maxlen=temporal_window)
        self.gaze_direction_buffer = deque(maxlen=temporal_window)
        self.blink_buffer = deque(maxlen=temporal_window)
        
        # Blink detection state
        self.blink_counter = 0
        self.blink_total = 0
        self.is_blinking = False
        
        logger.info("EyeFeatureExtractor initialized")
        
    def _calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate the eye aspect ratio (EAR) which is a measure of eye openness.
        
        Parameters:
        -----------
        eye_landmarks : array-like
            6 landmarks defining the eye
            
        Returns:
        --------
        float
            Eye aspect ratio (lower values indicate more closed eyes)
        """
        # Compute the euclidean distances between the horizontal landmarks
        h1 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        h2 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        h3 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Compute the euclidean distances between the vertical landmarks
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[4])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[5])
        
        # Compute the eye aspect ratio
        ear = (v1 + v2) / (2.0 * (h1 + h2 + h3))
        
        return ear
    
    def _calculate_pupil_size(self, pupil_landmarks):
        """
        Calculate the pupil size based on pupil landmarks.
        
        Parameters:
        -----------
        pupil_landmarks : array-like
            Landmarks defining the pupil area
            
        Returns:
        --------
        float
            Estimated pupil size
        """
        # Calculate the center of the pupil
        center = np.mean(pupil_landmarks, axis=0)
        
        # Calculate the average distance from center to each landmark
        distances = [np.linalg.norm(point - center) for point in pupil_landmarks]
        avg_radius = np.mean(distances)
        
        # Pupil size is approximated as the area of the circle
        pupil_size = np.pi * (avg_radius ** 2)
        
        return pupil_size
    
    def _calculate_gaze_direction(self, eye_landmarks, pupil_landmarks):
        """
        Calculate gaze direction based on pupil position relative to eye contour.
        
        Parameters:
        -----------
        eye_landmarks : array-like
            6 landmarks defining the eye contour
        pupil_landmarks : array-like
            Landmarks defining the pupil
            
        Returns:
        --------
        tuple
            (x, y) normalized gaze direction relative to eye center
        """
        # Calculate eye center
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Calculate pupil center
        pupil_center = np.mean(pupil_landmarks, axis=0)
        
        # Calculate the horizontal and vertical distances
        # Normalize by eye width and height
        eye_width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        eye_height = np.linalg.norm(eye_landmarks[1] - eye_landmarks[4])
        
        if eye_width == 0 or eye_height == 0:
            return 0.0, 0.0
        
        gaze_x = (pupil_center[0] - eye_center[0]) / (eye_width / 2)
        gaze_y = (pupil_center[1] - eye_center[1]) / (eye_height / 2)
        
        # Clamp values to the range [-1, 1]
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))
        
        return gaze_x, gaze_y
    
    def _update_blink_detection(self, ear):
        """
        Update blink detection state based on the current eye aspect ratio.
        
        Parameters:
        -----------
        ear : float
            Current eye aspect ratio
            
        Returns:
        --------
        bool
            True if a blink is detected in this frame, False otherwise
        """
        # Check if eye is closed based on the EAR threshold
        is_closed = ear < self.blink_threshold
        blink_detected = False
        
        if is_closed and not self.is_blinking:
            # Start of a potential blink
            self.blink_counter = 1
            self.is_blinking = True
        elif is_closed and self.is_blinking:
            # Continuation of a potential blink
            self.blink_counter += 1
        elif not is_closed and self.is_blinking:
            # End of a potential blink
            if self.blink_counter >= self.min_blink_frames:
                # Valid blink detected
                self.blink_total += 1
                blink_detected = True
            
            # Reset blink state
            self.blink_counter = 0
            self.is_blinking = False
        
        # Add to blink buffer (1 for blink, 0 for no blink)
        self.blink_buffer.append(1 if blink_detected else 0)
        
        return blink_detected
    
    def _calculate_perclos(self):
        """
        Calculate PERCLOS (percentage of eye closure over time).
        
        Returns:
        --------
        float
            PERCLOS value (0.0 to 1.0)
        """
        if not self.eye_aspect_ratio_buffer:
            return 0.0
            
        # Count frames where eye is considered closed
        closed_frames = sum(1 for ear in self.eye_aspect_ratio_buffer if ear < self.blink_threshold)
        
        # Calculate percentage
        perclos = closed_frames / len(self.eye_aspect_ratio_buffer)
        
        return perclos
    
    def _calculate_blink_rate(self):
        """
        Calculate blink rate (blinks per minute) based on the blink buffer.
        
        Returns:
        --------
        float
            Blink rate in blinks per minute
        """
        if not self.blink_buffer:
            return 0.0
            
        # Count blinks in the buffer
        blink_count = sum(self.blink_buffer)
        
        # Calculate blinks per minute (assuming 30 fps)
        fps = 30
        seconds = len(self.blink_buffer) / fps
        minutes = seconds / 60
        
        if minutes == 0:
            return 0.0
            
        blink_rate = blink_count / minutes
        
        return blink_rate
    
    def process_frame(self, frame):
        """
        Process a single frame to extract eye-related features.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input video frame
            
        Returns:
        --------
        dict
            Dictionary containing extracted features, or None if no face is detected
        """
        if frame is None:
            logger.warning("Received empty frame")
            return None
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            logger.debug("No face detected in the frame")
            return None
            
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract eye landmarks
        left_eye = np.array([(face_landmarks[idx].x * frame.shape[1], 
                             face_landmarks[idx].y * frame.shape[0]) 
                            for idx in LEFT_EYE_INDICES])
        
        right_eye = np.array([(face_landmarks[idx].x * frame.shape[1], 
                              face_landmarks[idx].y * frame.shape[0]) 
                             for idx in RIGHT_EYE_INDICES])
                             
        left_pupil = np.array([(face_landmarks[idx].x * frame.shape[1], 
                               face_landmarks[idx].y * frame.shape[0]) 
                              for idx in LEFT_PUPIL])
                              
        right_pupil = np.array([(face_landmarks[idx].x * frame.shape[1], 
                                face_landmarks[idx].y * frame.shape[0]) 
                               for idx in RIGHT_PUPIL])
        
        # Calculate eye aspect ratio for both eyes
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)
        
        # Average EAR from both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Update eye aspect ratio buffer
        self.eye_aspect_ratio_buffer.append(ear)
        
        # Detect blinks
        blink_detected = self._update_blink_detection(ear)
        
        # Calculate pupil size
        left_pupil_size = self._calculate_pupil_size(left_pupil)
        right_pupil_size = self._calculate_pupil_size(right_pupil)
        pupil_size = (left_pupil_size + right_pupil_size) / 2.0
        
        # Update pupil size buffer
        self.pupil_size_buffer.append(pupil_size)
        
        # Calculate gaze direction
        left_gaze = self._calculate_gaze_direction(left_eye, left_pupil)
        right_gaze = self._calculate_gaze_direction(right_eye, right_pupil)
        
        # Average gaze direction from both eyes
        gaze_x = (left_gaze[0] + right_gaze[0]) / 2.0
        gaze_y = (left_gaze[1] + right_gaze[1]) / 2.0
        
        # Update gaze direction buffer
        self.gaze_direction_buffer.append((gaze_x, gaze_y))
        
        # Calculate PERCLOS
        perclos = self._calculate_perclos()
        
        # Calculate blink rate
        blink_rate = self._calculate_blink_rate()
        
        # Return extracted features
        features = {
            'eye_aspect_ratio': ear,
            'blink_detected': blink_detected,
            'perclos': perclos,
            'blink_rate': blink_rate,
            'pupil_size': pupil_size,
            'gaze_direction_x': gaze_x,
            'gaze_direction_y': gaze_y
        }
        
        return features
    
    def extract_features_from_video(self, video_path):
        """
        Extract eye-related features from a video file.
        
        Parameters:
        -----------
        video_path : str
            Path to the video file
            
        Returns:
        --------
        dict
            Dictionary containing temporal features extracted from the video
        """
        logger.info(f"Extracting features from video: {video_path}")
        
        # Reset buffers to ensure clean state
        self.eye_aspect_ratio_buffer.clear()
        self.pupil_size_buffer.clear()
        self.gaze_direction_buffer.clear()
        self.blink_buffer.clear()
        self.blink_counter = 0
        self.blink_total = 0
        self.is_blinking = False
        
        # Open video
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        video = cv2.VideoCapture(video_path)
        
        # Process each frame
        frame_features = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            # Extract features from frame
            features = self.process_frame(frame)
            if features:
                frame_features.append(features)
        
        video.release()
        
        # If no frames were processed successfully, return None
        if not frame_features:
            logger.warning(f"No features extracted from video: {video_path}")
            return None
            
        # Calculate temporal features from accumulated frame features
        temporal_features = self.calculate_temporal_features(frame_features)
        
        return temporal_features
    
    def extract_features_from_image(self, image_path):
        """
        Extract eye-related features from a single image.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        dict
            Dictionary containing features extracted from the image
        """
        logger.info(f"Extracting features from image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Extract features from image
        features = self.process_frame(image)
        
        return features
    
    def calculate_temporal_features(self, frame_features):
        """
        Calculate temporal features from a sequence of frame features.
        
        Parameters:
        -----------
        frame_features : list
            List of feature dictionaries from individual frames
            
        Returns:
        --------
        dict
            Dictionary containing temporal features
        """
        if not frame_features:
            logger.warning("No frame features to calculate temporal features from")
            return None
            
        # Extract feature sequences
        ear_values = [f['eye_aspect_ratio'] for f in frame_features if 'eye_aspect_ratio' in f]
        pupil_sizes = [f['pupil_size'] for f in frame_features if 'pupil_size' in f]
        gaze_x_values = [f['gaze_direction_x'] for f in frame_features if 'gaze_direction_x' in f]
        gaze_y_values = [f['gaze_direction_y'] for f in frame_features if 'gaze_direction_y' in f]
        
        # Skip calculation if any feature lists are empty
        if not all([ear_values, pupil_sizes, gaze_x_values, gaze_y_values]):
            logger.warning("Some feature lists are empty, cannot calculate temporal features")
            return None
            
        # Calculate statistics
        temporal_features = {
            # EAR (Eye Aspect Ratio) statistics
            'ear_mean': np.mean(ear_values),
            'ear_std': np.std(ear_values),
            'ear_min': np.min(ear_values),
            'ear_max': np.max(ear_values),
            
            # PERCLOS (calculated from EAR values)
            'perclos': sum(1 for ear in ear_values if ear < self.blink_threshold) / len(ear_values),
            
            # Blink statistics
            'blink_rate': self.blink_total / (len(frame_features) / 30 / 60),  # Blinks per minute (assuming 30 fps)
            'blink_duration': self.blink_counter / 30 if self.is_blinking else 0,  # Current blink duration in seconds
            
            # Pupil size statistics
            'pupil_size_mean': np.mean(pupil_sizes),
            'pupil_size_std': np.std(pupil_sizes),
            'pupil_size_min': np.min(pupil_sizes),
            'pupil_size_max': np.max(pupil_sizes),
            
            # Gaze direction statistics
            'gaze_x_mean': np.mean(gaze_x_values),
            'gaze_x_std': np.std(gaze_x_values),
            'gaze_y_mean': np.mean(gaze_y_values),
            'gaze_y_std': np.std(gaze_y_values),
            
            # Gaze distribution (how much the gaze moves around)
            'gaze_dispersion': np.mean([np.sqrt(x**2 + y**2) for x, y in zip(gaze_x_values, gaze_y_values)]),
            
            # Number of frames analyzed
            'frame_count': len(frame_features)
        }
        
        logger.info(f"Calculated temporal features from {len(frame_features)} frames")
        
        return temporal_features
    
    def visualize_landmarks(self, frame):
        """
        Visualize face landmarks on the frame for debugging.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input video frame
            
        Returns:
        --------
        numpy.ndarray
            Frame with landmarks visualized
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        # Convert back to BGR for OpenCV display
        output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
                
                # Draw eye contours
                for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * output_frame.shape[1])
                    y = int(landmark.y * output_frame.shape[0])
                    cv2.circle(output_frame, (x, y), 2, (0, 0, 255), -1)
                
                # Draw pupil landmarks
                for idx in LEFT_PUPIL + RIGHT_PUPIL:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * output_frame.shape[1])
                    y = int(landmark.y * output_frame.shape[0])
                    cv2.circle(output_frame, (x, y), 2, (255, 0, 0), -1)
        
        return output_frame


def extract_features_from_datasets(datasets, extractor=None):
    """
    Extract eye-related features from all datasets.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing preprocessed datasets
    extractor : EyeFeatureExtractor, optional
        Feature extractor to use, if None a new one will be created
        
    Returns:
    --------
    dict
        Dictionary containing extracted features
    """
    logger.info("Extracting features from datasets")
    
    # Initialize feature storage
    features = {
        'blink_patterns': [],
        'eye_closure_patterns': [],
        'pupil_metrics': [],
        'gaze_patterns': [],
        'labels': []
    }
    
    if extractor is None:
        extractor = EyeFeatureExtractor()
    
    # Process ZJU Eye-Blink Dataset (if available)
    if 'zju' in datasets:
        logger.info("Processing ZJU Eye-Blink dataset")
        zju_data = datasets['zju']
        
        for video_path in zju_data['train']['videos']:
            # Extract features from video
            temporal_features = extractor.extract_features_from_video(video_path)
            
            if temporal_features:
                features['blink_patterns'].append(temporal_features)
                # Label as normal blink pattern
                features['labels'].append('neutral')
    
    # Process Real-Life Drowsiness Dataset (if available)
    if 'drowsiness' in datasets:
        logger.info("Processing Real-Life Drowsiness dataset")
        drowsy_data = datasets['drowsiness']
        
        # Process alert videos
        for video_path in drowsy_data['train']['alert']:
            # Extract features from video
            temporal_features = extractor.extract_features_from_video(video_path)
            
            if temporal_features:
                features['eye_closure_patterns'].append(temporal_features)
                features['labels'].append('alert')
        
        # Process drowsy videos
        for video_path in drowsy_data['train']['drowsy']:
            # Extract features from video
            temporal_features = extractor.extract_features_from_video(video_path)
            
            if temporal_features:
                features['eye_closure_patterns'].append(temporal_features)
                features['labels'].append('fatigue')
    
    # Process AffectNet for frustration (if available)
    if 'affectnet' in datasets:
        logger.info("Processing AffectNet dataset")
        affect_data = datasets['affectnet']
        
        for image_path in affect_data['train']['frustration']:
            # Extract features from image
            image_features = extractor.extract_features_from_image(image_path)
            
            if image_features:
                features['pupil_metrics'].append(image_features)
                features['labels'].append('frustration')
                
        for image_path in affect_data['train']['neutral']:
            # Extract features from image
            image_features = extractor.extract_features_from_image(image_path)
            
            if image_features:
                features['pupil_metrics'].append(image_features)
                features['labels'].append('neutral')
    
    # Process Eye Gaze Net dataset (if available)
    if 'eyegaze' in datasets:
        logger.info("Processing Eye Gaze Net dataset")
        eyegaze_data = datasets['eyegaze']
        
        for i, image_path in enumerate(eyegaze_data['train']['images']):
            # Extract features from image
            image_features = extractor.extract_features_from_image(image_path)
            
            if image_features:
                # Add ground truth gaze data
                image_features['gt_gaze_x'] = eyegaze_data['train']['gaze_x'][i]
                image_features['gt_gaze_y'] = eyegaze_data['train']['gaze_y'][i]
                
                features['gaze_patterns'].append(image_features)
                features['labels'].append('gaze')
    
    logger.info(f"Feature extraction complete. "
               f"Blink patterns: {len(features['blink_patterns'])}, "
               f"Eye closure patterns: {len(features['eye_closure_patterns'])}, "
               f"Pupil metrics: {len(features['pupil_metrics'])}, "
               f"Gaze patterns: {len(features['gaze_patterns'])}")
    
    return features
