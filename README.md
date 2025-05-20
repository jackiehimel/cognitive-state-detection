# Cognitive State Detection using Eye-Tracking

This project implements a cognitive state detection system using eye-tracking data, specifically designed to identify fatigue and frustration states. It processes eye-related features using machine learning techniques trained on public datasets.

## Features

- Blink detection and pattern analysis
- PERCLOS (Percentage of eye closure) calculation
- Pupil size monitoring and analysis
- Gaze pattern analysis
- Cognitive state classification (fatigue and frustration detection)
- WebGazer.js-style adaptive calibration for long coding sessions

## Project Structure

- `src/` - Source code for the project
  - `preprocessing/` - Dataset preprocessing utilities
  - `features/` - Feature extraction modules
    - `extraction.py` - Eye feature extraction from video/images
    - `calibration.py` - Adaptive calibration system
  - `models/` - ML model implementations
  - `visualization/` - Visualization utilities for thesis figures
- `data/` - Directory for datasets (not included in repo)
- `results/` - Output directory for results and visualizations

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download the datasets from Kaggle:
   - [ZJU Eye-Blink Dataset](https://www.kaggle.com/datasets/vikassingh1996/zju-eyeblink-dataset)
   - [Real-Life Drowsiness Dataset](https://www.kaggle.com/datasets/ismailnasri20/real-life-drowsiness-dataset)
   - [CEW Dataset](https://www.kaggle.com/datasets/vikassingh1996/close-eye-or-open-eye)
   - [CAFE Dataset](https://www.kaggle.com/datasets/saworz/cafe-dataset)
   - [AffectNet](https://www.kaggle.com/datasets/mouadriali/affectnet-emotions-dataset)

3. Place the datasets in the `data/` directory

## Usage

Run the complete pipeline:
```
python src/main.py
```

Generate thesis figures:
```
python src/visualization/generate_figures.py
```

## Implementation Details

This system implements a comprehensive eye-tracking based cognitive state detection system as described in the thesis. It uses MediaPipe for face detection and extracts eye-related features to classify cognitive states including fatigue and frustration.
