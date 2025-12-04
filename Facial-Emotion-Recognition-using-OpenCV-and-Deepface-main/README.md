# Facial Emotion Recognition System

A comprehensive real-time facial emotion detection system with web interface, multi-face detection, and database storage capabilities. This system achieves **87.40% accuracy** on emotion classification using trained deep learning models.

## Features

- 🎯 Real-time multi-face emotion detection
- 🌐 Professional web interface with live video streaming
- 💾 Database storage for detected faces with emotions
- 📊 High accuracy emotion classification (87.40%)
- 🎨 Support for 7 emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- 🔍 Advanced face validation to prevent false positives
- 📈 Comprehensive accuracy metrics and training results

## Technologies Used

- **DeepFace** - Deep learning facial analysis with emotion detection models
- **OpenCV** - Computer vision library for face detection and video processing
- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web framework for the user interface
- **SQLite** - Database for storing detected faces and emotions
- **Haar Cascade** - Face detection algorithm

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application:**
   ```bash
   python app.py
   ```

3. **Access the System:**
   - Open browser at `http://127.0.0.1:5000`
   - Live video feed with emotion detection
   - Database view at `http://127.0.0.1:5000/database`

## System Architecture

### Core Components

1. **Face Detection Pipeline:**
   - Haar Cascade classifier for face detection
   - Validation filters (texture analysis, skin tone, size constraints)
   - Confidence thresholding to eliminate false positives

2. **Emotion Classification:**
   - Trained deep learning models for emotion recognition
   - Support for multiple model backends (FER2013, FER+, AffectNet, RAF-DB)
   - Real-time prediction with optimized performance

3. **Web Interface:**
   - Live video streaming with MJPEG
   - Real-time statistics and emotion tracking
   - Professional dark-themed UI

4. **Database System:**
   - SQLite database for face storage
   - Automatic deduplication (one entry per person-emotion combination)
   - Search, filter, and export capabilities

## Training & Accuracy

The system was trained on multiple emotion datasets and achieves:

- **Overall Accuracy:** 87.40%
- **Precision:** 88.19%
- **Recall:** 87.40%
- **F1-Score:** 87.55%

### Per-Emotion Performance:
- Happy: 94.0%
- Surprise: 92.3%
- Neutral: 87.8%
- Angry: 88.3%
- Fear: 83.3%
- Sad: 82.9%
- Disgust: 78.2%

Training results and visualizations are available in the `accuracy_results/` directory including:
- Training/validation curves
- Confusion matrix
- Per-emotion accuracy graphs
- Detailed classification reports

## Usage Guide

### Live Detection:
1. Enter person's name in the input field
2. Click "Set Name" to associate detections
3. Toggle "Save: ON/OFF" to control database storage
4. Faces are automatically saved with their detected emotions

### Database Management:
1. Navigate to `/database` route
2. View all detected faces with thumbnails
3. Search by name or filter by emotion
4. Export data to CSV for analysis
5. Delete individual entries as needed

## Project Structure

```
├── app.py                    # Main Flask application
├── database.py               # Database management module
├── emotion.py               # Desktop version (legacy)
├── templates/
│   ├── index.html           # Main web interface
│   └── database.html        # Database viewer
├── accuracy_results/        # Training results and metrics
├── detected_faces/          # Stored face images
├── haarcascade_frontalface_default.xml
└── requirements.txt
```

## Performance Optimizations

- Frame skipping (process every 3rd frame)
- Reduced camera resolution (480x360)
- Face validation pipeline
- Efficient database queries
- JPEG compression for streaming

## License

This project is available for educational and research purposes.



