# Facial Emotion Recognition System

A real-time facial emotion detection system with web interface, multi-face detection, and database storage capabilities. This system achieves **87.40% accuracy** using MTCNN face detection and trained deep learning emotion recognition models.

## Features

- 🎯 Real-time multi-face emotion detection with MTCNN
- 🤖 **Ensemble Model Approach**: Uses 4 emotion models (FER2013, FER+, AffectNet, RAF-DB) with majority voting
- 🌐 Professional web interface with live video streaming
- 💾 Database storage with automatic deduplication
- 📊 High accuracy emotion classification (87.40%)
- 🎨 Balanced detection for all 7 emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- 🧠 Intelligent emotion selection algorithm with score averaging
- 📈 Comprehensive training results and accuracy metrics

## Technologies Used

- **DeepFace** - Deep learning facial analysis with emotion detection models
- **MTCNN** - Multi-task Cascaded Convolutional Networks for face detection
- **OpenCV** - Computer vision library for video processing
- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web framework for the user interface
- **SQLite** - Database for storing detected faces and emotions

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
   - MTCNN (Multi-task Cascaded Convolutional Networks) for robust face detection
   - 70% confidence threshold for optimal detection
   - Histogram equalization for enhanced facial features
   - Handles multiple faces simultaneously

2. **Emotion Classification:**
   - DeepFace with ensemble of 4 models: FER2013 (Emotion), FER+, AffectNet, RAF-DB
   - Majority voting system for robust emotion prediction
   - Score averaging across all models for confidence calculation
   - Intelligent emotion selection algorithm analyzing top 3 predictions
   - Adaptive confidence thresholds (12-20%) based on emotion type
   - Balanced detection preventing neutral/happy bias
   - Real-time prediction optimized for performance

3. **Web Interface:**
   - Flask-based responsive web application
   - Live video streaming with MJPEG format
   - Real-time emotion statistics and confidence scores
   - Professional dark-themed UI with clean design
   - Person name input for database association

4. **Database System:**
   - SQLite database for persistent face storage
   - Automatic deduplication (one entry per person-emotion pair)
   - Search and filter by name or emotion
   - CSV export functionality for data analysis
   - Thumbnail preview for all stored faces

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
├── app.py                    # Main Flask application with MTCNN integration
├── database.py               # SQLite database management module
├── emotion.py               # Desktop version (alternative implementation)
├── templates/
│   ├── index.html           # Main web interface with live detection
│   └── database.html        # Database viewer and management
├── accuracy_results/        # Training results and performance metrics
│   ├── confusion_matrix.png
│   ├── training_validation_curves.png
│   ├── classification_report.txt
│   └── README.md
├── detected_faces/          # Stored face images (auto-created)
├── emotion_database.db      # SQLite database (auto-created)
├── haarcascade_frontalface_default.xml  # Haar cascade file (legacy)
├── requirements.txt         # Python dependencies
├── PROJECT_SUMMARY.txt      # Project overview
└── README.md               # This file
```

## Technical Details

### Emotion Detection Algorithm
The system uses an intelligent emotion selection approach:
1. Analyzes top 3 emotion predictions from the model
2. Considers score differences between emotions
3. Prioritizes expressive emotions when scores are close
4. Applies adaptive confidence thresholds based on emotion difficulty

### Performance Optimizations
- Frame skipping (process every 3rd frame) for real-time performance
- Camera resolution optimization (480x360) for faster processing
- MTCNN confidence filtering (80% threshold)
- Histogram equalization for improved feature detection
- Efficient database queries with indexing
- JPEG compression for video streaming

### Adaptive Thresholds
- **Disgust & Fear**: 12% minimum confidence (hardest to detect)
- **Surprise**: 15% minimum confidence
- **Other emotions**: 20% minimum confidence
- Dynamic selection when emotions have similar scores

## Key Improvements

### Balanced Emotion Detection
The system implements intelligent algorithms to ensure all 7 emotions are detected fairly:
- No artificial bias toward neutral or happy
- Special handling for difficult emotions (disgust, fear, surprise)
- Score-based decision making considering multiple predictions
- Adaptive thresholds based on emotion characteristics

### Real-World Performance
- Processes 10-15 frames per second on standard hardware
- Handles multiple faces in a single frame
- Works in various lighting conditions with histogram equalization
- Minimal false positives with MTCNN confidence filtering

## Future Enhancements
- Support for additional emotion models
- Real-time emotion analytics and graphs
- Multi-language interface support
- Enhanced database reporting features
- Model fine-tuning with custom datasets

## License

This project is available for educational and research purposes.

## Credits

Built using:
- DeepFace library for emotion recognition
- MTCNN for face detection
- Flask for web framework
- OpenCV for computer vision operations



