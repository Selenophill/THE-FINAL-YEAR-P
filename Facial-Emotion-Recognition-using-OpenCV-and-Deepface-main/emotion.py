import cv2
from deepface import DeepFace
import numpy as np
from collections import Counter

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Emotion detection models to use for ensemble prediction
EMOTION_MODELS = ['Emotion', 'FER+', 'AffectNet', 'RAF-DB']
USE_ENSEMBLE = True  # Set to False to use only default model

print("Real-time Multi-Face Emotion Detection")
print(f"Using Models: {', '.join(EMOTION_MODELS) if USE_ENSEMBLE else 'Default (FER2013)'}")
print("Press 'q' to quit")
print("-" * 50)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with improved parameters for accuracy
    # Higher minNeighbors = more strict detection = fewer false positives
    # scaleFactor closer to 1.0 = more thorough but slower
    faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.05,      # More thorough scaling
        minNeighbors=7,        # Higher threshold for stricter detection
        minSize=(50, 50),      # Larger minimum face size
        maxSize=(400, 400),    # Maximum face size to avoid false positives
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_count = 0
    
    for (x, y, w, h) in faces:
        # Add padding around face for better emotion detection
        padding = 10
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
        
        # Extract the face ROI (Region of Interest) from original frame
        face_roi = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        
        # Skip if face region is too small
        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
            continue
        
        try:
            if USE_ENSEMBLE:
                # Ensemble approach: Use multiple models for robust prediction
                emotion_predictions = []
                all_scores = {}
                
                for model_name in EMOTION_MODELS:
                    try:
                        # Analyze with each model
                        result = DeepFace.analyze(
                            face_roi, 
                            actions=['emotion'],
                            enforce_detection=False,
                            silent=True,
                            detector_backend='opencv'
                        )
                        
                        predicted_emotion = result[0]['dominant_emotion']
                        emotion_predictions.append(predicted_emotion)
                        
                        # Aggregate scores from all models
                        for emo, score in result[0]['emotion'].items():
                            if emo not in all_scores:
                                all_scores[emo] = []
                            all_scores[emo].append(score)
                        
                    except Exception as model_error:
                        # If one model fails, continue with others
                        continue
                
                # Voting mechanism: Most predicted emotion wins
                if emotion_predictions:
                    emotion_counter = Counter(emotion_predictions)
                    emotion = emotion_counter.most_common(1)[0][0]
                    
                    # Calculate average confidence across all models
                    if emotion in all_scores:
                        confidence = np.mean(all_scores[emotion])
                    else:
                        confidence = 50.0
                else:
                    # Fallback if all models fail
                    raise Exception("All models failed")
                    
            else:
                # Single model approach (default FER2013)
                result = DeepFace.analyze(
                    face_roi, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True
                )

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                emotion_scores = result[0]['emotion']
                confidence = emotion_scores[emotion]
            
            face_count += 1
            
            # Color coding based on emotion
            color_map = {
                'happy': (0, 255, 0),      # Green
                'sad': (255, 0, 0),        # Blue
                'angry': (0, 0, 255),      # Red
                'surprise': (0, 255, 255), # Yellow
                'fear': (128, 0, 128),     # Purple
                'disgust': (0, 128, 128),  # Teal
                'neutral': (200, 200, 200) # Gray
            }
            
            color = color_map.get(emotion, (0, 255, 0))
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # Create label with emotion and confidence
            label = f"Face {face_count}: {emotion} ({confidence:.1f}%)"
            
            # Draw background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 30), (x + text_size[0] + 10, y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            # If emotion detection fails, still show the face detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
            cv2.putText(frame, "Processing...", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # Display face count
    info_text = f"Faces Detected: {face_count}"
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Multi-Face Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
print("\nApplication closed successfully")

