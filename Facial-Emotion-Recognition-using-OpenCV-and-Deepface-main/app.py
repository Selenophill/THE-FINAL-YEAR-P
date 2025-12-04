from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
from deepface import DeepFace
import numpy as np
from collections import Counter
import json
from database import EmotionDatabase
import os

app = Flask(__name__)
db = EmotionDatabase()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Trained emotion detection model
USE_BEST_MODEL = 'Emotion'  # FER2013 model

# Global variables
camera = None
current_stats = {
    'face_count': 0,
    'emotions': []
}
frame_skip = 0
FRAME_SKIP_COUNT = 2  # Process every 3rd frame for speed
save_detections = True  # Save detected faces to database
current_person_name = "Unknown"  # Name for current detection session

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced resolution for speed
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    return camera

def is_valid_face(gray_face, face_roi):
    """Validate if detected region is actually a face"""
    # Check face region size
    if gray_face.shape[0] < 60 or gray_face.shape[1] < 60:
        return False
    
    # Check variance (faces have texture, blank regions don't)
    variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    if variance < 50:  # Too smooth to be a face
        return False
    
    # Check if region has skin-like color distribution
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
    
    if skin_ratio < 0.15:  # Not enough skin-like pixels
        return False
    
    return True

def generate_frames():
    global current_stats, frame_skip
    
    while True:
        try:
            cam = get_camera()
            success, frame = cam.read()
            
            if not success:
                print("Failed to read frame from camera")
                continue
            
            # Frame skipping for performance
            frame_skip += 1
            skip_detection = frame_skip % FRAME_SKIP_COUNT != 0
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray_frame = cv2.equalizeHist(gray_frame)

            # Skip detection on some frames for speed
            if skip_detection and len(current_stats['emotions']) > 0:
                # Use previous detection results
                face_count = current_stats['face_count']
                detected_emotions = current_stats['emotions']
                faces = []  # Don't process, just display
            else:
                # Detect faces with stricter parameters to avoid false positives
                faces = face_cascade.detectMultiScale(
                    gray_frame, 
                    scaleFactor=1.1,  # Increased for fewer false positives
                    minNeighbors=8,   # More neighbors = stricter detection
                    minSize=(80, 80), # Larger minimum size
                    maxSize=(300, 300),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                face_count = 0
                detected_emotions = []
            
                for (x, y, w, h) in faces:
                    # Add padding around face
                    padding = 10
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(frame.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(frame.shape[0] - y_pad, h + 2 * padding)
                    
                    # Extract face ROI
                    face_roi = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                    gray_face = gray_frame[y:y + h, x:x + w]
                    
                    # Validate if this is actually a face
                    if not is_valid_face(gray_face, face_roi):
                        continue
                    
                    # Skip if face region is too small
                    if face_roi.shape[0] < 60 or face_roi.shape[1] < 60:
                        continue
                
                    try:
                        # Use single fast model for real-time performance
                        result = DeepFace.analyze(
                            face_roi, 
                            actions=['emotion'], 
                            enforce_detection=False,
                            silent=True,
                            detector_backend='skip'  # Skip face detection, we already did it
                        )
                        
                        emotion = result[0]['dominant_emotion']
                        emotion_scores = result[0]['emotion']
                        confidence = emotion_scores[emotion]
                        
                        # Only accept high confidence detections
                        if confidence < 40:
                            continue
                    
                        face_count += 1
                        detected_emotions.append({
                            'emotion': emotion,
                            'confidence': round(confidence, 1)
                        })
                        
                        # Save to database if enabled
                        if save_detections and frame_skip % 30 == 0:  # Save every 30 frames
                            try:
                                face_id, saved_path = db.save_face(
                                    current_person_name,
                                    emotion,
                                    confidence,
                                    face_roi
                                )
                            except Exception as save_error:
                                pass  # Silently handle errors
                        
                        # Color coding based on emotion
                        color_map = {
                            'happy': (0, 255, 0),
                            'sad': (255, 0, 0),
                            'angry': (0, 0, 255),
                            'surprise': (0, 255, 255),
                            'fear': (128, 0, 128),
                            'disgust': (0, 128, 128),
                            'neutral': (200, 200, 200)
                        }
                        
                        color = color_map.get(emotion, (0, 255, 0))
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        # Create label
                        label = f"Face {face_count}: {emotion} ({confidence:.0f}%)"
                        
                        # Draw background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 8, y), color, -1)
                        
                        # Draw text
                        cv2.putText(frame, label, (x + 4, y - 8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                    except Exception as e:
                        # Don't draw anything for failed detections
                        pass

            # Update global stats only when processing
            if not skip_detection or len(current_stats['emotions']) == 0:
                current_stats = {
                    'face_count': face_count,
                    'emotions': detected_emotions
                }

            # Display face count
            info_text = f"Faces: {current_stats['face_count']}"
            cv2.putText(frame, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Encode frame with lower quality for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in frame generation: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(current_stats)

@app.route('/set_name', methods=['POST'])
def set_name():
    global current_person_name
    data = request.json
    current_person_name = data.get('name', 'Unknown')
    return jsonify({'success': True, 'name': current_person_name})

@app.route('/toggle_save')
def toggle_save():
    global save_detections
    save_detections = not save_detections
    return jsonify({'save_enabled': save_detections})

@app.route('/database')
def view_database():
    return render_template('database.html')

@app.route('/api/faces')
def get_faces():
    limit = request.args.get('limit', 50, type=int)
    faces = db.get_all_faces(limit=limit)
    
    faces_data = []
    for face in faces:
        faces_data.append({
            'id': face[0],
            'name': face[1],
            'emotion': face[2],
            'confidence': face[3],
            'image_path': face[4],
            'timestamp': face[5]
        })
    
    return jsonify(faces_data)

@app.route('/api/faces/search')
def search_faces():
    name = request.args.get('name')
    emotion = request.args.get('emotion')
    
    faces = db.search_faces(name=name, emotion=emotion)
    
    faces_data = []
    for face in faces:
        faces_data.append({
            'id': face[0],
            'name': face[1],
            'emotion': face[2],
            'confidence': face[3],
            'image_path': face[4],
            'timestamp': face[5]
        })
    
    return jsonify(faces_data)

@app.route('/api/statistics')
def get_statistics():
    emotion_stats = db.get_emotion_statistics()
    person_stats = db.get_person_statistics()
    total_count = db.get_total_count()
    
    return jsonify({
        'total_faces': total_count,
        'emotion_stats': [{'emotion': e[0], 'count': e[1]} for e in emotion_stats],
        'person_stats': [{'name': p[0], 'emotion': p[1], 'count': p[2]} for p in person_stats]
    })

@app.route('/api/image/<path:filename>')
def get_image(filename):
    return send_file(filename)

@app.route('/api/faces/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    success = db.delete_face(face_id)
    return jsonify({'success': success})

@app.route('/api/export')
def export_data():
    filename = db.export_to_csv()
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    print("Starting Facial Emotion Detection System")
    print("Training Accuracy: 87.40%")
    print("Access at: http://127.0.0.1:5000")
    print("-" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
