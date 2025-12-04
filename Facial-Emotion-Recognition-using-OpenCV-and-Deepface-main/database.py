"""
Database module for storing face detections with emotions
"""
import sqlite3
import os
from datetime import datetime
import base64

class EmotionDatabase:
    def __init__(self, db_path='emotion_database.db', images_dir='detected_faces'):
        self.db_path = db_path
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create sessions table to track detection sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_faces INTEGER DEFAULT 0
            )
        ''')
        
        # Create emotion statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT NOT NULL UNIQUE,
                count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def face_exists(self, name, emotion):
        """Check if a face with this name and emotion already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM faces 
            WHERE name = ? AND emotion = ?
        ''', (name, emotion))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def save_face(self, name, emotion, confidence, image, session_id=None):
        """Save a detected face to database (only once per person-emotion combination)"""
        import cv2
        
        # Check if this person-emotion combination already exists
        if self.face_exists(name, emotion):
            return None, None  # Already saved, skip
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{name}_{emotion}_{timestamp}.jpg"
        image_path = os.path.join(self.images_dir, filename)
        
        # Save image
        cv2.imwrite(image_path, image)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO faces (name, emotion, confidence, image_path)
            VALUES (?, ?, ?, ?)
        ''', (name, emotion, confidence, image_path))
        
        face_id = cursor.lastrowid
        
        # Update emotion statistics
        cursor.execute('''
            INSERT OR REPLACE INTO emotion_stats (emotion, count, last_updated)
            VALUES (?, 
                    COALESCE((SELECT count FROM emotion_stats WHERE emotion = ?), 0) + 1,
                    CURRENT_TIMESTAMP)
        ''', (emotion, emotion))
        
        conn.commit()
        conn.close()
        
        return face_id, image_path
    
    def get_all_faces(self, limit=None):
        """Retrieve all faces from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM faces ORDER BY timestamp DESC'
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query)
        faces = cursor.fetchall()
        conn.close()
        
        return faces
    
    def get_faces_by_name(self, name):
        """Get all faces for a specific person"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM faces WHERE name = ? ORDER BY timestamp DESC
        ''', (name,))
        
        faces = cursor.fetchall()
        conn.close()
        
        return faces
    
    def get_faces_by_emotion(self, emotion):
        """Get all faces with specific emotion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM faces WHERE emotion = ? ORDER BY timestamp DESC
        ''', (emotion,))
        
        faces = cursor.fetchall()
        conn.close()
        
        return faces
    
    def get_emotion_statistics(self):
        """Get emotion distribution statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT emotion, COUNT(*) as count 
            FROM faces 
            GROUP BY emotion 
            ORDER BY count DESC
        ''')
        
        stats = cursor.fetchall()
        conn.close()
        
        return stats
    
    def get_person_statistics(self):
        """Get statistics per person"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, emotion, COUNT(*) as count 
            FROM faces 
            GROUP BY name, emotion 
            ORDER BY name, count DESC
        ''')
        
        stats = cursor.fetchall()
        conn.close()
        
        return stats
    
    def search_faces(self, name=None, emotion=None, start_date=None, end_date=None):
        """Advanced search for faces"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM faces WHERE 1=1'
        params = []
        
        if name:
            query += ' AND name LIKE ?'
            params.append(f'%{name}%')
        
        if emotion:
            query += ' AND emotion = ?'
            params.append(emotion)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        faces = cursor.fetchall()
        conn.close()
        
        return faces
    
    def delete_face(self, face_id):
        """Delete a face record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get image path before deleting
        cursor.execute('SELECT image_path FROM faces WHERE id = ?', (face_id,))
        result = cursor.fetchone()
        
        if result:
            image_path = result[0]
            if os.path.exists(image_path):
                os.remove(image_path)
            
            cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
    
    def get_total_count(self):
        """Get total number of faces in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM faces')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def export_to_csv(self, filename='face_detections.csv'):
        """Export database to CSV"""
        import csv
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces ORDER BY timestamp DESC')
        faces = cursor.fetchall()
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Emotion', 'Confidence', 'Image Path', 'Timestamp'])
            writer.writerows(faces)
        
        conn.close()
        return filename
