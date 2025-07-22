import cv2
import numpy as np
import os
import pickle
import time
import logging
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, db_path="face_database.pkl", face_data_path="face_data"):
        self.db_path = db_path
        self.face_data_path = face_data_path
        os.makedirs(self.face_data_path, exist_ok=True)
        
        # Initialize with better parameters
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=85
        )
        self.name_map = {}
        self.last_detected_names = []
        self.is_trained = False
        self.last_trained = None
        self.load_database_and_model()
        
        # Configure logging
        self.logger = logging.getLogger('FaceRecognitionSystem')
        self.logger.setLevel(logging.INFO)
        
    def load_database_and_model(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.name_map = data
                        self.logger.info(f"Loaded database with {len(self.name_map)} users")
                    else:
                        self.name_map = {}
                        self.logger.warning("Database file was corrupted - starting fresh")
            
            if os.path.exists("recognizer.yml"):
                self.recognizer.read("recognizer.yml")
                self.is_trained = True
                self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.logger.info("Loaded trained recognizer model")
                
        except Exception as e:
            self.logger.error(f"Error loading database/model: {str(e)}")
            self.name_map = {}
            self.is_trained = False

    def save_database(self):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.name_map, f)
            self.logger.debug("Database saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving database: {str(e)}")

    def save_model(self):
        try:
            self.recognizer.save("recognizer.yml")
            self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.debug("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")

    def detect_faces(self, gray_frame):
        # Improved face detection with multiple attempts
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no faces found, try with different parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(80, 80)
            )
            
        return faces

    def register_face(self, image, name):
        try:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray_frame)
            
            if len(faces) != 1:
                msg = f"Found {len(faces)} faces. Please ensure only one face is in view."
                self.logger.warning(msg)
                return False, msg
                
            (x, y, w, h) = faces[0]
            face_roi = gray_frame[y:y+h, x:x+w]
            
            # Find existing user or create new
            person_id = None
            for pid, pname in self.name_map.items():
                if pname.lower() == name.lower():
                    person_id = pid
                    break
                    
            if person_id is None:
                person_id = max(self.name_map.keys(), default=-1) + 1
                self.name_map[person_id] = name
                self.save_database()
                self.logger.info(f"Created new user ID {person_id} for {name}")

            # Save face sample
            timestamp = int(time.time() * 1000)
            filename = f"{person_id}_{timestamp}.png"
            filepath = os.path.join(self.face_data_path, filename)
            cv2.imwrite(filepath, face_roi)
            
            self.logger.info(f"Saved face sample for {name} (ID: {person_id})")
            return True, f"Saved sample for {name}"
            
        except Exception as e:
            self.logger.error(f"Error registering face: {str(e)}")
            return False, f"Error: {str(e)}"

    def train_model(self):
        try:
            face_samples, ids = [], []
            if not os.path.exists(self.face_data_path):
                return False, "Training failed: Face data directory not found"
                
            image_paths = [os.path.join(self.face_data_path, f) 
                          for f in os.listdir(self.face_data_path)
                          if f.endswith('.png')]
            
            if not image_paths:
                return False, "Training failed: No face samples found"
                
            for image_path in image_paths:
                try:
                    pil_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if pil_image is None:
                        continue
                    face_samples.append(pil_image)
                    person_id = int(os.path.split(image_path)[-1].split("_")[0])
                    ids.append(person_id)
                except Exception as e:
                    self.logger.warning(f"Error loading {image_path}: {str(e)}")
                    continue
                    
            if len(face_samples) < 2:
                return False, "Training failed: Need at least 2 face samples"
                
            self.recognizer.train(face_samples, np.array(ids))
            self.save_model()
            self.is_trained = True
            msg = f"Model trained on {len(face_samples)} samples"
            self.logger.info(msg)
            return True, msg
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            return False, f"Training error: {str(e)}"
    
    def delete_user(self, name_to_delete):
        try:
            person_id_to_delete = None
            for pid, pname in self.name_map.items():
                if pname.lower() == name_to_delete.lower():
                    person_id_to_delete = pid
                    break
                    
            if person_id_to_delete is None:
                return False, "User not found"
                
            # Delete all face samples
            files_deleted = 0
            for filename in os.listdir(self.face_data_path):
                if filename.startswith(f"{person_id_to_delete}_"):
                    try:
                        os.remove(os.path.join(self.face_data_path, filename))
                        files_deleted += 1
                    except Exception as e:
                        self.logger.warning(f"Error deleting {filename}: {str(e)}")
                        continue
                        
            # Remove from name map
            del self.name_map[person_id_to_delete]
            self.save_database()
            
            msg = f"Deleted {name_to_delete} and {files_deleted} samples"
            self.logger.info(msg)
            return True, msg
            
        except Exception as e:
            self.logger.error(f"Error deleting user: {str(e)}")
            return False, f"Error: {str(e)}"

    def process_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(gray)
            self.last_detected_names = []
            
            for (x, y, w, h) in faces:
                name = "Unknown"
                confidence = 0
                
                if self.is_trained:
                    person_id, conf = self.recognizer.predict(gray[y:y+h, x:x+w])
                    confidence = max(0, min(100, round(100 - conf)))
                    
                    if confidence > 65:  # Only show if confidence > 65%
                        name = self.name_map.get(person_id, "Unknown")
                
                self.last_detected_names.append((name, confidence))
                
                # Draw rectangle and label with confidence
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence}%)" if name != "Unknown" else name
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame

    def get_system_stats(self):
        return {
            'users_registered': len(self.name_map),
            'is_trained': self.is_trained,
            'last_trained': self.last_trained,
            'face_samples': len(os.listdir(self.face_data_path)) 
                           if os.path.exists(self.face_data_path) else 0
        }