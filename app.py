from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
import threading
import time
import os
import logging
import sys
from datetime import datetime
from face_recognizer import FaceRecognitionSystem

app = Flask(__name__)

# Configure logging to show messages in the console and save to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('face_recognition.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Global Variables ---
camera = None
face_system = None
INITIALIZATION_ERROR = None
camera_index = None
camera_backend = None
latest_frame = None
lock = threading.Lock()
run_processing = True

# --- Camera Initialization (Corrected and Improved) ---
def initialize_camera():
    """
    Tries multiple methods to initialize the camera, finds a working one,
    and handles errors gracefully.
    """
    global camera, INITIALIZATION_ERROR, camera_index, camera_backend
    
    # List of camera indices and backends to try in order of preference
    attempts = [
        # Try index 0 with specific Windows backends first
        {'index': 0, 'backend': cv2.CAP_DSHOW, 'name': 'DirectShow'},
        {'index': 0, 'backend': cv2.CAP_MSMF, 'name': 'Media Foundation'},
        # Try other indices on Windows
        {'index': 1, 'backend': cv2.CAP_DSHOW, 'name': 'DirectShow (Index 1)'},
        # Platform-agnostic attempts
        {'index': 0, 'backend': cv2.CAP_ANY, 'name': 'Default (Index 0)'},
        {'index': 1, 'backend': cv2.CAP_ANY, 'name': 'Default (Index 1)'},
        {'index': -1, 'backend': cv2.CAP_ANY, 'name': 'Auto-select'},
        # Linux-specific
        {'index': 0, 'backend': cv2.CAP_V4L2, 'name': 'V4L2 (Linux)'}
    ]
    
    for attempt in attempts:
        # Skip backends that are not relevant for the current OS
        if sys.platform != 'win32' and attempt['name'] in ['DirectShow', 'Media Foundation', 'DirectShow (Index 1)']:
            continue
        if sys.platform != 'linux' and attempt['name'] == 'V4L2 (Linux)':
            continue
            
        try:
            # *** THE CORE FIX IS HERE ***
            # Pass index and backend as SEPARATE arguments, not added together.
            camera = cv2.VideoCapture(attempt['index'], attempt['backend'])
            
            if camera and camera.isOpened():
                # Confirm the camera is working by reading a frame
                ret, frame = camera.read()
                if ret and frame is not None:
                    # Success! Store the working configuration
                    camera_index = attempt['index']
                    camera_backend = attempt['name']
                    logger.info(f"✅ Camera initialized successfully: index={camera_index}, backend='{camera_backend}'")
                    # Set a preferred resolution
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    INITIALIZATION_ERROR = None # Clear any previous errors
                    return True
                else:
                    # Camera opened but couldn't provide a frame, release it
                    camera.release()
                    logger.warning(f"⚠️ Camera opened but failed to read frame with {attempt['name']}.")
                    
        except Exception as e:
            logger.error(f"❌ Camera attempt with '{attempt['name']}' failed: {e}")
            if camera:
                camera.release()
    
    # If all attempts fail, set the final error message
    INITIALIZATION_ERROR = """Camera initialization failed. Please:
1. Verify camera is connected and powered on.
2. Close ALL other applications using the camera (Zoom, etc).
3. Check and update camera drivers in Device Manager.
4. Try a different USB port for the camera."""
    logger.error("All camera initialization attempts failed.")
    return False

def camera_reconnect_logic():
    """Handles releasing the old camera and re-initializing."""
    global camera
    logger.info("Attempting to reconnect camera...")
    if camera:
        camera.release()
        time.sleep(1) # Give hardware time to reset
    
    success = initialize_camera()
    message = "Camera reconnected successfully" if success else INITIALIZATION_ERROR
    return success, message

# --- Background Threads ---
def capture_frames():
    """Continuously captures frames from the camera in a background thread."""
    global latest_frame
    
    while run_processing:
        if camera is None or not camera.isOpened():
            time.sleep(1) # Wait if camera is not ready
            continue
            
        try:
            ret, frame = camera.read()
            if ret:
                with lock:
                    latest_frame = frame.copy() # Store the latest frame
            else:
                logger.warning("Frame capture failed. Camera might be disconnected.")
                time.sleep(2) # Wait before trying again
                
            time.sleep(1/30)  # Aim for ~30fps
            
        except Exception as e:
            logger.error(f"Critical error in capture thread: {e}")
            time.sleep(2)

def stream_generator():
    """A generator function that yields frames for the video feed."""
    while run_processing:
        try:
            if INITIALIZATION_ERROR:
                # If there's an error, generate a frame with the error message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA ERROR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset = 100
                for line in INITIALIZATION_ERROR.split('\n'):
                    cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 30
            elif latest_frame is not None:
                with lock:
                    frame = latest_frame.copy()
                # Process the frame for face recognition if the system is ready
                if face_system:
                    frame = face_system.process_frame(frame)
            else:
                # Show a "loading" frame if nothing is available yet
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Initializing...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                # Yield the frame in the format required for multipart streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1/30)
                
        except Exception as e:
            logger.error(f"Error in stream generator: {e}")
            time.sleep(1)

# --- System Initialization ---
try:
    if initialize_camera():
        face_system = FaceRecognitionSystem()
    # The error is already set in the function, so no need for an else block
except Exception as e:
    INITIALIZATION_ERROR = f"A critical error occurred on startup: {e}"
    logger.critical(INITIALIZATION_ERROR)

# Start the background thread for capturing frames
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# --- Helper Functions ---
def get_system_status():
    """Consolidates status information from all parts of the system."""
    status = {
        'camera_initialized': camera.isOpened() if camera else False,
        'camera_index': camera_index,
        'camera_backend': camera_backend,
        'last_error': INITIALIZATION_ERROR,
        'timestamp': datetime.now().isoformat()
    }
    if face_system:
        status.update(face_system.get_system_stats())
    else:
        status.update({'users_registered': 0, 'is_trained': False, 'last_trained': None, 'face_samples': 0})
    return status

def check_system_ready(func):
    """Decorator to protect routes that need the face system to be initialized."""
    def wrapper(*args, **kwargs):
        if not face_system:
            return jsonify({'status': 'error', 'message': 'System is not ready. Face Recognition module failed to initialize.'}), 503
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/system_status', methods=['GET'])
def system_status():
    return jsonify(get_system_status())

@app.route('/camera_reconnect', methods=['POST'])
def handle_camera_reconnect():
    success, message = camera_reconnect_logic()
    status_code = 200 if success else 500
    return jsonify({'status': 'success' if success else 'error', 'message': message, 'system_status': get_system_status()}), status_code

@app.route('/users', methods=['GET'])
@check_system_ready
def get_users():
    users = sorted(list(face_system.name_map.values()))
    return jsonify({'users': users})

@app.route('/register', methods=['POST'])
@check_system_ready
def register_face():
    data = request.get_json()
    name = data.get('name')
    image_data_url = data.get('image')

    if not name or not image_data_url:
        return jsonify({'status': 'error', 'message': 'Missing name or image data'}), 400

    try:
        # Decode the base64 image from the data URL
        header, encoded = image_data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Could not decode image'}), 400

        success, message = face_system.register_face(img, name)
        if success:
            return jsonify({'status': 'success', 'message': message})
        else:
            return jsonify({'status': 'error', 'message': message}), 400
            
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        return jsonify({'status': 'error', 'message': f'Server error: {e}'}), 500

@app.route('/retrain', methods=['POST'])
@check_system_ready
def retrain_model():
    success, message = face_system.train_model()
    status_code = 200 if success else 400
    return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

@app.route('/users/<name>', methods=['DELETE'])
@check_system_ready
def delete_user(name):
    success, message = face_system.delete_user(name)
    if not success:
        return jsonify({'status': 'error', 'message': message}), 404 # Not Found or Bad Request
    
    # After deleting a user, retrain the model with the remaining data
    if face_system.name_map: # Check if there are any users left
        train_success, train_message = face_system.train_model()
        if not train_success:
            message += f" | WARNING: Post-deletion retrain failed: {train_message}"
    else:
        # No users left, so mark model as not trained
        face_system.is_trained = False
        logger.info("All users deleted. Model is no longer trained.")

    return jsonify({'status': 'success', 'message': message})

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # use_reloader=False is important to prevent the app from initializing twice
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        # Clean up resources on exit
        run_processing = False
        if capture_thread.is_alive():
            capture_thread.join()
        if camera:
            camera.release()
        logger.info("Application shutdown complete.")
