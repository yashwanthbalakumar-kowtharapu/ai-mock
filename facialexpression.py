import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
def get_expression_analysis():
    """Return facial expression analysis results"""
    # Get latest frame analysis results
    expressions = analyze_facial_expressions(frame)
    
    # Format response
    response = {
        'positive': {
            'confident': expressions['positive']['confident'],
            'eye_contact': expressions['positive']['eye_contact'], 
            'relaxed': expressions['positive']['relaxed'],
            'natural_smile': expressions['positive']['natural_smile']
        },
        'negative': {
            'eye_rolling': expressions['negative']['eye_rolling'],
            'nervous': expressions['negative']['nervous'],
            'hurried': expressions['negative']['hurried'], 
            'scowling': expressions['negative']['scowling'],
            'unexpressive': expressions['negative']['unexpressive'],
            'narrowed_eyes': expressions['negative']['narrowed_eyes']
        }
    }
    
    return response
# Global variables for camera control
camera = None
is_camera_running = False

def start_camera():
    """Start the camera capture"""
    global camera, is_camera_running
    if not is_camera_running:
        camera = cv2.VideoCapture(0)
        is_camera_running = True
        return {'status': 'success', 'message': 'Camera started'}
    return {'status': 'error', 'message': 'Camera already running'}

def stop_camera():
    """Stop the camera capture"""
    global camera, is_camera_running
    if is_camera_running:
        camera.release()
        is_camera_running = False
        return {'status': 'success', 'message': 'Camera stopped'}
    return {'status': 'error', 'message': 'Camera not running'}

def get_frame():
    """Get current frame from camera"""
    global camera, is_camera_running
    if is_camera_running:
        ret, frame = camera.read()
        if ret:
            return frame
    return None


def analyze_facial_expressions(frame):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    expressions = {
        'negative': {
            'eye_rolling': 0,
            'nervous': 0, 
            'hurried': 0,
            'scowling': 0,
            'unexpressive': 0,
            'narrowed_eyes': 0
        },
        'positive': {
            'confident': 0,
            'eye_contact': 0,
            'relaxed': 0,
            'natural_smile': 0
        }
    }


    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # Analyze eye movement for rolling
        left_eye = face_landmarks.landmark[33]  # Left eye center
        right_eye = face_landmarks.landmark[263]  # Right eye center
        if left_eye.y > 0.5 and right_eye.y > 0.5:
            expressions['negative']['eye_rolling'] = 1
            
        # Check for eye contact
        if 0.4 < left_eye.y < 0.6 and 0.4 < right_eye.y < 0.6:
            expressions['positive']['eye_contact'] = 1
            
        # Analyze mouth corners for smiles/scowls and count expressions
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        
        # Count positive and negative expressions
        pos_count = sum(expressions['positive'].values())
        neg_count = sum(expressions['negative'].values())
        
        # Print dominant expression type
        if pos_count > neg_count:
            print("Positive expression detected")
        elif neg_count > pos_count:
            print("Negative expression detected")
        else:
            print("Neutral expression")
        
        if left_mouth.y < 0.6 and right_mouth.y < 0.6:
            expressions['positive']['natural_smile'] = 1
        elif left_mouth.y > 0.6 and right_mouth.y > 0.6:
            expressions['negative']['scowling'] = 1
            
        # Use DeepFace for emotion analysis
        try:
            emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Map emotions to our expression categories
            if emotion_analysis[0]['emotion']['happy'] > 50:
                expressions['positive']['relaxed'] = 1
                expressions['positive']['confident'] = 1
            
            if emotion_analysis[0]['emotion']['fear'] > 30:
                expressions['negative']['nervous'] = 1
                expressions['negative']['hurried'] = 1
                
            if emotion_analysis[0]['emotion']['neutral'] > 70:
                expressions['negative']['unexpressive'] = 1
                
            if emotion_analysis[0]['emotion']['angry'] > 30:
                expressions['negative']['narrowed_eyes'] = 1
                
        except:
            pass
            
    return expressions

def start_expression_analysis():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        expressions = analyze_facial_expressions(frame)
        
        # Display results on frame
        y_pos = 30
        for category in expressions:
            for expression, value in expressions[category].items():
                pass
                
        cv2.imshow('Facial Expression Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_expression_analysis()
