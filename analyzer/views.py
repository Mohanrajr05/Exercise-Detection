import os
import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pathlib import Path
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style

# --- GLOBAL CONFIGURATION ---
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploaded_videos'
MODEL_DIR = BASE_DIR / 'models'
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    'frame_count': 16,
    'image_size': (160, 160)
}

# --- MEDIAPIPE INITIALIZATION ---
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def preprocess_frames_for_model(frames_list):
    """Prepares a list of frames for model prediction."""
    processed_frames = [
        cv2.resize(frame, MODEL_CONFIG['image_size']).astype('float32') / 255.0
        for frame in frames_list
    ]
    return np.expand_dims(np.array(processed_frames), axis=0)

# --- ENHANCED RULE-BASED LOGIC WITH FORM FEEDBACK ---

def update_pushup_state(landmarks, state):
    """Rule-based logic for counting push-ups with form validation."""
    state['feedback'] = []
    is_form_correct = False
    
    try:
        # Get coordinates for left side
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # 1. Form Validation: Check if back is straight
        body_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
        if body_angle > 155 and body_angle < 205:
            is_form_correct = True
        else:
            state['feedback'].append("Keep your back straight")
            is_form_correct = False

        # 2. Rep Counting (only if form is correct)
        if is_form_correct:
            elbow_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
            if elbow_angle < 90 and not state['is_down']: # Down position
                state['is_down'] = True
            elif elbow_angle > 160 and state['is_down']: # Up position
                state['count'] += 1
                state['is_down'] = False
                state['feedback'].append("Great Rep!")

    except Exception:
        state['feedback'].append("Ensure your full body is visible")
    return state

def update_plank_state(landmarks, state, is_reverse=False, is_side=False):
    """Rule-based logic for timing planks with form feedback."""
    state['feedback'] = []
    try:
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        body_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
        
        is_form_correct = False
        # Angle thresholds vary slightly for different plank types
        if is_side:
            if body_angle > 150 and body_angle < 210: is_form_correct = True
        elif is_reverse:
             if body_angle > 150 and body_angle < 210: is_form_correct = True
        else: # Regular plank
            if body_angle > 160 and body_angle < 200: is_form_correct = True

        if is_form_correct:
            state['feedback'].append("Holding form correctly")
            if state['start_time'] is None:
                state['start_time'] = time.time()
            state['duration'] = state['cumulative_time'] + (time.time() - state['start_time'])
        else:
            state['feedback'].append("Straighten your body")
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
    except Exception:
        if state['start_time'] is not None:
            state['cumulative_time'] += time.time() - state['start_time']
            state['start_time'] = None
    return state

def update_squat_state(landmarks, state):
    """Rule-based logic for counting squats with form validation."""
    state['feedback'] = []
    try:
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        knee_angle = calculate_angle(hip_l, knee_l, ankle_l)
        
        # 1. Going down: check for depth (hip below knee) and angle
        if knee_angle < 100 and (hip_y > knee_y) and not state['is_down']:
            state['is_down'] = True
            state['feedback'].append("Good Depth!")
        # 2. Coming up
        elif knee_angle > 165 and state['is_down']:
            state['count'] += 1
            state['is_down'] = False
            state['feedback'].append("Good Squat!")
        
        # 3. Feedback during movement
        if not state['is_down'] and (knee_angle < 120) and (hip_y < knee_y):
            state['feedback'].append("Go deeper!")

    except Exception:
        state['feedback'].append("Ensure your full body is visible")
    return state

def update_situp_state(landmarks, state):
    """Rule-based logic for counting sit-ups."""
    state['feedback'] = []
    try:
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        hip_angle = calculate_angle(shoulder_l, hip_l, knee_l)

        if hip_angle < 90 and not state['is_down']: # Up position
            state['is_down'] = True
        elif hip_angle > 150 and state['is_down']: # Down position
            state['is_down'] = False
            state['count'] += 1
            state['feedback'].append("Nice Rep!")

    except Exception:
        state['feedback'].append("Lie flat and keep feet on ground")
    return state

def update_jumping_jacks_state(landmarks, state):
    """Rule-based logic for counting jumping jacks."""
    state['feedback'] = []
    try:
        shoulder_l, shoulder_r = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        ankle_l, ankle_r = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        shoulder_width = abs(shoulder_l.x - shoulder_r.x)
        ankle_dist = abs(ankle_l.x - ankle_r.x)
        arms_are_up = wrist_l.y < shoulder_l.y

        # "Up" position: feet wide and arms up
        if ankle_dist > shoulder_width * 0.9 and arms_are_up and not state['is_up']:
            state['is_up'] = True
        # "Down" position: feet together and arms down
        elif ankle_dist < shoulder_width * 0.5 and not arms_are_up and state['is_up']:
            state['is_up'] = False
            state['count'] += 1
            state['feedback'].append("Good Jack!")
        
        if state['is_up'] and not arms_are_up:
            state['feedback'].append("Arms up!")
        if not state['is_up'] and arms_are_up:
             state['feedback'].append("Arms down!")

    except Exception:
        state['feedback'].append("Stand straight and face camera")
    return state

# --- UNIFIED ANALYSIS AND GENERATOR FUNCTIONS ---

def analyze_video_single_pass(filepath, exercise_model, rule_logic_func, initial_rule_state):
    """Analyzes a video in a single pass for model prediction and rule-based counting."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return {'error': 'Cannot open video file.'}

    frames_buffer = []
    model_detected_segments = 0
    total_segments_processed = 0
    rule_state = initial_rule_state.copy()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frames_buffer.append(frame.copy())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        if results.pose_landmarks:
            rule_state = rule_logic_func(results.pose_landmarks.landmark, rule_state)

        if len(frames_buffer) == MODEL_CONFIG['frame_count']:
            total_segments_processed += 1
            if exercise_model:
                model_input = preprocess_frames_for_model(frames_buffer)
                prediction = exercise_model.predict(model_input, verbose=0)
                if np.argmax(prediction[0]) == 1:
                    model_detected_segments += 1
            frames_buffer.pop(0)

    cap.release()

    model_confidence = (model_detected_segments / total_segments_processed * 100) if total_segments_processed > 0 else 0
    final_results = {
        'model_exercise_confidence_percent': round(model_confidence, 2),
        'model_status': 'Model loaded and analyzed.' if exercise_model else 'Model not available.'
    }
    if 'count' in rule_state:
        final_results['rule_based_reps'] = rule_state['count']
    if 'duration' in rule_state:
        final_results['rule_based_duration_seconds'] = round(rule_state.get('cumulative_time', rule_state.get('duration',0)), 2)

    return final_results


def generate_live_feed(camera_index, exercise_model, rule_logic_func, initial_rule_state, exercise_name):
    """A single, generic generator for all live exercise analysis feeds with real-time feedback."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        return

    frames_buffer = []
    model_label, model_color = "Initializing...", (0, 255, 255)
    rule_state = initial_rule_state.copy()

    while True:
        success, frame = camera.read()
        if not success: break

        display_frame = cv2.resize(frame.copy(), (640, 480))
        frames_buffer.append(frame.copy())

        # 1. Model Prediction
        if len(frames_buffer) == MODEL_CONFIG['frame_count']:
            if exercise_model:
                model_input = preprocess_frames_for_model(frames_buffer)
                prediction = exercise_model.predict(model_input, verbose=0)
                if np.argmax(prediction[0]) == 1:
                    model_label, model_color = f"{exercise_name} Detected", (0, 255, 0)
                else:
                    model_label, model_color = f"Not {exercise_name}", (0, 0, 255)
            else:
                model_label, model_color = f"{exercise_name} Model N/A", (255, 0, 0)
            frames_buffer.pop(0)

        # 2. Rule-based analysis and form feedback
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        if results.pose_landmarks:
            rule_state = rule_logic_func(results.pose_landmarks.landmark, rule_state)
            draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=get_default_pose_landmarks_style())
        
        # 3. Display Info
        cv2.putText(display_frame, model_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_color, 2)
        
        if 'count' in rule_state:
            rule_text = f"Reps: {rule_state['count']}"
        elif 'duration' in rule_state:
            duration = int(rule_state.get('duration', 0))
            rule_text = f"Duration: {duration}s"
        else:
            rule_text = "No rule data"
        cv2.putText(display_frame, rule_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # --- NEW: Display Real-Time Feedback ---
        y_pos = 110
        for msg in rule_state.get('feedback', []):
            color = (0, 255, 0) if "Good" in msg or "Great" in msg or "Nice" in msg else (0, 0, 255)
            cv2.putText(display_frame, msg, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

# --- MODEL AND EXERCISE CONFIGURATION ---
def try_load_model(model_name):
    model_path = MODEL_DIR / f"{model_name}_classification_model.keras"
    if model_path.exists():
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    return None

EXERCISE_CONFIG = {
    "pushup": {
        "model": try_load_model("pushup"),
        "rule_logic": update_pushup_state,
        "initial_state": {'count': 0, 'is_down': False, 'feedback': []}
    },
    "plank": {
        "model": try_load_model("plank"),
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_reverse=False, is_side=False),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    },
    "squat": {
        "model": try_load_model("squat"),
        "rule_logic": update_squat_state,
        "initial_state": {'count': 0, 'is_down': False, 'feedback': []}
    },
    "situp": {
        "model": try_load_model("situp"),
        "rule_logic": update_situp_state,
        "initial_state": {'count': 0, 'is_down': False, 'feedback': []}
    },
    "jumping_jacks": {
        "model": try_load_model("jumping_jacks"),
        "rule_logic": update_jumping_jacks_state,
        "initial_state": {'count': 0, 'is_up': False, 'feedback': []}
    },
    "reverse_plank": {
        "model": try_load_model("reverse_plank"),
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_reverse=True),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    },
    "side_plank": {
        "model": try_load_model("side_plank"),
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_side=True),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    }
}

# --- DJANGO VIEWS ---
def _video_upload_handler(request, exercise_name):
    """Generic handler for all video upload POST requests."""
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file provided.'}, status=400)

    video, config = request.FILES['video'], EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return JsonResponse({'error': f'Invalid exercise: {exercise_name}'}, status=400)

    fs = FileSystemStorage(location=UPLOAD_DIR)
    filepath = UPLOAD_DIR / fs.save(video.name, video)
    
    try:
        results = analyze_video_single_pass(filepath, config["model"], config["rule_logic"], config["initial_state"])
        results['exercise_type'] = exercise_name
        return JsonResponse(results)
    except Exception as e:
        return JsonResponse({'error': f'Analysis error: {e}'}, status=500)
    finally:
        if filepath.exists(): os.remove(filepath)

# Simplified Django Views
@csrf_exempt
def upload_and_analyze_pushup(request): return _video_upload_handler(request, "pushup")
@csrf_exempt
def upload_and_analyze_plank(request): return _video_upload_handler(request, "plank")
@csrf_exempt
def upload_and_analyze_squat(request): return _video_upload_handler(request, "squat")
@csrf_exempt
def upload_and_analyze_situp(request): return _video_upload_handler(request, "situp")
@csrf_exempt
def upload_and_analyze_jumping_jacks(request): return _video_upload_handler(request, "jumping_jacks")
@csrf_exempt
def upload_and_analyze_reverse_plank(request): return _video_upload_handler(request, "reverse_plank")
@csrf_exempt
def upload_and_analyze_side_plank(request): return _video_upload_handler(request, "side_plank")

# Simplified Live Feed Views
def _live_feed_response(exercise_name):
    config = EXERCISE_CONFIG[exercise_name]
    return StreamingHttpResponse(generate_live_feed(0, config['model'], config['rule_logic'], config['initial_state'], exercise_name.replace('_', ' ').title()), content_type='multipart/x-mixed-replace; boundary=frame')

def live_pushup(request): return _live_feed_response("pushup")
def live_plank(request): return _live_feed_response("plank")
def live_squat(request): return _live_feed_response("squat")
def live_situp(request): return _live_feed_response("situp")
def live_jumping_jacks(request): return _live_feed_response("jumping_jacks")
def live_reverse_plank(request): return _live_feed_response("reverse_plank")
def live_side_plank(request): return _live_feed_response("side_plank")

# --- Home Page View ---
def home(request):
    """Renders the main page, passing the load status of all models."""
    context = {f"{name}_model_loaded": config["model"] is not None for name, config in EXERCISE_CONFIG.items()}
    return render(request, 'analyzer/home.html', context)