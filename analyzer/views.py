import os
import tempfile
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import mediapipe as mp
from tensorflow.keras.models import load_model # Keras load_model
from django.shortcuts import render
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import time

# Configuration for models (must match train_models.py)
MODEL_CONFIG = {
    'frame_count': 16, # From train_models.py CONFIG
    'image_size': (160, 160) # From train_models.py CONFIG
}

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploaded_videos'
MODEL_DIR = BASE_DIR / 'models' # Assuming models are in <project_root>/models/

UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True) # Should be created by train_models.py

# --- Load Models ---
def try_load_model(model_path_str):
    path = Path(model_path_str)
    if path.exists():
        try:
            # Suppress TensorFlow INFO messages if desired
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed
            # tf.get_logger().setLevel('WARNING')

            # For .h5 models, you might need custom_objects if you have custom layers/losses
            # For .keras format, usually straightforward.
            return load_model(path, compile=False) # Set compile=False if you don't need to resume training
        except Exception as e:
            print(f"Error loading model {path}: {e}")
            return None
    else:
        print(f"Model file not found: {path}")
        return None

# Use .keras as saved by the improved train_models.py
PUSHUP_MODEL_PATH = MODEL_DIR / "pushup_classification_model.keras"
PLANK_MODEL_PATH = MODEL_DIR / "plank_classification_model.keras"
SITUP_MODEL_PATH = MODEL_DIR / "situp_classification_model.keras"
SQUAT_MODEL_PATH = MODEL_DIR / "squat_classification_model.keras"
JUMPING_JACKS_MODEL_PATH = MODEL_DIR / "jumping_jacks_classification_model.keras"
REVERSE_PLANK_MODEL_PATH = MODEL_DIR / "reverse_plank_classification_model.keras"
SIDE_PLANK_MODEL_PATH = MODEL_DIR / "side_plank_classification_model.keras"

# newly updated (10-06-2025)

pushup_model = try_load_model(PUSHUP_MODEL_PATH)
plank_model = try_load_model(PLANK_MODEL_PATH)
situp_model = try_load_model(SITUP_MODEL_PATH)
squat_model = try_load_model(SQUAT_MODEL_PATH)
jumping_jacks_model = try_load_model(JUMPING_JACKS_MODEL_PATH)
reverse_plank_model = try_load_model(REVERSE_PLANK_MODEL_PATH)
side_plank_model = try_load_model(SIDE_PLANK_MODEL_PATH)

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Helper Functions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c) # x, y coordinates
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def preprocess_frames_for_model(frames_list):
    """
    Prepares a list of frames for model prediction.
    Frames should be raw BGR frames from OpenCV.
    """
    processed_frames = []
    for frame in frames_list:
        resized_frame = cv2.resize(frame, MODEL_CONFIG['image_size'])
        normalized_frame = resized_frame.astype('float32') / 255.0
        processed_frames.append(normalized_frame)
    
    # Ensure we have exactly frame_count frames, pad if necessary (though less likely for live)
    while len(processed_frames) < MODEL_CONFIG['frame_count']:
        processed_frames.append(np.zeros((*MODEL_CONFIG['image_size'], 3), dtype=np.float32))

    return np.expand_dims(np.array(processed_frames[:MODEL_CONFIG['frame_count']]), axis=0)


# --- Shared Analysis Logic for Uploaded Videos ---
def analyze_uploaded_video_with_model(filepath, exercise_model, exercise_name, rule_based_analyzer):
    """
    Analyzes an uploaded video using both the Keras model and rule-based logic.
    filepath: Path to the video file.
    exercise_model: Loaded Keras model for the specific exercise.
    exercise_name: String name of the exercise (e.g., "pushup").
    rule_based_analyzer: Function that performs rule-based analysis (returns dict with counts/duration).
    """
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return {'error': f'Cannot open video file: {filepath}'}, 0 # Added 0 for model_detected_count

    frames_buffer = []
    model_detected_exercise_segments = 0
    total_segments_processed = 0
    
    # --- First pass: Keras model classification of segments ---
    if exercise_model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use original frame for model input preprocessing
            frames_buffer.append(frame.copy())

            if len(frames_buffer) == MODEL_CONFIG['frame_count']:
                total_segments_processed += 1
                # Preprocess the buffer for the model
                model_input = preprocess_frames_for_model(frames_buffer)
                prediction = exercise_model.predict(model_input, verbose=0) # verbose=0
                
                # Assuming binary classification: class 0 = not_exercise, class 1 = exercise
                # And labels in train_models.py are {"not_exercise": 0, "exercise": 1}
                predicted_class_index = np.argmax(prediction[0])
                
                if predicted_class_index == 1: # 1 is the 'exercise' class
                    model_detected_exercise_segments += 1
                
                frames_buffer.pop(0) # Slide window: remove the oldest frame
        
    cap.release() # Release for the first pass

    # --- Second pass: Rule-based analysis (re-open video) ---
    # This is inefficient but simpler than trying to combine loops perfectly for now.
    # For production, you'd ideally process frames once.
    rule_based_results = rule_based_analyzer(str(filepath))


    model_confidence_percentage = 0
    if total_segments_processed > 0:
        model_confidence_percentage = (model_detected_exercise_segments / total_segments_processed) * 100

    final_response = {
        'exercise_type': exercise_name,
        'model_exercise_detection_segments': f"{model_detected_exercise_segments}/{total_segments_processed}",
        'model_exercise_confidence_percent': round(model_confidence_percentage, 2),
        **rule_based_results # Merge rule-based results
    }
    
    if not exercise_model:
        final_response['model_status'] = f"{exercise_name.capitalize()} model not loaded. Analysis is rule-based only."
        
    return final_response


# --- Rule-based analyzer functions (extracted from original upload_and_analyze_*) ---
def rule_based_pushup_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    pushup_counter = 0
    pushup_state = False # False = up, True = down
    consecutive_frames = 0
    min_frames_for_state_change = 2 # Require 2 frames to confirm state change

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                if elbow_angle < 90 and not pushup_state: # Moving to 'down' state
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        pushup_state = True
                        consecutive_frames = 0
                elif elbow_angle > 160 and pushup_state: # Moving to 'up' state, rep completed
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        pushup_state = False
                        pushup_counter += 1
                        consecutive_frames = 0
                elif not (elbow_angle < 90 or elbow_angle > 160) : # Reset consecutive frames if in intermediate angle
                    consecutive_frames = 0

            except Exception: # Catch errors if landmarks are not visible
                consecutive_frames = 0 # Reset on error
                pass
    cap.release()
    return {'pushup_count': pushup_counter}

def rule_based_plank_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    plank_frames_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default if fps not available

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Check body alignment (angle close to 180 degrees)
                body_straight_angle = calculate_angle(shoulder_l, hip_l, ankle_l)
                
                # More robust check for hip height relative to shoulders and ankles (Y-coordinates)
                # This assumes camera is roughly side-on.
                # A simple check: hip_y should be between shoulder_y and ankle_y, or very close.
                # A stricter check involves ensuring the body is relatively parallel to the ground.
                # For simplicity, we primarily rely on the body_straight_angle.
                # A tolerance for the y-coordinates can be added:
                # e.g. abs(hip_l[1] - (shoulder_l[1] + ankle_l[1]) / 2) < some_threshold_normalized_by_body_height
                
                if body_straight_angle > 160 and body_straight_angle < 200: # Allow some tolerance around 180
                     # Add a check for hip relative to shoulder y-coordinate (hips not sagging or too high)
                     # This is a heuristic and may need tuning
                    if abs(shoulder_l[1] - hip_l[1]) < 0.15 : # Assuming y is normalized 0-1, 0.15 is a guess
                        plank_frames_count +=1

            except Exception:
                pass
    cap.release()
    plank_seconds = round(plank_frames_count / fps, 2)
    return {'plank_duration_seconds': plank_seconds}


def rule_based_situp_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    situp_counter = 0
    situp_state = False  # False = down (lying), True = up (sitting)
    consecutive_frames = 0
    min_frames_for_state_change = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip_angle = calculate_angle(shoulder, hip, knee) # Angle at the hip

                if hip_angle < 90 and not situp_state: # Moving to 'up' state
                    consecutive_frames +=1
                    if consecutive_frames >= min_frames_for_state_change:
                        situp_state = True
                        consecutive_frames = 0
                elif hip_angle > 150 and situp_state: # Moving to 'down' state, rep completed
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        situp_state = False
                        situp_counter += 1
                        consecutive_frames = 0
                elif not (hip_angle < 90 or hip_angle > 150) :
                    consecutive_frames = 0
            except Exception:
                consecutive_frames = 0
                pass
    cap.release()
    return {'situp_count': situp_counter}

def rule_based_squat_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    squat_counter = 0
    squat_state = False  # False = up (standing), True = down (squatting)
    consecutive_frames = 0
    min_frames_for_state_change = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                knee_angle = calculate_angle(hip, knee, ankle)

                if knee_angle < 90 and not squat_state : # Moving to 'down' state
                    consecutive_frames +=1
                    if consecutive_frames >= min_frames_for_state_change:
                        squat_state = True
                        consecutive_frames = 0
                elif knee_angle > 160 and squat_state: # Moving to 'up' state, rep completed
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        squat_state = False
                        squat_counter += 1
                        consecutive_frames = 0
                elif not (knee_angle < 90 or knee_angle > 160) :
                     consecutive_frames = 0
            except Exception:
                consecutive_frames = 0
                pass
    cap.release()
    return {'squat_count': squat_counter}

# newly updated (10-06-2025)

def rule_based_jumping_jacks_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    jj_counter = 0
    # State: False = legs together (down), True = legs apart (up)
    jj_state = False
    consecutive_frames = 0
    min_frames_for_state_change = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                # Get landmarks for shoulders, hips, and ankles
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                ankle_l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                ankle_r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                # Calculate horizontal distance between ankles normalized by shoulder width
                shoulder_width = abs(shoulder_l.x - shoulder_r.x)
                ankle_dist = abs(ankle_l.x - ankle_r.x)
                
                # Check if arms are up (y-coordinate of wrist is less than shoulder)
                arms_are_up = wrist_l.y < shoulder_l.y

                # Condition for "up" state: ankles are wide apart and arms are up
                if ankle_dist > shoulder_width * 0.8 and arms_are_up and not jj_state:
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        jj_state = True
                        consecutive_frames = 0
                # Condition for "down" state: ankles are close together
                elif ankle_dist < shoulder_width * 0.4 and jj_state:
                    consecutive_frames += 1
                    if consecutive_frames >= min_frames_for_state_change:
                        jj_state = False
                        jj_counter += 1 # Count rep on returning to down state
                        consecutive_frames = 0
                else:
                    consecutive_frames = 0
            except Exception:
                consecutive_frames = 0
                pass
    cap.release()
    return {'jumping_jacks_count': jj_counter}


def rule_based_reverse_plank_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    plank_frames_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                # Use left side landmarks for consistency
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                body_angle = calculate_angle(shoulder, hip, ankle)

                # For reverse plank, body should be straight (angle ~180)
                if body_angle > 150 and body_angle < 210:
                    plank_frames_count += 1
            except Exception:
                pass
    cap.release()
    plank_seconds = round(plank_frames_count / fps, 2)
    return {'reverse_plank_duration_seconds': plank_seconds}


def rule_based_side_plank_analyzer(video_path_str):
    cap = cv2.VideoCapture(video_path_str)
    plank_frames_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                # Check both left and right sides, and see which one is more visible/straight
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                body_angle_left = calculate_angle(shoulder_l, hip_l, ankle_l)
                body_angle_right = calculate_angle(shoulder_r, hip_r, ankle_r)

                # A good side plank has a straight body on the visible side
                if (body_angle_left > 150 and body_angle_left < 210) or \
                   (body_angle_right > 150 and body_angle_right < 210):
                    plank_frames_count += 1
            except Exception:
                pass
    cap.release()
    plank_seconds = round(plank_frames_count / fps, 2)
    return {'side_plank_duration_seconds': plank_seconds}

# --- Django Views for Upload ---
@csrf_exempt
def upload_and_analyze_pushup(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, pushup_model, "pushup", rule_based_pushup_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            print(f"Error analyzing pushup video: {e}")
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists():
                 os.remove(filepath) # Clean up uploaded file
    return render(request, 'analyzer/home.html') # Or appropriate template

@csrf_exempt
def upload_and_analyze_plank(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, plank_model, "plank", rule_based_plank_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            print(f"Error analyzing plank video: {e}")
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists():
                 os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def upload_and_analyze_situp(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, situp_model, "situp", rule_based_situp_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            print(f"Error analyzing situp video: {e}")
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists():
                 os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def upload_and_analyze_squat(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, squat_model, "squat", rule_based_squat_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            print(f"Error analyzing squat video: {e}")
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists():
                 os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def upload_and_analyze_jumping_jacks(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, jumping_jacks_model, "jumping_jacks", rule_based_jumping_jacks_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists(): os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def upload_and_analyze_reverse_plank(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, reverse_plank_model, "reverse_plank", rule_based_reverse_plank_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists(): os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def upload_and_analyze_side_plank(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename
        try:
            analysis_results = analyze_uploaded_video_with_model(filepath, side_plank_model, "side_plank", rule_based_side_plank_analyzer)
            return JsonResponse(analysis_results)
        except Exception as e:
            return JsonResponse({'error': f'Error analyzing the video: {str(e)}'}, status=500)
        finally:
            if filepath.exists(): os.remove(filepath)
    return JsonResponse({'error': 'Invalid request'}, status=400)


# --- Live Analysis (Generator Functions) ---
def generate_live_frames(camera_index, exercise_model, exercise_name_caps, rule_based_counter_logic):
    """
    Generic live frame generator for exercises.
    camera_index: Index of the camera (e.g., 0).
    exercise_model: Loaded Keras model.
    exercise_name_caps: Capitalized name for display (e.g., "Pushup").
    rule_based_counter_logic: Function taking (landmarks, state_vars_dict) and returning (count, new_state_vars_dict, display_text_for_counter).
                               state_vars_dict stores things like 'counter', 'state', 'consecutive_frames'.
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        # Yield a single frame with an error message
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Cannot open camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    frames_buffer_live = []
    model_prediction_text = f"Loading {exercise_name_caps} Model..."
    model_prediction_color = (200, 200, 0) # Yellow for loading/unknown

    # Initialize state for rule-based counter
    # Example for pushups: {'counter': 0, 'state': False, 'consecutive_frames': 0, 'min_frames_for_state_change': 2}
    # This needs to be adapted per exercise if rule_based_counter_logic is generic
    # For now, specific gen functions will call this with their specific logic.
    
    # For simplicity, the original structure of gen_*_frames is largely kept,
    # but they will call a shared drawing and model prediction part.
    
    # This generic function is a bit too complex to directly replace the existing gen_*_frames
    # without significant refactoring of their internal state management.
    # Instead, I will augment the existing gen_*_frames functions.
    print(f"Camera {camera_index} opened for {exercise_name_caps}") # Debug
    
    # Placeholder, this function is not used directly, but its concepts are integrated below
    camera.release()


def gen_frames(): # Pushup Live
    camera = cv2.VideoCapture(0)
    pushup_counter = 0
    pushup_state = False # False = up, True = down
    consecutive_frames_rule = 0
    min_frames_for_state_change = 2
    
    frames_for_model_buffer = []
    model_label = "Analyzing..."
    model_label_color = (0, 255, 255) # Yellow

    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame from camera for pushup")
            break
        
        processed_frame = frame.copy() # For display
        processed_frame = cv2.resize(processed_frame, (640, 480))
        
        # Model prediction part
        frames_for_model_buffer.append(frame.copy()) # Use original frame for model
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if pushup_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = pushup_model.predict(model_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                if predicted_class == 1: # Assuming 1 is "pushup"
                    model_label = "Pushup Detected"
                    model_label_color = (0, 255, 0) # Green
                else:
                    model_label = "Not Pushup"
                    model_label_color = (0, 0, 255) # Red
            else:
                model_label = "Pushup Model N/A"
                model_label_color = (255, 0, 0)
            frames_for_model_buffer.pop(0) # Slide window

        # Rule-based counting part (MediaPipe)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                if elbow_angle < 90 and not pushup_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        pushup_state = True
                        consecutive_frames_rule = 0
                elif elbow_angle > 160 and pushup_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        pushup_state = False
                        pushup_counter += 1
                        consecutive_frames_rule = 0
                elif not (elbow_angle < 90 or elbow_angle > 160):
                     consecutive_frames_rule = 0
                
                draw_landmarks(
                    processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )
            except Exception:
                pass
        
        cv2.putText(processed_frame, f"Pushups: {pushup_counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, model_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)


        ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()
    print("Pushup camera released.")


def gen_plank_frames():
    camera = cv2.VideoCapture(0)
    plank_start_time = None
    plank_duration = 0
    
    frames_for_model_buffer = []
    model_label = "Analyzing..."
    model_label_color = (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = frame.copy()
        processed_frame = cv2.resize(processed_frame, (640, 480))

        # Model prediction part
        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if plank_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = plank_model.predict(model_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                if predicted_class == 1: # Plank
                    model_label = "Plank Detected"
                    model_label_color = (0, 255, 0)
                else:
                    model_label = "No Plank"
                    model_label_color = (0, 0, 255)
            else:
                model_label = "Plank Model N/A"
                model_label_color = (255,0,0)
            frames_for_model_buffer.pop(0)

        # Rule-based logic for plank duration
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        currently_planking = False
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                body_straight_angle = calculate_angle(shoulder_l, hip_l, ankle_l)

                if body_straight_angle > 160 and body_straight_angle < 200 and abs(shoulder_l[1] - hip_l[1]) < 0.15:
                    currently_planking = True
                    if plank_start_time is None:
                        plank_start_time = time.time()
                    plank_duration = int(time.time() - plank_start_time)
                else:
                    plank_start_time = None # Reset timer if form is broken
                    # Keep last duration or reset to 0? Let's reset if form breaks significantly
                    # plank_duration = 0 # Option: reset duration if not in plank
                
                draw_landmarks(
                    processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )
            except Exception:
                plank_start_time = None
        else: # No landmarks detected
            plank_start_time = None


        cv2.putText(processed_frame, model_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)
        rule_text = "Plank Rule: Active" if currently_planking else "Plank Rule: Inactive"
        rule_color = (0,255,0) if currently_planking else (0,0,255)
        cv2.putText(processed_frame, rule_text, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rule_color, 2)
        if plank_start_time is not None : # Only show duration if currently planking by rule
             cv2.putText(processed_frame, f"Duration: {plank_duration}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()


def gen_situp_frames():
    camera = cv2.VideoCapture(0)
    situp_counter = 0
    situp_state = False 
    consecutive_frames_rule = 0
    min_frames_for_state_change = 2
    
    frames_for_model_buffer = []
    model_label = "Analyzing..."
    model_label_color = (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = frame.copy()
        processed_frame = cv2.resize(processed_frame, (640, 480))

        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if situp_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = situp_model.predict(model_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                if predicted_class == 1: # Situp
                    model_label = "Situp Detected"
                    model_label_color = (0, 255, 0)
                else:
                    model_label = "Not Situp"
                    model_label_color = (0, 0, 255)
            else:
                model_label = "Situp Model N/A"
                model_label_color = (255,0,0)
            frames_for_model_buffer.pop(0)

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip_angle = calculate_angle(shoulder, hip, knee)

                if hip_angle < 90 and not situp_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        situp_state = True
                        consecutive_frames_rule = 0
                elif hip_angle > 150 and situp_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        situp_state = False
                        situp_counter += 1
                        consecutive_frames_rule = 0
                elif not (hip_angle < 90 or hip_angle > 150):
                    consecutive_frames_rule = 0
                
                draw_landmarks(
                    processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )
            except Exception:
                pass
        
        cv2.putText(processed_frame, f"Situps: {situp_counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, model_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)

        ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()


def gen_squat_frames():
    camera = cv2.VideoCapture(0)
    squat_counter = 0
    squat_state = False
    consecutive_frames_rule = 0
    min_frames_for_state_change = 2

    frames_for_model_buffer = []
    model_label = "Analyzing..."
    model_label_color = (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = frame.copy()
        processed_frame = cv2.resize(processed_frame, (640, 480))

        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if squat_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = squat_model.predict(model_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                if predicted_class == 1: # Squat
                    model_label = "Squat Detected"
                    model_label_color = (0, 255, 0)
                else:
                    model_label = "Not Squat"
                    model_label_color = (0, 0, 255)
            else:
                model_label = "Squat Model N/A"
                model_label_color = (255,0,0)
            frames_for_model_buffer.pop(0)

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                knee_angle = calculate_angle(hip, knee, ankle)

                if knee_angle < 90 and not squat_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        squat_state = True
                        consecutive_frames_rule = 0
                elif knee_angle > 160 and squat_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames_for_state_change:
                        squat_state = False
                        squat_counter += 1
                        consecutive_frames_rule = 0
                elif not (knee_angle < 90 or knee_angle > 160):
                     consecutive_frames_rule = 0
                
                draw_landmarks(
                    processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )
            except Exception:
                pass
        
        cv2.putText(processed_frame, f"Squats: {squat_counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, model_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)

        ret_enc, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

# In views.py, add these with the other gen_*_frames live functions

def gen_jumping_jacks_frames():
    camera = cv2.VideoCapture(0)
    jj_counter, jj_state, consecutive_frames_rule, min_frames = 0, False, 0, 2
    
    frames_for_model_buffer = []
    model_label, model_label_color = "Analyzing...", (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = cv2.resize(frame.copy(), (640, 480))

        # Model prediction part
        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if jumping_jacks_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = jumping_jacks_model.predict(model_input, verbose=0)
                if np.argmax(prediction[0]) == 1:
                    model_label, model_label_color = "Jumping Jacks", (0, 255, 0)
                else:
                    model_label, model_label_color = "Not Jumping Jacks", (0, 0, 255)
            else:
                model_label, model_label_color = "JJ Model N/A", (255,0,0)
            frames_for_model_buffer.pop(0)

        # Rule-based counting part
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                ankle_l = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                ankle_r = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                shoulder_width = abs(shoulder_l.x - shoulder_r.x)
                ankle_dist = abs(ankle_l.x - ankle_r.x)
                arms_are_up = wrist_l.y < shoulder_l.y

                if ankle_dist > shoulder_width * 0.8 and arms_are_up and not jj_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames:
                        jj_state = True
                        consecutive_frames_rule = 0
                elif ankle_dist < shoulder_width * 0.4 and jj_state:
                    consecutive_frames_rule += 1
                    if consecutive_frames_rule >= min_frames:
                        jj_state = False
                        jj_counter += 1
                        consecutive_frames_rule = 0
                else:
                    consecutive_frames_rule = 0
                draw_landmarks(processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception: pass
        
        cv2.putText(processed_frame, f"Jumping Jacks: {jj_counter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, model_label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()


def gen_reverse_plank_frames():
    # This logic is very similar to the regular plank, timing the duration
    camera = cv2.VideoCapture(0)
    plank_start_time = None
    plank_duration = 0
    
    frames_for_model_buffer = []
    model_label, model_label_color = "Analyzing...", (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = cv2.resize(frame.copy(), (640, 480))
        
        # Model prediction
        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if reverse_plank_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = reverse_plank_model.predict(model_input, verbose=0)
                if np.argmax(prediction[0]) == 1:
                    model_label, model_label_color = "Reverse Plank", (0, 255, 0)
                else:
                    model_label, model_label_color = "Not Reverse Plank", (0, 0, 255)
            else:
                model_label, model_label_color = "Model N/A", (255,0,0)
            frames_for_model_buffer.pop(0)

        # Rule-based timing
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        currently_planking = False
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                body_angle = calculate_angle(shoulder, hip, ankle)

                if body_angle > 150 and body_angle < 210:
                    currently_planking = True
                    if plank_start_time is None: plank_start_time = time.time()
                    plank_duration = int(time.time() - plank_start_time)
                else:
                    plank_start_time = None
                draw_landmarks(processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception: plank_start_time = None
        else:
            plank_start_time = None

        cv2.putText(processed_frame, model_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)
        if plank_start_time is not None:
             cv2.putText(processed_frame, f"Duration: {plank_duration}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()    
    
# gen_side_plank_frames would be very similar to gen_reverse_plank_frames, just calling side_plank_model
# and using its logic. For brevity, I'll omit the redundant live generator for side plank, 
# as its structure is identical to the reverse plank one. You can create it by copying 
# gen_reverse_plank_frames and replacing "reverse" with "side".

def gen_side_plank_frames():
    # This logic is very similar to the regular plank, timing the duration
    camera = cv2.VideoCapture(0)
    plank_start_time = None
    plank_duration = 0
    
    frames_for_model_buffer = []
    model_label, model_label_color = "Analyzing...", (0, 255, 255)

    while True:
        success, frame = camera.read()
        if not success: break
        
        processed_frame = cv2.resize(frame.copy(), (640, 480))
        
        # Model prediction
        frames_for_model_buffer.append(frame.copy())
        if len(frames_for_model_buffer) == MODEL_CONFIG['frame_count']:
            if side_plank_model:
                model_input = preprocess_frames_for_model(frames_for_model_buffer)
                prediction = side_plank_model.predict(model_input, verbose=0)
                if np.argmax(prediction[0]) == 1:
                    model_label, model_label_color = "Side Plank", (0, 255, 0)
                else:
                    model_label, model_label_color = "Not Side Plank", (0, 0, 255)
            else:
                model_label, model_label_color = "Model N/A", (255,0,0)
            frames_for_model_buffer.pop(0)

        # Rule-based timing
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        currently_planking = False
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                body_angle = calculate_angle(shoulder, hip, ankle)

                if body_angle > 150 and body_angle < 210:
                    currently_planking = True
                    if plank_start_time is None: plank_start_time = time.time()
                    plank_duration = int(time.time() - plank_start_time)
                else:
                    plank_start_time = None
                draw_landmarks(processed_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            except Exception: plank_start_time = None
        else:
            plank_start_time = None

        cv2.putText(processed_frame, model_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, model_label_color, 2)
        if plank_start_time is not None:
             cv2.putText(processed_frame, f"Duration: {plank_duration}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()


# --- URL Endpoints for Live Video ---
def live_pushup(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_situp(request):
    return StreamingHttpResponse(gen_situp_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_plank(request):
    return StreamingHttpResponse(gen_plank_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_squat(request):
    return StreamingHttpResponse(gen_squat_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_jumping_jacks(request):
    return StreamingHttpResponse(gen_jumping_jacks_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_reverse_plank(request):
    return StreamingHttpResponse(gen_reverse_plank_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_side_plank(request):
    return StreamingHttpResponse(gen_side_plank_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


# --- Home Page ---
def home(request):
    # You'll need to create/update this template to provide UIs for uploading videos
    # and viewing live streams for each exercise.
    return render(request, 'analyzer/home.html', {
        'pushup_model_loaded': pushup_model is not None,
        'plank_model_loaded': plank_model is not None,
        'situp_model_loaded': situp_model is not None,
        'squat_model_loaded': squat_model is not None,
        'jumping_jacks_model_loaded': jumping_jacks_model is not None,
        'reverse_plank_model_loaded': reverse_plank_model is not None,
        'side_plank_model_loaded': side_plank_model is not None,
    })