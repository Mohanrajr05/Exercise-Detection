import os
import cv2
import time
import numpy as np
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
UPLOAD_DIR.mkdir(exist_ok=True)

# --- MEDIAPIPE INITIALIZATION ---
pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, landmark_type):
    """Safely extract x, y coordinates from a landmark."""
    lm = landmarks[landmark_type.value]
    return [lm.x, lm.y]

def get_landmark(landmarks, landmark_type):
    """Get raw landmark object."""
    return landmarks[landmark_type.value]

def is_body_horizontal(landmarks):
    """Check if body is in horizontal position (for pushup/plank)."""
    try:
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        
        # In horizontal position, shoulder and hip should have similar Y values
        # Relaxed check - only need shoulder and hip alignment
        shoulder_hip_diff = abs(shoulder.y - hip.y)
        
        # If Y difference is small, body is relatively horizontal
        return shoulder_hip_diff < 0.25
    except (IndexError, AttributeError):
        return False

def is_body_vertical(landmarks):
    """Check if body is in vertical/standing position."""
    try:
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # In standing position, shoulder should be above hip, hip above ankle
        # Y values: smaller = higher position
        is_upright = shoulder.y < hip.y < ankle.y
        
        # Check vertical alignment (X positions should be similar)
        shoulder_hip_x_diff = abs(shoulder.x - hip.x)
        
        return is_upright and shoulder_hip_x_diff < 0.2
    except (IndexError, AttributeError):
        return False

def is_lying_down(landmarks):
    """Check if person is lying on back (for situp)."""
    try:
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        
        # When lying down, shoulder and hip Y values are similar
        # and shoulder X should be different from hip X (body extended horizontally)
        y_diff = abs(shoulder.y - hip.y)
        x_diff = abs(shoulder.x - hip.x)
        
        return y_diff < 0.15 and x_diff > 0.1
    except (IndexError, AttributeError):
        return False

def landmarks_visible(landmarks, required_landmarks, threshold=0.3):
    """Check if all required landmarks have reasonable visibility."""
    try:
        for lm_type in required_landmarks:
            lm = landmarks[lm_type.value]
            if lm.visibility < threshold:
                return False
        return True
    except (IndexError, AttributeError):
        return False


# --- IMPROVED RULE-BASED LOGIC ---

def update_pushup_state(landmarks, state):
    """
    Improved push-up detection with strict position validation.
    Requires:
    1. Body in horizontal position
    2. Wrists below shoulders (hands on ground)
    3. Proper elbow angle transition
    """
    state['feedback'] = []
    
    # Required landmarks for pushup (core upper body only - ankle often out of frame)
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP
    ]
    
    try:
        # Check landmark visibility
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        # Get coordinates (only need upper body + hip for pushup)
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        elbow = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        wrist = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        
        shoulder_coords = [shoulder.x, shoulder.y]
        elbow_coords = [elbow.x, elbow.y]
        wrist_coords = [wrist.x, wrist.y]
        
        # VALIDATION 1: Check if body is horizontal (pushup position)
        # Shoulder and hip should be at similar Y levels
        if not is_body_horizontal(landmarks):
            state['feedback'].append("Get into pushup position")
            state['in_position'] = False
            return state
        
        # VALIDATION 2: Check if hands are supporting body
        # In pushup, wrists should be near or below shoulder level
        if wrist.y < shoulder.y - 0.15:  # Wrist significantly above shoulder = not in pushup
            state['feedback'].append("Place hands on ground")
            state['in_position'] = False
            return state
        
        state['in_position'] = True
        state['feedback'].append("In position ✓")
        
        # REP COUNTING: Track elbow angle
        elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
        
        # Require consecutive frames to prevent noise
        if 'frame_count_down' not in state:
            state['frame_count_down'] = 0
            state['frame_count_up'] = 0
        
        if elbow_angle < 90:  # Down position
            state['frame_count_down'] += 1
            state['frame_count_up'] = 0
            if state['frame_count_down'] >= 3 and not state['is_down']:
                state['is_down'] = True
                state['feedback'].append("Good depth!")
        elif elbow_angle > 150:  # Up position
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 3 and state['is_down']:
                state['count'] += 1
                state['is_down'] = False
                state['feedback'].append("Great Rep!")
        else:
            state['feedback'].append(f"Elbow: {int(elbow_angle)}°")

    except (IndexError, AttributeError):
        state['feedback'].append("Position not detected")
    
    return state


def update_plank_state(landmarks, state, is_reverse=False, is_side=False):
    """
    Improved plank detection with position validation.
    Requires body to be horizontal and straight.
    """
    state['feedback'] = []
    
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_ELBOW
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
            return state
        
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        elbow = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        
        # VALIDATION 1: Body must be horizontal
        if not is_body_horizontal(landmarks):
            state['feedback'].append("Get into plank position")
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
            return state
        
        # VALIDATION 2: Check body alignment
        body_angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y],
            [ankle.x, ankle.y]
        )
        
        # Different thresholds for plank types
        if is_side:
            angle_valid = 140 < body_angle < 220
        elif is_reverse:
            angle_valid = 150 < body_angle < 210
        else:
            angle_valid = 155 < body_angle < 205
        
        if angle_valid:
            state['feedback'].append("Holding form correctly ✓")
            if state['start_time'] is None:
                state['start_time'] = time.time()
            state['duration'] = state['cumulative_time'] + (time.time() - state['start_time'])
        else:
            if body_angle < 155:
                state['feedback'].append("Raise your hips")
            else:
                state['feedback'].append("Lower your hips")
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
                
    except (IndexError, AttributeError):
        if state['start_time'] is not None:
            state['cumulative_time'] += time.time() - state['start_time']
            state['start_time'] = None
    
    return state


def update_squat_state(landmarks, state):
    """
    Improved squat detection.
    Requires person to be standing upright before counting.
    """
    state['feedback'] = []
    
    required = [
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        knee = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        
        # VALIDATION 1: Must be in standing position (not sitting/lying)
        if not is_body_vertical(landmarks):
            state['feedback'].append("Stand upright to begin")
            return state
        
        knee_angle = calculate_angle(
            [hip.x, hip.y],
            [knee.x, knee.y],
            [ankle.x, ankle.y]
        )
        
        # Initialize frame counters for noise reduction
        if 'frame_count_down' not in state:
            state['frame_count_down'] = 0
            state['frame_count_up'] = 0
        
        # Check squat depth: hip should go below knee level
        hip_below_knee = hip.y > knee.y
        
        if knee_angle < 100 and hip_below_knee:
            state['frame_count_down'] += 1
            state['frame_count_up'] = 0
            if state['frame_count_down'] >= 3 and not state['is_down']:
                state['is_down'] = True
                state['feedback'].append("Good depth!")
        elif knee_angle > 160:
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 3 and state['is_down']:
                state['count'] += 1
                state['is_down'] = False
                state['feedback'].append("Good Squat!")
        else:
            if not state['is_down'] and knee_angle < 140:
                state['feedback'].append("Go deeper!")
            else:
                state['feedback'].append(f"Knee angle: {int(knee_angle)}°")

    except (IndexError, AttributeError):
        state['feedback'].append("Position not detected")
    
    return state


def update_situp_state(landmarks, state):
    """
    Improved sit-up detection.
    Requires person to be lying down first.
    """
    state['feedback'] = []
    
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        knee = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        
        hip_angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y],
            [knee.x, knee.y]
        )
        
        # VALIDATION: Check if in lying/situp position (not standing)
        # In situp, the person should be horizontal or nearly so
        if is_body_vertical(landmarks):
            state['feedback'].append("Lie down on your back")
            return state
        
        # Initialize frame counters
        if 'frame_count_up' not in state:
            state['frame_count_up'] = 0
            state['frame_count_down'] = 0
        
        if hip_angle < 90:  # Up position (crunched)
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 3 and not state['is_down']:
                state['is_down'] = True
        elif hip_angle > 150:  # Down position (lying flat)
            state['frame_count_down'] += 1
            state['frame_count_up'] = 0
            if state['frame_count_down'] >= 3 and state['is_down']:
                state['is_down'] = False
                state['count'] += 1
                state['feedback'].append("Nice Rep!")
        else:
            state['feedback'].append(f"Hip angle: {int(hip_angle)}°")

    except (IndexError, AttributeError):
        state['feedback'].append("Position not detected")
    
    return state


def update_jumping_jacks_state(landmarks, state):
    """
    Improved jumping jacks detection.
    Requires standing position with visible full body.
    """
    state['feedback'] = []
    
    required = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_WRIST
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        # VALIDATION: Must be standing
        if not is_body_vertical(landmarks):
            state['feedback'].append("Stand upright to begin")
            return state
        
        shoulder_l = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        shoulder_r = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        ankle_l = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        ankle_r = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        wrist_l = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        wrist_r = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        
        shoulder_width = abs(shoulder_l.x - shoulder_r.x)
        ankle_dist = abs(ankle_l.x - ankle_r.x)
        
        # Check both arms
        left_arm_up = wrist_l.y < shoulder_l.y
        right_arm_up = wrist_r.y < shoulder_r.y
        arms_are_up = left_arm_up and right_arm_up
        
        # Initialize frame counters
        if 'frame_count_up' not in state:
            state['frame_count_up'] = 0
            state['frame_count_down'] = 0
        
        # "Up" position: feet wide and BOTH arms up
        if ankle_dist > shoulder_width * 1.0 and arms_are_up:
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 2 and not state['is_up']:
                state['is_up'] = True
        # "Down" position: feet together and arms down
        elif ankle_dist < shoulder_width * 0.6 and not arms_are_up:
            state['frame_count_down'] += 1
            state['frame_count_up'] = 0
            if state['frame_count_down'] >= 2 and state['is_up']:
                state['is_up'] = False
                state['count'] += 1
                state['feedback'].append("Good Jack!")
        else:
            if state['is_up']:
                if not arms_are_up:
                    state['feedback'].append("Keep arms up!")
            else:
                if arms_are_up:
                    state['feedback'].append("Spread your legs!")

    except (IndexError, AttributeError):
        state['feedback'].append("Position not detected")
    
    return state


# --- EXERCISE CONFIGURATION ---
EXERCISE_CONFIG = {
    "pushup": {
        "rule_logic": update_pushup_state,
        "initial_state": {'count': 0, 'is_down': False, 'in_position': False, 'feedback': [], 'frame_count_down': 0, 'frame_count_up': 0}
    },
    "plank": {
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_reverse=False, is_side=False),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    },
    "squat": {
        "rule_logic": update_squat_state,
        "initial_state": {'count': 0, 'is_down': False, 'feedback': [], 'frame_count_down': 0, 'frame_count_up': 0}
    },
    "situp": {
        "rule_logic": update_situp_state,
        "initial_state": {'count': 0, 'is_down': False, 'feedback': [], 'frame_count_up': 0, 'frame_count_down': 0}
    },
    "jumping_jacks": {
        "rule_logic": update_jumping_jacks_state,
        "initial_state": {'count': 0, 'is_up': False, 'feedback': [], 'frame_count_up': 0, 'frame_count_down': 0}
    },
    "reverse_plank": {
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_reverse=True),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    },
    "side_plank": {
        "rule_logic": lambda lm, st: update_plank_state(lm, st, is_side=True),
        "initial_state": {'duration': 0, 'start_time': None, 'cumulative_time': 0, 'feedback': []}
    }
}

# --- VIDEO ANALYSIS ---
def analyze_video(filepath, rule_logic_func, initial_rule_state):
    """Analyzes a video using rule-based logic."""
    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return {'error': 'Cannot open video file.'}

    rule_state = initial_rule_state.copy()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        if results.pose_landmarks:
            rule_state = rule_logic_func(results.pose_landmarks.landmark, rule_state)

    cap.release()

    final_results = {'status': 'success'}
    if 'count' in rule_state:
        final_results['reps'] = rule_state['count']
    if 'duration' in rule_state:
        final_results['duration_seconds'] = round(rule_state.get('cumulative_time', rule_state.get('duration', 0)), 2)

    return final_results

# --- LIVE FEED GENERATOR ---
def generate_live_feed(camera_index, rule_logic_func, initial_rule_state, exercise_name):
    """Generator for live exercise analysis with real-time feedback."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        return

    rule_state = initial_rule_state.copy()

    while True:
        success, frame = camera.read()
        if not success:
            break

        display_frame = cv2.resize(frame.copy(), (640, 480))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        
        if results.pose_landmarks:
            rule_state = rule_logic_func(results.pose_landmarks.landmark, rule_state)
            draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                         landmark_drawing_spec=get_default_pose_landmarks_style())
        
        # Display exercise name
        cv2.putText(display_frame, exercise_name.replace('_', ' ').title(), (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Display count or duration
        if 'count' in rule_state:
            rule_text = f"Reps: {rule_state['count']}"
        elif 'duration' in rule_state:
            duration = int(rule_state.get('duration', 0))
            rule_text = f"Duration: {duration}s"
        else:
            rule_text = ""
        cv2.putText(display_frame, rule_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Display real-time feedback
        y_pos = 110
        for msg in rule_state.get('feedback', []):
            color = (0, 255, 0) if any(word in msg for word in ["Good", "Great", "Nice", "Holding", "✓"]) else (0, 165, 255)
            cv2.putText(display_frame, msg, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

# --- DJANGO VIEWS ---
def _video_upload_handler(request, exercise_name):
    """Generic handler for video upload POST requests."""
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file provided.'}, status=400)

    config = EXERCISE_CONFIG.get(exercise_name)
    if not config:
        return JsonResponse({'error': f'Invalid exercise: {exercise_name}'}, status=400)

    video = request.FILES['video']
    fs = FileSystemStorage(location=UPLOAD_DIR)
    filepath = UPLOAD_DIR / fs.save(video.name, video)
    
    try:
        results = analyze_video(filepath, config["rule_logic"], config["initial_state"])
        results['exercise_type'] = exercise_name
        return JsonResponse(results)
    except Exception as e:
        return JsonResponse({'error': f'Analysis error: {str(e)}'}, status=500)
    finally:
        if filepath.exists():
            os.remove(filepath)

# Video Upload Views
@csrf_exempt
def upload_and_analyze_pushup(request):
    return _video_upload_handler(request, "pushup")

@csrf_exempt
def upload_and_analyze_plank(request):
    return _video_upload_handler(request, "plank")

@csrf_exempt
def upload_and_analyze_squat(request):
    return _video_upload_handler(request, "squat")

@csrf_exempt
def upload_and_analyze_situp(request):
    return _video_upload_handler(request, "situp")

@csrf_exempt
def upload_and_analyze_jumping_jacks(request):
    return _video_upload_handler(request, "jumping_jacks")

@csrf_exempt
def upload_and_analyze_reverse_plank(request):
    return _video_upload_handler(request, "reverse_plank")

@csrf_exempt
def upload_and_analyze_side_plank(request):
    return _video_upload_handler(request, "side_plank")

# Live Feed Views
def _live_feed_response(exercise_name):
    config = EXERCISE_CONFIG[exercise_name]
    return StreamingHttpResponse(
        generate_live_feed(0, config['rule_logic'], config['initial_state'], exercise_name),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def live_pushup(request):
    return _live_feed_response("pushup")

def live_plank(request):
    return _live_feed_response("plank")

def live_squat(request):
    return _live_feed_response("squat")

def live_situp(request):
    return _live_feed_response("situp")

def live_jumping_jacks(request):
    return _live_feed_response("jumping_jacks")

def live_reverse_plank(request):
    return _live_feed_response("reverse_plank")

def live_side_plank(request):
    return _live_feed_response("side_plank")

# Home Page
def home(request):
    """Renders the main page."""
    return render(request, 'analyzer/home.html')

# --- BROWSER-BASED FRAME ANALYSIS ---
# Session-based state storage for each user
import json
import copy

@csrf_exempt
def reset_analysis_state(request):
    """Reset the exercise state for a new session."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            exercise = data.get('exercise', 'pushup')
        except:
            exercise = request.POST.get('exercise', 'pushup')
        
        config = EXERCISE_CONFIG.get(exercise)
        if config:
            # Store fresh state in session
            request.session[f'{exercise}_state'] = copy.deepcopy(config['initial_state'])
            request.session.modified = True
            return JsonResponse({'status': 'reset', 'exercise': exercise})
        return JsonResponse({'error': 'Invalid exercise'}, status=400)
    return JsonResponse({'error': 'POST required'}, status=405)

@csrf_exempt  
def analyze_live_frame(request):
    """Analyze a single frame sent from the browser."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    if 'frame' not in request.FILES:
        return JsonResponse({'error': 'No frame provided'}, status=400)
    
    exercise = request.POST.get('exercise', 'pushup')
    config = EXERCISE_CONFIG.get(exercise)
    if not config:
        return JsonResponse({'error': f'Invalid exercise: {exercise}'}, status=400)
    
    try:
        # Read image from request
        frame_file = request.FILES['frame']
        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Could not decode frame'}, status=400)
        
        # Get or initialize state from session
        state_key = f'{exercise}_state'
        if state_key not in request.session:
            request.session[state_key] = copy.deepcopy(config['initial_state'])
        
        state = request.session[state_key]
        
        # Process frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        
        if results.pose_landmarks:
            # Run exercise detection
            state = config['rule_logic'](results.pose_landmarks.landmark, state)
            # Save updated state to session
            request.session[state_key] = state
            request.session.modified = True
        else:
            state['feedback'] = ['No pose detected - ensure full body is visible']
        
        # Build response
        response = {'status': 'success', 'feedback': state.get('feedback', [])}
        
        if 'count' in state:
            response['reps'] = state['count']
        if 'duration' in state:
            response['duration'] = round(state.get('duration', 0), 1)
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)