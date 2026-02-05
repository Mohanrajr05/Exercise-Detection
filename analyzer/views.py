import os
import copy
import cv2
import time
import numpy as np
import mediapipe as mp
import math
from pathlib import Path
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

# Import MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- GLOBAL CONFIGURATION ---
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploaded_videos'
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / 'models/pose_landmarker.task'

# --- MEDIAPIPE INITIALIZATION ---
# Create BaseOptions for the PoseLandmarker
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

# Create PoseLandmarkerOptions
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global Landmarker instance (replacement for pose_detector)
landmarker = vision.PoseLandmarker.create_from_options(options)

# --- REPLACEMENT FOR MP_POSE (Since we lost mp.solutions.pose) ---
class PoseLandmark:
    # Manual mapping of MediaPipe Pose landmarks
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class MP_POSE_COMPAT:
    PoseLandmark = PoseLandmark

mp_pose = MP_POSE_COMPAT

# --- CUSTOM DRAWING UTILS (Since we lost mp.solutions.drawing_utils) ---
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

def draw_landmarks(image, landmarks_list, connections=POSE_CONNECTIONS, landmark_drawing_spec=None):
    """Draws the landmarks and the connections on the image."""
    if not landmarks_list:
        return
        
    height, width, _ = image.shape
    
    # Draw connections
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        if start_idx >= len(landmarks_list) or end_idx >= len(landmarks_list):
            continue
            
        start_point = landmarks_list[start_idx]
        end_point = landmarks_list[end_idx]
        
        # Check visibility
        if hasattr(start_point, 'visibility') and start_point.visibility < 0.5:
            continue
        if hasattr(end_point, 'visibility') and end_point.visibility < 0.5:
            continue
            
        x1, y1 = int(start_point.x * width), int(start_point.y * height)
        x2, y2 = int(end_point.x * width), int(end_point.y * height)
        
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
    # Draw landmarks
    for lm in landmarks_list:
        if hasattr(lm, 'visibility') and lm.visibility < 0.5:
            continue
            
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
        cv2.circle(image, (cx, cy), 2, (255, 255, 255), -1)

# Dummy style getter just to satisfy imports if needed, though we won't use it
def get_default_pose_landmarks_style():
    return None

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle between three points (a-b-c)."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    # Vectors
    ba = a - b
    bc = c - b
    
    # Normalize
    ba_mag = np.linalg.norm(ba)
    bc_mag = np.linalg.norm(bc)
    
    # Avoid division by zero
    if ba_mag == 0 or bc_mag == 0:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (ba_mag * bc_mag)
    
    # Clip for float stability
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_landmark_coords(landmarks, landmark_idx):
    """Safely extract x, y coordinates from a landmark."""
    # landmark_idx is int value from PoseLandmark enum
    lm = landmarks[landmark_idx]
    return [lm.x, lm.y]

def get_landmark(landmarks, landmark_idx):
    """Get raw landmark object."""
    return landmarks[landmark_idx]

def is_body_horizontal(landmarks):
    """Check if body is in horizontal position (for pushup/plank)."""
    try:
        shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        
        shoulder_hip_diff = abs(shoulder.y - hip.y)
        return shoulder_hip_diff < 0.25
    except (IndexError, AttributeError):
        return False

def is_body_vertical(landmarks):
    """Check if body is in vertical/standing position."""
    try:
        shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        ankle = get_landmark(landmarks, PoseLandmark.LEFT_ANKLE)
        
        is_upright = shoulder.y < hip.y < ankle.y
        shoulder_hip_x_diff = abs(shoulder.x - hip.x)
        return is_upright and shoulder_hip_x_diff < 0.2
    except (IndexError, AttributeError):
        return is_upright and shoulder_hip_x_diff < 0.2
    except (IndexError, AttributeError):
        return False

def check_hip_vertical_position(shoulder, hip, ankle):
    """
    Determines if hips are 'low' (sagging) or 'high' (piking) relative to the body line.
    Returns: 'low', 'high', or 'ok' (if very close to line)
    
    Coordinate system: Y increases downwards (0 at top).
    """
    try:
        # Avoid division by zero
        if abs(ankle.x - shoulder.x) < 0.01:
            return 'ok' # Vertical body? Unlikely for plank.
            
        # Line equation: y - y1 = m * (x - x1)
        # Expected Y at hip.x position
        slope = (ankle.y - shoulder.y) / (ankle.x - shoulder.x)
        expected_y = shoulder.y + slope * (hip.x - shoulder.x)
        
        # Calculate deviation
        # Y is larger downwards. So if hip.y > expected_y, hip is BELOW the line (closer to floor if prone)
        deviation = hip.y - expected_y
        
        # Threshold for 'ok' could be small, but we rely on angle for validity.
        # This function is mainly for FEEDBACK direction when angle is INVALID.
        
        if deviation > 0: 
            return 'low' # Sagging (closer to floor)
        else:
            return 'high' # Piking (further from floor)
            
    except Exception:
        return 'low' # Default fallback

def is_lying_down(landmarks):
    """Check if person is lying on back (for situp)."""
    try:
        shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        
        y_diff = abs(shoulder.y - hip.y)
        x_diff = abs(shoulder.x - hip.x)
        return y_diff < 0.15 and x_diff > 0.1
    except (IndexError, AttributeError):
        return False

def landmarks_visible(landmarks, required_landmarks, threshold=0.3):
    """Check if all required landmarks have reasonable visibility."""
    try:
        for lm_idx in required_landmarks:
            lm = landmarks[lm_idx]
            if hasattr(lm, 'visibility') and lm.visibility < threshold:
                return False
        return True
    except (IndexError, AttributeError):
        return False


# --- IMPROVED RULE-BASED LOGIC ---

def update_pushup_state(landmarks, state):
    """
    Enhanced push-up detection with form quality tracking.
    Tracks per-rep metrics and provides corrective feedback.
    """
    state['feedback'] = []
    
    # Initialize rep tracking if not present
    if 'rep_details' not in state:
        state['rep_details'] = []  # List of per-rep feedback
    if 'current_rep_min_angle' not in state:
        state['current_rep_min_angle'] = 180  # Track lowest angle in current rep
    if 'current_rep_issues' not in state:
        state['current_rep_issues'] = []  # Issues detected during current rep
    
    try:
        # Get landmarks from both sides
        left_shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        left_elbow = get_landmark(landmarks, PoseLandmark.LEFT_ELBOW)
        left_wrist = get_landmark(landmarks, PoseLandmark.LEFT_WRIST)
        left_hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        left_ankle = get_landmark(landmarks, PoseLandmark.LEFT_ANKLE)
        
        right_shoulder = get_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER)
        right_elbow = get_landmark(landmarks, PoseLandmark.RIGHT_ELBOW)
        right_wrist = get_landmark(landmarks, PoseLandmark.RIGHT_WRIST)
        right_hip = get_landmark(landmarks, PoseLandmark.RIGHT_HIP)
        
        # Calculate visibility scores
        left_vis = sum([
            getattr(left_shoulder, 'visibility', 0) or 0,
            getattr(left_elbow, 'visibility', 0) or 0,
            getattr(left_wrist, 'visibility', 0) or 0,
            getattr(left_hip, 'visibility', 0) or 0
        ])
        right_vis = sum([
            getattr(right_shoulder, 'visibility', 0) or 0,
            getattr(right_elbow, 'visibility', 0) or 0,
            getattr(right_wrist, 'visibility', 0) or 0,
            getattr(right_hip, 'visibility', 0) or 0
        ])
        
        # Use the more visible side
        if left_vis >= right_vis:
            shoulder, elbow, wrist, hip, ankle = left_shoulder, left_elbow, left_wrist, left_hip, left_ankle
        else:
            shoulder, elbow, wrist, hip, ankle = right_shoulder, right_elbow, right_wrist, right_hip, get_landmark(landmarks, PoseLandmark.RIGHT_ANKLE)
        
        # Check minimum visibility
        min_vis = min(
            getattr(shoulder, 'visibility', 1) or 1,
            getattr(elbow, 'visibility', 1) or 1,
            getattr(wrist, 'visibility', 1) or 1
        )
        if min_vis < 0.1:
            state['feedback'].append("Ensure body is visible")
            return state
        
        shoulder_coords = [shoulder.x, shoulder.y]
        elbow_coords = [elbow.x, elbow.y]
        wrist_coords = [wrist.x, wrist.y]
        hip_coords = [hip.x, hip.y]
        
        state['in_position'] = True
        
        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
        
        # Track minimum angle during the rep (for depth assessment)
        if state.get('is_down', False) or elbow_angle < 130:
            state['current_rep_min_angle'] = min(state['current_rep_min_angle'], elbow_angle)
        
        # === FORM CHECKS (during the rep) ===
        
        # Check 1: Body alignment (hip sagging or piking)
        try:
            ankle_coords = [getattr(ankle, 'x', hip.x), getattr(ankle, 'y', hip.y + 0.3)]
            body_angle = calculate_angle(shoulder_coords, hip_coords, ankle_coords)
            
            if body_angle < 150:
                if "Hip sagging" not in state['current_rep_issues']:
                    state['current_rep_issues'].append("Hip sagging")
            elif body_angle > 200:
                if "Hips too high" not in state['current_rep_issues']:
                    state['current_rep_issues'].append("Hips too high")
        except:
            pass
        
        # Check 2: Elbow flare (elbows should stay close to body)
        # If wrist is far from shoulder horizontally, elbows may be flaring
        wrist_shoulder_x_diff = abs(wrist.x - shoulder.x)
        if wrist_shoulder_x_diff > 0.25:
            if "Elbows flaring out" not in state['current_rep_issues']:
                state['current_rep_issues'].append("Elbows flaring out")
        
        # Initialize frame counters
        if 'frame_count_down' not in state:
            state['frame_count_down'] = 0
            state['frame_count_up'] = 0
        
        # Thresholds
        DOWN_THRESHOLD = 110
        UP_THRESHOLD = 145
        FRAMES_REQUIRED = 2
        
        # Perfect depth threshold
        PERFECT_DEPTH = 90
        GOOD_DEPTH = 100
        
        if elbow_angle < DOWN_THRESHOLD:
            state['frame_count_down'] += 1
            state['frame_count_up'] = 0
            if state['frame_count_down'] >= FRAMES_REQUIRED and not state['is_down']:
                state['is_down'] = True
                
        elif elbow_angle > UP_THRESHOLD:
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            
            if state['frame_count_up'] >= FRAMES_REQUIRED and state['is_down']:
                # Rep completed - evaluate form
                state['count'] += 1
                rep_num = state['count']
                min_angle = state['current_rep_min_angle']
                
                # Determine depth rating
                if min_angle <= PERFECT_DEPTH:
                    depth_rating = "Excellent depth"
                    depth_score = 3
                elif min_angle <= GOOD_DEPTH:
                    depth_rating = "Good depth"
                    depth_score = 2
                elif min_angle <= DOWN_THRESHOLD:
                    depth_rating = "Shallow - go deeper"
                    depth_score = 1
                else:
                    depth_rating = "Incomplete rep"
                    depth_score = 0
                
                # Compile rep feedback
                rep_feedback = {
                    'rep_number': rep_num,
                    'min_angle': round(min_angle, 1),
                    'depth_rating': depth_rating,
                    'depth_score': depth_score,  # 0-3 scale
                    'form_issues': state['current_rep_issues'].copy() if state['current_rep_issues'] else ["Good form"],
                    'is_correct': depth_score >= 2 and len(state['current_rep_issues']) == 0
                }
                state['rep_details'].append(rep_feedback)
                
                # Reset for next rep
                state['is_down'] = False
                state['current_rep_min_angle'] = 180
                state['current_rep_issues'] = []
                
                # Provide immediate feedback
                if rep_feedback['is_correct']:
                    state['feedback'].append(f"Rep #{rep_num}: Perfect! ✓")
                else:
                    issues = ", ".join(rep_feedback['form_issues'][:2])
                    state['feedback'].append(f"Rep #{rep_num}: {depth_rating}. {issues}")
        else:
            state['feedback'].append(f"Angle: {int(elbow_angle)}°")

    except (IndexError, AttributeError) as e:
        state['feedback'].append("Position not detected")
    
    return state


def update_plank_state(landmarks, state, is_reverse=False, is_side=False):
    state['feedback'] = []
    
    required = [
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_ANKLE
    ]
    
    # Initialize time tracking if not present
    if 'cumulative_time' not in state:
        state['cumulative_time'] = 0.0
        state['start_time'] = None
        state['duration'] = 0.0
        
    try:
        # Check if full body is visible
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            # Pause timer if running
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
            return state
        
        shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        ankle = get_landmark(landmarks, PoseLandmark.LEFT_ANKLE)
        
        # Check horizontal alignment (Skip for Reverse Plank and Side Plank as arm length varies incline)
        is_horizontal = is_body_horizontal(landmarks)
        if not is_reverse and not is_side and not is_horizontal:
            state['feedback'].append("Get into plank position")
            if state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
            return state
        
        body_angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y],
            [ankle.x, ankle.y]
        )
        
        if is_reverse:
             print(f"DEBUG REVERSE: Angle={body_angle:.1f}, Horizontal={is_horizontal}, HipPos={check_hip_vertical_position(shoulder, hip, ankle)}")
        
        # Determine strictness based on variation
        if is_side:
            angle_valid = 150 < body_angle < 210 # Stricter for side
        elif is_reverse:
             # Reverse plank often has more variation due to shoulder flexibility
            angle_valid = 150 < body_angle < 210
        else:
            angle_valid = 160 < body_angle < 200 # Standard plank usually ~180
        
        if angle_valid:
            state['feedback'].append("Perfect Form! Holding... ⏱️")
            
            # Duration calculation depends on mode (Live vs Upload)
            if 'time_per_frame' in state:
                 # Upload Mode: Use fixed frame time
                 state['cumulative_time'] += state['time_per_frame']
                 state['duration'] = state['cumulative_time']
            else:
                 # Live Mode: Use wall clock
                 if state['start_time'] is None:
                     state['start_time'] = time.time()
                 
                 current_hold = time.time() - state['start_time']
                 state['duration'] = state['cumulative_time'] + current_hold
            
        else:
            # Pause timer (only relevant for Live Mode wall clock)
            if 'time_per_frame' not in state and state['start_time'] is not None:
                state['cumulative_time'] += time.time() - state['start_time']
                state['start_time'] = None
            
            # Feedback
            # Feedback
            position = check_hip_vertical_position(shoulder, hip, ankle)
            if position == 'low':
                state['feedback'].append("Raise your hips!")
            else:
                 state['feedback'].append("Lower your hips!")
                
    except (IndexError, AttributeError):
        if 'time_per_frame' not in state and state['start_time'] is not None:
            state['cumulative_time'] += time.time() - state['start_time']
            state['start_time'] = None
    
    return state
    
    return state


def update_squat_state(landmarks, state):
    state['feedback'] = []
    
    required = [
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE,
        PoseLandmark.LEFT_ANKLE,
        PoseLandmark.LEFT_SHOULDER
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        knee = get_landmark(landmarks, PoseLandmark.LEFT_KNEE)
        ankle = get_landmark(landmarks, PoseLandmark.LEFT_ANKLE)
        
        if not is_body_vertical(landmarks):
            state['feedback'].append("Stand upright to begin")
            return state
        
        knee_angle = calculate_angle(
            [hip.x, hip.y],
            [knee.x, knee.y],
            [ankle.x, ankle.y]
        )
        
        if 'frame_count_down' not in state:
            state['frame_count_down'] = 0
            state['frame_count_up'] = 0
        
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
    state['feedback'] = []
    
    required = [
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        hip = get_landmark(landmarks, PoseLandmark.LEFT_HIP)
        knee = get_landmark(landmarks, PoseLandmark.LEFT_KNEE)
        
        hip_angle = calculate_angle(
            [shoulder.x, shoulder.y],
            [hip.x, hip.y],
            [knee.x, knee.y]
        )
        
        if is_body_vertical(landmarks):
            state['feedback'].append("Lie down on your back")
            return state
        
        if 'frame_count_up' not in state:
            state['frame_count_up'] = 0
            state['frame_count_down'] = 0
        
        if hip_angle < 90:
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 3 and not state['is_down']:
                state['is_down'] = True
        elif hip_angle > 150:
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
    state['feedback'] = []
    
    required = [
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_ANKLE,
        PoseLandmark.RIGHT_ANKLE,
        PoseLandmark.LEFT_WRIST
    ]
    
    try:
        if not landmarks_visible(landmarks, required):
            state['feedback'].append("Ensure full body is visible")
            return state
        
        if not is_body_vertical(landmarks):
            state['feedback'].append("Stand upright to begin")
            return state
        
        shoulder_l = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        shoulder_r = get_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER)
        ankle_l = get_landmark(landmarks, PoseLandmark.LEFT_ANKLE)
        ankle_r = get_landmark(landmarks, PoseLandmark.RIGHT_ANKLE)
        wrist_l = get_landmark(landmarks, PoseLandmark.LEFT_WRIST)
        wrist_r = get_landmark(landmarks, PoseLandmark.RIGHT_WRIST)
        
        shoulder_width = abs(shoulder_l.x - shoulder_r.x)
        ankle_dist = abs(ankle_l.x - ankle_r.x)
        
        left_arm_up = wrist_l.y < shoulder_l.y
        right_arm_up = wrist_r.y < shoulder_r.y
        arms_are_up = left_arm_up and right_arm_up
        
        if 'frame_count_up' not in state:
            state['frame_count_up'] = 0
            state['frame_count_down'] = 0
        
        if ankle_dist > shoulder_width * 1.0 and arms_are_up:
            state['frame_count_up'] += 1
            state['frame_count_down'] = 0
            if state['frame_count_up'] >= 2 and not state['is_up']:
                state['is_up'] = True
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



def update_bicep_curl_state(landmarks, state):
    """
    Bicep Curl detection logic.
    Tracks elbow angle for reps.
    """
    state['feedback'] = []
    
    # Initialize rep tracking if not present
    if 'rep_details' not in state:
        state['rep_details'] = []
    
    try:
        # Get landmarks for both arms
        l_shoulder = get_landmark(landmarks, PoseLandmark.LEFT_SHOULDER)
        l_elbow = get_landmark(landmarks, PoseLandmark.LEFT_ELBOW)
        l_wrist = get_landmark(landmarks, PoseLandmark.LEFT_WRIST)
        
        r_shoulder = get_landmark(landmarks, PoseLandmark.RIGHT_SHOULDER)
        r_elbow = get_landmark(landmarks, PoseLandmark.RIGHT_ELBOW)
        r_wrist = get_landmark(landmarks, PoseLandmark.RIGHT_WRIST)
        
        # Determine angle mode
        use_3d = state.get('use_3d', False)
        
        # Calculate angles
        l_angle = None
        if l_shoulder and l_elbow and l_wrist:
            if use_3d:
                l_angle = calculate_angle_3d(
                    [l_shoulder.x, l_shoulder.y, l_shoulder.z], 
                    [l_elbow.x, l_elbow.y, l_elbow.z], 
                    [l_wrist.x, l_wrist.y, l_wrist.z]
                )
            else:
                 l_angle = calculate_angle([l_shoulder.x, l_shoulder.y], [l_elbow.x, l_elbow.y], [l_wrist.x, l_wrist.y])
             
        r_angle = None
        if r_shoulder and r_elbow and r_wrist:
            if use_3d:
                r_angle = calculate_angle_3d(
                    [r_shoulder.x, r_shoulder.y, r_shoulder.z], 
                    [r_elbow.x, r_elbow.y, r_elbow.z], 
                    [r_wrist.x, r_wrist.y, r_wrist.z]
                )
            else:
                 r_angle = calculate_angle([r_shoulder.x, r_shoulder.y], [r_elbow.x, r_elbow.y], [r_wrist.x, r_wrist.y])
        
        # Determine active arm (track the one "performing" the curl)
        angle = None
        side = "Left"
        
        # Identify which arm is active (sharper angle = more engagement in curl usually)
        if l_angle is not None and (r_angle is None or l_angle < r_angle):
            angle = l_angle
            side = "Left"
        elif r_angle is not None:
            angle = r_angle
            side = "Right"
            
        if angle is None:
            return state
            
        # Thresholds (Relaxed)
        UP_THRESH = 75   # Peak contraction
        DOWN_THRESH = 145 # Full extension
        
        # State machine
        if angle > DOWN_THRESH:
            state['stage'] = 'down'
            # Reset feedback when extended
            
        if angle < UP_THRESH and state.get('stage') == 'down':
            state['stage'] = 'up'
            state['count'] += 1
            state['feedback'].append(f"{side} Curl Rep {state['count']}!")
            # Add visual feedback via 'last_rep' mechanism if we want, 
            # but current 'ui_feedback' handles it.
            
            # Record rep details
            state['rep_details'].append({
                'rep_count': state['count'],
                'min_angle': angle,
                'form_issues': ['Good form'], # Add specific issues if we detect them
                'is_correct': True,
                'depth_score': 3.0 # Perfect contraction (Scale 0-3 to match UI)
            })
            
        # Feedback Logic
        state['feedback'] = []
        
        # Encourage full extension if they are lingering in middle
        if state.get('stage') == 'up' and angle > 90 and angle < 140:
             state['feedback'].append("Fully extend arm")
             
        # Encourage full curl if they are lingering in middle
        elif state.get('stage') == 'down' and angle < 130 and angle > 80:
             state['feedback'].append("Curl all the way up")
             
        # Positive reinforcement
        if angle < UP_THRESH + 10:
             state['feedback'].append("Good squeeze!")
        elif angle > DOWN_THRESH - 10:
             state['feedback'].append("Good extension")
             
    except Exception as e:
        print(f"Error in bicep curl: {e}")
        
    return state


# --- GLOBAL STATE STORAGE ---
# Persist state across stream reconnections
GLOBAL_LIVE_STATE = {}

# --- EXERCISE CONFIGURATION ---
EXERCISE_CONFIG = {
    "bicep_curl": {
        "rule_logic": update_bicep_curl_state,
        "initial_state": {'count': 0, 'stage': 'down', 'feedback': []}
    },
    "pushup": {
        "rule_logic": update_pushup_state,
        "initial_state": {
            'count': 0, 'is_down': False, 'in_position': False, 'feedback': [],
            'frame_count_down': 0, 'frame_count_up': 0,
            'rep_details': [], 'current_rep_min_angle': 180, 'current_rep_issues': []
        }
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

    # Create local landmarker for thread safety and VIDEO mode (better accuracy)
    base_ops = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    # Using VIDEO mode for uploaded files - allows temporal consistency if we wanted it,
    # though our logic is currently stateless.
    # Note: For VIDEO mode, timestamps are required. 
    # Let's stick to IMAGE mode for simplicity unless we add timestamps, 
    # but local instance is the key fix.
    ops = vision.PoseLandmarkerOptions(
        base_options=base_ops,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.IMAGE
    )

    rule_state = copy.deepcopy(initial_rule_state)
    
    # Inject video FPS for accurate duration tracking in uploads
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        rule_state['time_per_frame'] = 1.0 / fps
    else:
        rule_state['time_per_frame'] = 0.033 # Fallback to ~30fps

    with vision.PoseLandmarker.create_from_options(ops) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
    
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect
            results = landmarker.detect(mp_image)
            
            if results.pose_landmarks:
                rule_state = rule_logic_func(results.pose_landmarks[0], rule_state)
    
    cap.release()

    final_results = {'status': 'success'}
    
    if 'count' in rule_state:
        final_results['reps'] = rule_state['count']
    
    if 'duration' in rule_state:
        # Use duration from state (which should now be accurate for uploads)
        final_results['duration_seconds'] = round(rule_state.get('duration', 0), 2)
    
    # Handles Rep-based exercises (Pushup, Curl, Squat, etc.)
    if 'rep_details' in rule_state and rule_state['rep_details']:
        rep_details = rule_state['rep_details']
        final_results['rep_details'] = rep_details
        
        # Calculate summary statistics
        total_reps = len(rep_details)
        correct_reps = sum(1 for r in rep_details if r.get('is_correct', False))
        avg_depth_score = sum(r.get('depth_score', 0) for r in rep_details) / total_reps if total_reps > 0 else 0
        
        # Collect all form issues
        all_issues = []
        for r in rep_details:
            issues = r.get('form_issues', [])
            if issues and issues != ['Good form']:
                all_issues.extend(issues)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
        # Generate summary
        final_results['form_summary'] = {
            'total_reps': total_reps,
            'correct_reps': correct_reps,
            'incorrect_reps': total_reps - correct_reps,
            'accuracy_percentage': round((correct_reps / total_reps * 100) if total_reps > 0 else 0, 1),
            'average_depth_score': round(avg_depth_score, 2),
            'common_issues': sorted(issue_counts.items(), key=lambda x: -x[1])[:3],
            'overall_feedback': _generate_overall_feedback(correct_reps, total_reps, issue_counts)
        }
        
    # Handle Duration-based exercises (Plank)
    elif 'duration' in rule_state:
        total_duration = rule_state.get('duration', 0)
        
        # We need to collect feedback history to generate a summary
        # Since update_plank_state just appends to 'feedback' list which might get cleared or is just a list of strings
        # We can scan the feedback history if we modified update_plank_state to keep a log, 
        # OR we can assume 'feedback' contains all messages if we didn't clear it.
        # But update_plank_state usually resets feedback=[] at start. 
        # Ideally, we should have accumulated issues in a separate list in state, e.g., 'issue_history'.
        # For now, let's just generate a simple summary based on duration.
        
        final_results['form_summary'] = {
            'duration_seconds': round(total_duration, 2),
            'accuracy_percentage': 100, # Placeholder, hard to calc without issue tracking
            'overall_feedback': f"Great job! You held the plank for {round(total_duration, 1)} seconds."
        }
        
        if total_duration < 10:
             final_results['form_summary']['overall_feedback'] += " Try to hold it longer next time!"
        elif total_duration > 60:
             final_results['form_summary']['overall_feedback'] += " That's an impressive duration!"

    return final_results

def _generate_overall_feedback(correct_reps, total_reps, issue_counts):
    """Generate overall workout feedback based on performance."""
    if total_reps == 0:
        return "No reps detected. Ensure full body is visible in the video."
    
    accuracy = (correct_reps / total_reps) * 100
    feedback = []
    
    if accuracy >= 90:
        feedback.append("Excellent workout! Your form is outstanding.")
    elif accuracy >= 70:
        feedback.append("Good workout! Most reps had proper form.")
    elif accuracy >= 50:
        feedback.append("Decent workout. Focus on improving form for better results.")
    else:
        feedback.append("Form needs work. Take it slower and focus on technique.")
    
    # Add specific recommendations
    if 'Shallow - go deeper' in str(issue_counts) or issue_counts.get('Shallow - go deeper', 0) > 0:
        feedback.append("Try to go deeper - aim to get your chest closer to the ground.")
    if issue_counts.get('Hip sagging', 0) >= 2:
        feedback.append("Engage your core to prevent hips from sagging.")
    if issue_counts.get('Hips too high', 0) >= 2:
        feedback.append("Lower your hips to maintain a straight body line.")
    if issue_counts.get('Elbows flaring out', 0) >= 2:
        feedback.append("Keep elbows closer to your body (45-degree angle).")
    
    return " ".join(feedback)

# --- LIVE FEED GENERATOR ---
@csrf_exempt
def get_live_status(request):
    """Return the current live analysis status for polling."""
    exercise = request.GET.get('exercise', 'pushup')
    
    if exercise in GLOBAL_LIVE_STATE:
        state = GLOBAL_LIVE_STATE[exercise]
        # Use persisted UI feedback if available, otherwise raw feedback
        feedback = state.get('ui_feedback', [])
        if not feedback:
            feedback = state.get('feedback', [])
            
        response = {'status': 'active', 'feedback': feedback}
        
        # Include rep details for more detailed feedback
        if 'rep_details' in state and state['rep_details']:
            last_rep = state['rep_details'][-1]
            response['last_rep_feedback'] = last_rep
            
        if 'count' in state:
            response['reps'] = state['count']
        if 'duration' in state:
            response['duration'] = round(state.get('duration', 0), 1)
            
        return JsonResponse(response)
        
    return JsonResponse({'status': 'idle', 'feedback': []})

# --- LIVE FEED GENERATOR ---
def generate_live_feed(camera_index, rule_logic_func, initial_rule_state, exercise_name):
    """Generator for live exercise analysis with real-time feedback."""
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        return

    # Use global state if available to persist across reconnections
    if exercise_name in GLOBAL_LIVE_STATE:
        rule_state = GLOBAL_LIVE_STATE[exercise_name]
    else:
        rule_state = copy.deepcopy(initial_rule_state)
        GLOBAL_LIVE_STATE[exercise_name] = rule_state
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        display_frame = cv2.resize(frame.copy(), (640, 480))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)
        
        # Clean current feedback for the frame processing cycle
        # Note: We don't want to clear GLOBAL feedback here blindly if we rely on it for polling.
        # But update_pushup_state clears it at start of call.
        # So polling will see "flickering" feedback?
        # Ideally, update_pushup_state would append to a "log" or "current_msg", 
        # but right now it resets `state['feedback']`.
        # To make polling work, we might need `update_pushup_state` to be slightly smarter or 
        # we capture the state RIGHT AFTER processing.
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            # Process frame (updates GLOBAL state in place)
            rule_state = rule_logic_func(landmarks, rule_state)
            # FORCE SYNC global state to ensure polling sees latest data
            GLOBAL_LIVE_STATE[exercise_name] = rule_state
            
            # Draw landmarks only - NO TEXT
            draw_landmarks(display_frame, landmarks, POSE_CONNECTIONS)
        
        # Determine if we need to extend feedback visibility for polling
        # If 'feedback' is transient (1 frame), polling (every 200ms) might miss it.
        # We should create a field 'display_feedback' in state that behaves like our previous overlay buffer.
        
        current_feedback = rule_state.get('feedback', [])
        if current_feedback:
             # Logic to persist feedback for UI polling
             # We write to a new key 'ui_feedback' that persists
             if 'ui_end_time' not in rule_state:
                 rule_state['ui_end_time'] = 0
                 
             import time
             now = time.time()
             
             # If it's a rep message, show for 2s. If minor feedback, 0.5s
             duration = 2.0 if any("Rep #" in m for m in current_feedback) else 0.5
             rule_state['ui_feedback'] = current_feedback
             rule_state['ui_end_time'] = now + duration
        
        # Clean up stale UI feedback in state so polling doesn't show old stuff forever
        # (This logic runs 10-30 times a second, so it handles cleanup cleanly)
        if 'ui_end_time' in rule_state and time.time() > rule_state['ui_end_time']:
             rule_state['ui_feedback'] = []

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
            try:
                os.remove(filepath)
            except:
                pass

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
def _live_feed_response(exercise_name, initial_state=None):
    config = EXERCISE_CONFIG[exercise_name]
    state = initial_state if initial_state is not None else config['initial_state']
    return StreamingHttpResponse(
        generate_live_feed(0, config['rule_logic'], state, exercise_name),
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
            # Reset session state
            request.session[f'{exercise}_state'] = copy.deepcopy(config['initial_state'])
            request.session.modified = True
            
            # Reset global state (for live feed)
            if exercise in GLOBAL_LIVE_STATE:
                GLOBAL_LIVE_STATE[exercise] = copy.deepcopy(config['initial_state'])
                
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
        frame_file = request.FILES['frame']
        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Could not decode frame'}, status=400)
        
        state_key = f'{exercise}_state'
        if state_key not in request.session:
            request.session[state_key] = copy.deepcopy(config['initial_state'])
        
        state = request.session[state_key]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Using global landmarker - for more robustness in prod, consider per-request or pool
        results = landmarker.detect(mp_image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            state = config['rule_logic'](landmarks, state)
            request.session[state_key] = state
            request.session.modified = True
        else:
            state['feedback'] = ['No pose detected - ensure full body is visible']
        
        response = {'status': 'success', 'feedback': state.get('feedback', [])}
        
        if 'count' in state:
            response['reps'] = state['count']
        if 'duration' in state:
            response['duration'] = round(state.get('duration', 0), 1)
        
        return JsonResponse(response)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def live_bicep_curl(request):
    """API endpoint for live Bicep Curl feed."""
    # Use 3D logic for live feed (better for front-facing webcam)
    initial_state = copy.deepcopy(EXERCISE_CONFIG['bicep_curl']['initial_state'])
    initial_state['use_3d'] = True
    
    return _live_feed_response("bicep_curl", initial_state)

@csrf_exempt
def upload_and_analyze_bicep_curl(request):
    """API endpoint for uploaded Bicep Curl video analysis."""
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        filepath = fs.path(filename)
        
        config = EXERCISE_CONFIG['bicep_curl']
        result = analyze_video(filepath, config['rule_logic'], config['initial_state'])
        
        # Clean up file
        os.remove(filepath)
        
        return JsonResponse(result)
    return JsonResponse({'error': 'Invalid request'}, status=400)