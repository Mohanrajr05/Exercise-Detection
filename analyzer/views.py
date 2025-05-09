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
from tensorflow.keras.models import load_model
from django.shortcuts import render
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import time

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploaded_videos'
MODEL_DIR = BASE_DIR / 'models'

UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / 'pushup_classification_model.h5'
PLANK_MODEL_PATH = MODEL_DIR / 'plank_classification_model.h5'
SITUP_MODEL_PATH = MODEL_DIR / 'situp_classification_model.h5'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load pre-trained models
pushup_model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
plank_model = load_model(PLANK_MODEL_PATH) if PLANK_MODEL_PATH.exists() else None
situp_model = load_model(SITUP_MODEL_PATH) if SITUP_MODEL_PATH.exists() else None

# Helper function to calculate angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

@csrf_exempt
def upload_and_analyze_pushup(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename

        try:
            cap = cv2.VideoCapture(str(filepath))
            pushup_counter = 0
            pushup_state = False
            consecutive_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)

                    if elbow_angle < 70 and not pushup_state:
                        consecutive_frames += 1
                        if consecutive_frames >= 2:
                            pushup_state = True
                            consecutive_frames = 0

                    elif elbow_angle > 150 and pushup_state:
                        consecutive_frames += 1
                        if consecutive_frames >= 2:
                            pushup_state = False
                            pushup_counter += 1
                            consecutive_frames = 0

            cap.release()

            return JsonResponse({
                'exercise_type': 'pushup',
                'pushup_count': pushup_counter
            })

        except Exception as e:
            print(f"Error analyzing pushup video: {e}")
            return JsonResponse({'error': 'Error analyzing the video.'}, status=500)

    return render(request, 'home.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    pushup_counter = 0
    pushup_state = False
    consecutive_frames = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))  # Big frame size
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                if elbow_angle < 70 and not pushup_state:
                    consecutive_frames += 1
                    if consecutive_frames >= 2:
                        pushup_state = True
                        consecutive_frames = 0

                elif elbow_angle > 150 and pushup_state:
                    consecutive_frames += 1
                    if consecutive_frames >= 2:
                        pushup_state = False
                        pushup_counter += 1
                        consecutive_frames = 0

                # Draw pose landmarks
                draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )

            # Show pushup count on screen
            cv2.putText(frame, f"Pushups: {pushup_counter}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            
@csrf_exempt
def upload_and_analyze_plank(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename

        try:
            cap = cv2.VideoCapture(str(filepath))
            plank_frames = 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

                    hip_straight = abs(hip[1] - (shoulder[1] + ankle[1]) / 2) < 0.1
                    body_straight = calculate_angle(shoulder, hip, ankle) > 160

                    if hip_straight and body_straight:
                        plank_frames += 1

            cap.release()

            plank_seconds = round(plank_frames / fps, 2) if fps > 0 else 0

            return JsonResponse({
                'exercise_type': 'plank',
                'plank_duration_seconds': plank_seconds
            })

        except Exception as e:
            print(f"Error analyzing plank video: {e}")
            return JsonResponse({'error': 'Error analyzing the video.'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

 

import time  # Add this at the top

def gen_plank_frames():
    camera = cv2.VideoCapture(0)
    frames_list = []
    plank_detected = False
    plank_start_time = None
    plank_duration = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))  # Big frame size
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

                hip_straight = abs(hip[1] - (shoulder[1] + ankle[1]) / 2) < 0.1
                body_straight = calculate_angle(shoulder, hip, ankle) > 160

                if hip_straight and body_straight:
                    if not plank_detected:
                        plank_start_time = time.time()
                    plank_detected = True
                else:
                    plank_detected = False
                    plank_start_time = None
                    plank_duration = 0

            if plank_detected and plank_start_time is not None:
                plank_duration = int(time.time() - plank_start_time)

            # Draw label
            text = "Plank Detected" if plank_detected else "No Plank"
            color = (0, 255, 0) if plank_detected else (0, 0, 255)

            cv2.putText(frame, text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if plank_detected:
                cv2.putText(frame, f"Duration: {plank_duration}s", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Timer

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@csrf_exempt
def upload_and_analyze_situp(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage(location=UPLOAD_DIR)
        filename = fs.save(video.name, video)
        filepath = UPLOAD_DIR / filename

        try:
            cap = cv2.VideoCapture(str(filepath))
            situp_counter = 0
            situp_state = False
            consecutive_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

                    hip_angle = calculate_angle(shoulder, hip, knee)

                    if hip_angle < 70 and not situp_state:
                        consecutive_frames += 1
                        if consecutive_frames >= 2:
                            situp_state = True
                            consecutive_frames = 0

                    elif hip_angle > 150 and situp_state:
                        consecutive_frames += 1
                        if consecutive_frames >= 2:
                            situp_state = False
                            situp_counter += 1
                            consecutive_frames = 0

            cap.release()

            return JsonResponse({
                'exercise_type': 'situp',
                'situp_count': situp_counter
            })

        except Exception as e:
            print(f"Error analyzing situp video: {e}")
            return JsonResponse({'error': 'Error analyzing the video.'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def gen_situp_frames():
    camera = cv2.VideoCapture(0)
    situp_counter = 0
    situp_state = False
    consecutive_frames = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

                hip_angle = calculate_angle(shoulder, hip, knee)

                if hip_angle < 70 and not situp_state:
                    consecutive_frames += 1
                    if consecutive_frames >= 2:
                        situp_state = True
                        consecutive_frames = 0

                elif hip_angle > 150 and situp_state:
                    consecutive_frames += 1
                    if consecutive_frames >= 2:
                        situp_state = False
                        situp_counter += 1
                        consecutive_frames = 0

                # Draw pose landmarks
                draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_default_pose_landmarks_style()
                )

            # Show situp count on screen
            cv2.putText(frame, f"Situps: {situp_counter}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def live_pushup(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_situp(request):
    return StreamingHttpResponse(gen_situp_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_plank(request):
    return StreamingHttpResponse(gen_plank_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, 'analyzer/home.html')

