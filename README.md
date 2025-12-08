# Motion Detection Backend (Django)

## Overview
This is the backend for the Fitness Motion Analytics platform. It is built with Django and provides API endpoints for analyzing exercise videos (pushup, plank, situp, squat, lunges, high knees) using both rule-based logic (MediaPipe) and deep learning models (TensorFlow/Keras). It supports both video upload and live webcam analysis.

---

## Required Packages
Install all dependencies with:
```sh
pip install -r requirements.txt
```

**Key packages:**
- Django
- djangorestframework
- django-cors-headers
- numpy
- opencv-python, opencv-contrib-python
- mediapipe
- tensorflow, keras
- scikit-learn
- pillow

See `requirements.txt` for the full list and versions.

---

## Exercise Implementation
Each exercise is analyzed using two approaches:
- **Rule-based logic:** Uses MediaPipe Pose to extract body landmarks and applies geometric rules to count reps or measure duration.
- **Model-based logic:** (Optional) Uses a trained Keras model (MobileNetV2 + LSTM) to classify video segments as exercise or not.

### Pushup
- **Rule-based:** Counts a rep when the elbow angle drops below 90° (down) and then rises above 160° (up), with 2 consecutive frames required for state change.
- **Model-based:** Classifies 16-frame segments as pushup/not-pushup.

### Plank
- **Rule-based:** Measures duration when the body is straight (shoulder-hip-ankle angle ~180°) and hips are not sagging.
- **Model-based:** Classifies 16-frame segments as plank/not-plank.

### Situp
- **Rule-based:** Counts a rep when the hip angle drops below 90° (up) and then rises above 150° (down), with 2 consecutive frames for state change.
- **Model-based:** Classifies 16-frame segments as situp/not-situp.

### Squat
- **Rule-based:** Counts a rep when the knee angle drops below 90° (down) and then rises above 160° (up), with 2 consecutive frames for state change.
- **Model-based:** Classifies 16-frame segments as squat/not-squat.

### Lunges
- **Rule-based:** Counts a rep when either knee angle drops below 110° (lunge) and then rises above 150° (standing), with 2 consecutive frames for state change.
- **Model-based:** Classifies 16-frame segments as lunge/not-lunge.

### High Knees
- **Rule-based:** Counts a rep when either knee angle drops below 100° (up) and then rises above 160° (down), with 5 consecutive frames for state change.
- **Model-based:** Classifies 16-frame segments as high_knees/not-high_knees.

---

## Model Training
- Models are trained using `train_models.py`.
- Each exercise has its own dataset folder (e.g., `PushupDataset`, `SitupDataset`, etc.) with subfolders for each class (e.g., `pushup`, `not_pushup`).
- Models use MobileNetV2 for feature extraction and LSTM for temporal modeling.
- Trained models are saved in the `models/` directory as `.keras` files.

**To train all models:**
```sh
python train_models.py
```

---

## API Endpoints
All endpoints are under the default Django server (e.g., `http://localhost:8000/`).

### Video Upload (POST, multipart/form-data)
- `/analyze/analyze_pushup/`
- `/analyze/analyze_plank/`
- `/analyze/analyze_situp/`
- `/analyze/analyze_squat/`
- `/analyze/analyze_lunges/`
- `/analyze/analyze_high_knees/`

**Request:**
- `video` (file): The video to analyze

**Example (curl):**
```sh
curl -F "video=@your_video.mp4" http://localhost:8000/analyze/analyze_pushup/
```

### Live Webcam Analysis (GET, MJPEG stream)
- `/analyze/live_pushup/`
- `/analyze/live_plank/`
- `/analyze/live_situp/`
- `/analyze/live_squat/`
- `/analyze/live_lunges/`
- `/analyze/live_high_knees/`

Returns a live video stream with overlayed analytics (rep count, model prediction, etc.).

---

## Setup Instructions
1. **Clone the repo and navigate to the backend folder:**
   ```sh
   cd Motion-detection
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run migrations:**
   ```sh
   python manage.py migrate
   ```
5. **(Optional) Train models:**
   ```sh
   python train_models.py
   ```
6. **Start the server:**
   ```sh
   python manage.py runserver
   ```

---

## Important Notes
- **CORS:** The backend allows all origins for development. Adjust in `settings.py` for production.
- **Uploads:** Uploaded videos are stored in `uploaded_videos/` and deleted after processing.
- **Models:** If a model is missing, only rule-based analysis will be performed.
- **Camera:** For live endpoints, a webcam is required and must be accessible by OpenCV.
- **Performance:** For best results, use clear, well-lit videos with the full body visible.

---

## Contribution
- Fork the repo, create a branch, and submit a pull request.
- Please document any new endpoints or logic clearly in the code and README.

---

For questions or issues, open an issue in the repository. 
