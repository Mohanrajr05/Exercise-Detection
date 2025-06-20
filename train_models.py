import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TimeDistributed, LSTM, Dense, Dropout,
                                     Flatten, Input, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from pathlib import Path
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
import time
from concurrent.futures import ThreadPoolExecutor
import random

# --- IMPROVEMENT 1: Refined Configuration ---
# Adjusted learning rate, epochs, and patience for more robust training.
CONFIG = {
    'frame_count': 16,
    'image_size': (160, 160),
    'batch_size': 4,
    'epochs': 50,  # Increased epochs for more thorough training
    'lstm_units': [64, 32],
    'dropout_rate': 0.5,  # Increased dropout for better regularization
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42,
    'learning_rate': 5e-5,  # Lowered learning rate for finer tuning
    'num_parallel_calls': os.cpu_count() or 1,
    'use_augmentation': True,
    'early_stopping_patience': 10,  # Increased patience
    'reduce_lr_patience': 5       # Increased patience
}

class ExerciseModelTrainer:
    def __init__(self, dataset_path, model_path, labels):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.labels = labels
        self.num_classes = len(labels)
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*CONFIG['image_size'], 3),
            pooling='avg'
        )
        base_model.trainable = False
        return base_model

    # --- IMPROVEMENT 2: More Advanced Data Augmentation ---
    # Added zoom and color jitter to create more varied training data.
    def _augment_frame(self, frame):
        """Apply more diverse augmentations to a frame"""
        # Horizontal Flip
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        # Rotation
        if random.random() > 0.6: # Slightly less frequent
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((CONFIG['image_size'][0]//2, CONFIG['image_size'][1]//2), angle, 1)
            frame = cv2.warpAffine(frame, M, CONFIG['image_size'])

        # Brightness/Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2) # Contrast
            beta = random.randint(-20, 20)      # Brightness
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
        # Zoom
        if random.random() > 0.6:
            zoom_factor = random.uniform(1.0, 1.2)
            h, w = frame.shape[:2]
            ch, cw = h // 2, w // 2 # Center
            
            # Crop a zoomed-in area
            h_crop, w_crop = int(h / zoom_factor), int(w / zoom_factor)
            h_start, w_start = ch - h_crop // 2, cw - w_crop // 2
            
            cropped = frame[h_start:h_start+h_crop, w_start:w_start+w_crop]
            frame = cv2.resize(cropped, (w, h))

        return frame

    def _load_single_video(self, video_path_label_tuple):
        video_path, label = video_path_label_tuple
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            # Decide if this video will be augmented
            augment_this_video = CONFIG['use_augmentation'] and random.random() > 0.4 # Augment 60% of videos

            while len(frames) < CONFIG['frame_count']:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, CONFIG['image_size'])
                
                if augment_this_video:
                    frame = self._augment_frame(frame)

                frame = frame.astype('float32') / 255.0
                frames.append(frame)

            cap.release()

            while len(frames) < CONFIG['frame_count']:
                frames.append(np.zeros((*CONFIG['image_size'], 3), dtype=np.float32))

            return np.array(frames), label

        except Exception as e:
            print(f"Error loading {video_path}: {str(e)}")
            return None, None


    def _load_data(self):
        video_paths_labels = []

        for label_name, label_idx in self.labels.items():
            label_path = self.dataset_path / label_name
            if not label_path.exists():
                print(f"Warning: Label path {label_path} does not exist.")
                continue

            for video_file in os.listdir(label_path):
                video_path = label_path / video_file
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    video_paths_labels.append((video_path, label_idx))

        if not video_paths_labels:
             return np.array([]), np.array([])

        with ThreadPoolExecutor(max_workers=CONFIG['num_parallel_calls']) as executor:
            results = list(executor.map(self._load_single_video, video_paths_labels))

        data_list = [item[0] for item in results if item[0] is not None]
        labels_list = [item[1] for item in results if item[1] is not None]
        
        if not data_list:
            return np.array([]), np.array([])

        return np.array(data_list), np.array(labels_list)


    # --- IMPROVEMENT 3: Deeper and More Regularized Model ---
    # Added layers to help the model learn more complex features and prevent overfitting.
    def _build_model(self):
        model = Sequential([
            Input(shape=(CONFIG['frame_count'], *CONFIG['image_size'], 3)),
            TimeDistributed(self.feature_extractor),
            
            # Added a dense block for more feature learning post-CNN
            TimeDistributed(Dense(256, activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Dropout(CONFIG['dropout_rate'])),

            LSTM(CONFIG['lstm_units'][0], return_sequences=True),
            BatchNormalization(),
            Dropout(CONFIG['dropout_rate']), # Dropout after LSTM
            
            LSTM(CONFIG['lstm_units'][1]),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(CONFIG['dropout_rate']),
            Dense(self.num_classes, activation='softmax')
        ])

        optimizer = Adam(learning_rate=CONFIG['learning_rate'])

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self):
        print(f"\nLoading data from {self.dataset_path}...")
        start_time = time.time()
        data, labels = self._load_data()

        if data.size == 0:
            raise ValueError(f"No training data found in {self.dataset_path}.")

        print(f"Loaded {len(data)} videos in {time.time()-start_time:.2f} seconds")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Class distribution:", dict(zip(unique_labels, counts)))

        if len(unique_labels) < self.num_classes:
            print(f"Warning: Found {len(unique_labels)} classes, but model expects {self.num_classes}.")

        # Data splitting
        stratify_labels = labels if len(unique_labels) > 1 and all(c > 1 for c in counts) else None
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=stratify_labels
        )
        
        # Build model
        model = self._build_model()
        model.summary()

        # Callbacks
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=CONFIG['early_stopping_patience'], restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(self.model_path), monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=CONFIG['reduce_lr_patience'], min_lr=1e-7, verbose=1)
        ]

        # Train model
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            validation_split=CONFIG['validation_split'], # Use validation_split directly
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on the test set
        print("\nEvaluating on test set...")
        if self.model_path.exists():
            print("Loading best model for final evaluation...")
            model = tf.keras.models.load_model(str(self.model_path))
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        print(f"\nBest model saved to {self.model_path}")
        return history

def main():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # This list remains the same
    datasets_info = [
        {"name": "jumping_jacks", "dataset_folder": "JumpingJacksDataset", "labels": {"not_jumping_jacks": 0, "jumping_jacks": 1}},
        {"name": "reverse_plank", "dataset_folder": "ReversePlankDataset", "labels": {"not_reverse_plank": 0, "reverse_plank": 1}},
        {"name": "side_plank", "dataset_folder": "SidePlankDataset", "labels": {"not_side_plank": 0, "side_plank": 1}},
        {"name": "pushup", "dataset_folder": "PushupDataset", "labels": {"not_pushup": 0, "pushup": 1}},
        {"name": "plank", "dataset_folder": "PlankDataset", "labels": {"not_plank": 0, "plank": 1}},
        {"name": "situp", "dataset_folder": "SitupDataset", "labels": {"not_situp": 0, "situp": 1}},
        {"name": "squat", "dataset_folder": "SquatDataset", "labels": {"not_squat": 0, "squat": 1}},
    ]

    for info in datasets_info:
        print("\n" + "="*50)
        print(f"Training {info['name']} model...")
        print("="*50)
        
        trainer = ExerciseModelTrainer(
            dataset_path=script_dir / info["dataset_folder"],
            model_path=models_dir / f"{info['name']}_classification_model.keras",
            labels=info["labels"]
        )
        try:
            trainer.train()
        except Exception as e:
            print(f"Could not train {info['name']} model: {e}")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("Running on CPU")
        
    main()