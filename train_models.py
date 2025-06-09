import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TimeDistributed, LSTM, Dense, Dropout,
                                     Flatten, Input, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.utils import to_categorical # Not strictly needed for sparse_categorical_crossentropy
from sklearn.model_selection import train_test_split
from pathlib import Path
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
import time
from concurrent.futures import ThreadPoolExecutor
import random # For more augmentations

# Configuration optimized for CPU and small dataset
CONFIG = {
    'frame_count': 16,
    'image_size': (160, 160), # MobileNetV2 preferred sizes include 96, 128, 160, 192, 224
    'batch_size': 4, # Keep small for CPU, can increase if memory allows
    'epochs': 30, # Increased epochs, EarlyStopping will manage
    'lstm_units': [64, 32], # Kept as is, can be tuned
    'dropout_rate': 0.4, # Slightly increased dropout
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42,
    'learning_rate': 1e-4, # Adam optimizer default is 1e-3, 1e-4 is a good start
    'num_parallel_calls': os.cpu_count() // 2 or 1, # Use more available cores cautiously
    'use_augmentation': True,
    'early_stopping_patience': 7, # Increased patience
    'reduce_lr_patience': 4   # Increased patience
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
        base_model.trainable = False # Freeze base model
        return base_model

    def _augment_frame(self, frame):
        """Apply more diverse augmentations to a frame"""
        # Horizontal Flip
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)

        # Rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10) # Rotate between -10 and 10 degrees
            M = cv2.getRotationMatrix2D((CONFIG['image_size'][0]//2, CONFIG['image_size'][1]//2), angle, 1)
            frame = cv2.warpAffine(frame, M, CONFIG['image_size'])

        # Brightness
        if random.random() > 0.5:
            value = random.uniform(0.7, 1.3) # Adjust brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v_channel = cv2.split(hsv)
            v_channel = np.clip(v_channel * value, 0, 255).astype(np.uint8)
            final_hsv = cv2.merge((h, s, v_channel))
            frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            
        return frame

    def _load_single_video(self, video_path_label_tuple):
        video_path, label = video_path_label_tuple
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            processed_frames_count = 0

            while cap.isOpened() and processed_frames_count < CONFIG['frame_count']:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, CONFIG['image_size'])
                
                # Augmentation applied only during training (not explicitly handled here, but good to note)
                # For data loading, it's often better to augment on-the-fly in a generator
                # However, for simplicity here, we apply if use_augmentation is True
                if CONFIG['use_augmentation'] and random.random() > 0.5: # Apply augmentation to some videos
                    augmented_frame_set = []
                    for _ in range(CONFIG['frame_count']): # Augment all frames of this video consistently if chosen
                        # Create a copy for augmentation to avoid modifying the original frame for other ops
                        frame_to_augment = frame.copy() 
                        augmented_frame_set.append(self._augment_frame(frame_to_augment))
                    # This logic is flawed for per-frame augmentation *within* _load_single_video
                    # The original implementation augmented individual frames with 50% chance. Let's revert to that simpler one.
                    # The _augment_frame should take the raw frame before normalization.

                # Corrected augmentation application per frame
                raw_frame = frame.copy() # Keep original for potential non-augmented path
                if CONFIG['use_augmentation'] and random.random() > 0.5:
                     # Convert to BGR for augmentation functions if they expect it, then back if needed
                    frame_bgr = raw_frame 
                    frame_bgr = self._augment_frame(frame_bgr) # _augment_frame expects BGR
                    final_frame = frame_bgr
                else:
                    final_frame = raw_frame

                # Normalize
                final_frame = final_frame.astype('float32') / 255.0
                frames.append(final_frame)
                processed_frames_count += 1

            cap.release()

            # Pad if video is shorter than required frames
            while len(frames) < CONFIG['frame_count']:
                frames.append(np.zeros((*CONFIG['image_size'], 3), dtype=np.float32))

            return np.array(frames[:CONFIG['frame_count']]), label

        except Exception as e:
            print(f"Error loading {video_path}: {str(e)}")
            # Return a placeholder that can be filtered out
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
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']: # Basic check
                    video_paths_labels.append((video_path, label_idx))

        if not video_paths_labels:
             return np.array([]), np.array([])

        data_list = []
        labels_list = []

        # Using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=CONFIG['num_parallel_calls']) as executor:
            results = list(executor.map(self._load_single_video, video_paths_labels))

        for frames, label in results:
            if frames is not None and label is not None:
                data_list.append(frames)
                labels_list.append(label)
        
        if not data_list:
            return np.array([]), np.array([])

        return np.array(data_list), np.array(labels_list)


    def _build_model(self):
        model = Sequential([
            Input(shape=(CONFIG['frame_count'], *CONFIG['image_size'], 3)),
            TimeDistributed(self.feature_extractor),
            # TimeDistributed(Flatten()), # MobileNetV2 with pooling='avg' already flattens features
            TimeDistributed(Dense(128, activation='relu')), # Increased units
            TimeDistributed(BatchNormalization()),          # Added BatchNormalization
            TimeDistributed(Dropout(CONFIG['dropout_rate'])), # Added Dropout

            LSTM(CONFIG['lstm_units'][0], return_sequences=True),
            BatchNormalization(), # Batch norm between LSTMs can help
            LSTM(CONFIG['lstm_units'][1]),
            BatchNormalization(),

            Dropout(CONFIG['dropout_rate']),
            Dense(self.num_classes, activation='softmax') # Use num_classes
        ])

        optimizer = Adam(learning_rate=CONFIG['learning_rate'])

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy', # Use this if y_train is integer labels
            metrics=['accuracy']
        )
        return model

    def train(self):
        print(f"\nLoading data from {self.dataset_path}...")
        start_time = time.time()
        data, labels = self._load_data()

        if data.size == 0 or labels.size == 0:
            raise ValueError(f"No training data loaded from {self.dataset_path}! Check dataset structure and video files.")

        print(f"Loaded {len(data)} videos in {time.time()-start_time:.2f} seconds")
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")


        # Check class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Class distribution in loaded data:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} samples")

        if len(unique_labels) < self.num_classes:
            print(f"Warning: Only {len(unique_labels)} classes found in data, but model expects {self.num_classes}. This might lead to errors.")


        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=labels if len(unique_labels) > 1 and all(c > 1 for c in counts) else None # Stratify only if possible
        )

        # Further split training set for validation
        # Ensure y_train has enough samples for stratification if used
        unique_ytrain_labels, ytrain_counts = np.unique(y_train, return_counts=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=CONFIG['validation_split'],
            random_state=CONFIG['random_state'],
            stratify=y_train if len(unique_ytrain_labels) > 1 and all(c > 1 for c in ytrain_counts) else None
        )

        print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")
        print(f"Val shape: {X_val.shape}, {y_val.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")

        # Build model
        model = self._build_model()
        model.summary()

        # Callbacks
        self.model_path.parent.mkdir(parents=True, exist_ok=True) # Ensure model directory exists
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.model_path),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, # Reduce LR by half
                patience=CONFIG['reduce_lr_patience'],
                min_lr=1e-7, # Lower min_lr
                verbose=1
            )
        ]

        # Train model
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set (using the best weights restored by EarlyStopping)
        print("\nEvaluating on test set...")
        # Load the best model saved by ModelCheckpoint for final evaluation
        if self.model_path.exists():
            print("Loading best saved model for final evaluation...")
            model = tf.keras.models.load_model(str(self.model_path)) # Re-load best model
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Save the final model (ModelCheckpoint already saves the best one)
        # model.save(str(self.model_path)) # This might overwrite the best one if last epoch wasn't best
        print(f"\nBest model during training saved to {self.model_path}")

        return history

def main():
    # Ensure models directory exists
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    # Define common dataset root if applicable, or specify per trainer
    # base_dataset_dir = script_dir / "datasets" # Example structure

    datasets_info = [
        {
            "name": "pushup",
            "dataset_folder": "PushupDataset",
            "labels": {"not_pushup": 0, "pushup": 1}
        },
        {
            "name": "plank",
            "dataset_folder": "Dataset", # Assuming this is for plank
            "labels": {"not_plank": 0, "plank": 1}
        },
        {
            "name": "situp",
            "dataset_folder": "SitupDataset",
            "labels": {"not_situp": 0, "situp": 1}
        },
        {
            "name": "squat",
            "dataset_folder": "SquatDataset",
            "labels": {"not_squat": 0, "squat": 1}
        }
    ]

    for info in datasets_info:
        print("\n" + "="*50)
        print(f"Training {info['name']} model...")
        print("="*50)
        
        trainer = ExerciseModelTrainer(
            dataset_path=script_dir / info["dataset_folder"],
            model_path=models_dir / f"{info['name']}_classification_model.keras", # Use .keras format
            labels=info["labels"]
        )
        try:
            trainer.train()
        except ValueError as e:
            print(f"Could not train {info['name']} model: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during {info['name']} training: {e}")


if __name__ == "__main__":
    # Set memory growth for GPUs if available, to avoid OOM errors
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