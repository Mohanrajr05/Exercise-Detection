import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TimeDistributed, LSTM, Dense, Dropout, 
                                   Flatten, Input, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pathlib import Path
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration optimized for CPU and small dataset
CONFIG = {
    'frame_count': 16,
    'image_size': (160, 160),
    'batch_size': 4,
    'epochs': 20,
    'lstm_units': [64, 32],
    'dropout_rate': 0.3,
    'test_size': 0.2,
    'validation_split': 0.2,
    'random_state': 42,
    'learning_rate': 1e-4,
    'num_parallel_calls': 4,
    'use_augmentation': True,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3
}

class ExerciseModelTrainer:
    def __init__(self, dataset_path, model_path, labels):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.labels = labels
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
    
    def _augment_frame(self, frame):
        """Apply simple augmentations to a frame"""
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        return frame
    
    def _load_single_video(self, video_path, label):
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while cap.isOpened() and len(frames) < CONFIG['frame_count']:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, CONFIG['image_size'])
                if CONFIG['use_augmentation'] and np.random.rand() > 0.5:
                    frame = self._augment_frame(frame)
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
            
            cap.release()
            
            # Pad if video is shorter than required frames
            while len(frames) < CONFIG['frame_count']:
                frames.append(np.zeros((*CONFIG['image_size'], 3), dtype=np.float32))
            
            return np.array(frames[:CONFIG['frame_count']]), label
        
        except Exception as e:
            print(f"Error loading {video_path}: {str(e)}")
            return None
    
    def _load_data(self):
        video_paths = []
        labels = []
        
        for label_name, label_idx in self.labels.items():
            label_path = self.dataset_path / label_name
            if not label_path.exists():
                continue
                
            for video_file in os.listdir(label_path):
                video_path = label_path / video_file
                video_paths.append(video_path)
                labels.append(label_idx)
        
        # Load videos with limited parallelism
        with ThreadPoolExecutor(max_workers=CONFIG['num_parallel_calls']) as executor:
            results = list(executor.map(
                lambda x: self._load_single_video(x[0], x[1]), 
                zip(video_paths, labels)
            ))
        
        # Filter out None results and separate data/labels
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return np.array([]), np.array([])
            
        data, labels = zip(*valid_results)
        return np.array(data), np.array(labels)
    
    def _build_model(self):
        model = Sequential([
            Input(shape=(CONFIG['frame_count'], *CONFIG['image_size'], 3)),
            TimeDistributed(self.feature_extractor),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(64, activation='relu')),
            LSTM(CONFIG['lstm_units'][0], return_sequences=True),
            LSTM(CONFIG['lstm_units'][1]),
            Dropout(CONFIG['dropout_rate']),
            Dense(len(self.labels), activation='softmax')
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
        
        if len(data) == 0:
            raise ValueError("No training data found!")
        
        print(f"Loaded {len(data)} videos in {time.time()-start_time:.2f} seconds")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'],
            stratify=labels
        )
        
        # Further split training set for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=CONFIG['validation_split'],
            random_state=CONFIG['random_state'],
            stratify=y_train
        )
        
        print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")
        print(f"Val shape: {X_val.shape}, {y_val.shape}")
        print(f"Test shape: {X_test.shape}, {y_test.shape}")
        
        # Build model
        model = self._build_model()
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=CONFIG['early_stopping_patience'],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                str(self.model_path),
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=CONFIG['reduce_lr_patience'],
                min_lr=1e-6
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
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Save the final model
        model.save(str(self.model_path))
        print(f"\nModel saved to {self.model_path}")
        
        return history

def main():
    # Ensure models directory exists
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Train pushup model (single class)
    print("\n" + "="*50)
    print("Training pushup model...")
    print("="*50)
    pushup_trainer = ExerciseModelTrainer(
        dataset_path=Path(__file__).resolve().parent / "Correct sequence",
        model_path=models_dir / "pushup_classification_model.h5",
        labels={"pushup": 0}  # Single class
    )
    pushup_trainer.train()
    
    # Train plank model (two classes)
    print("\n" + "="*50)
    print("Training plank model...")
    print("="*50)
    plank_trainer = ExerciseModelTrainer(
        dataset_path=Path(__file__).resolve().parent / "Dataset",
        model_path=models_dir / "plank_classification_model.h5",
        labels={"not_plank": 0, "plank": 1}  # Two classes
    )
    plank_trainer.train()
    
    # Train situp model (two classes)
    print("\n" + "="*50)
    print("Training situp model...")
    print("="*50)
    situp_trainer = ExerciseModelTrainer(
        dataset_path=Path(__file__).resolve().parent / "SitupDataset",
        model_path=models_dir / "situp_classification_model.h5",
        labels={"not_situp": 0, "situp": 1}  # Two classes
    )
    situp_trainer.train()

if __name__ == "__main__":
    main()