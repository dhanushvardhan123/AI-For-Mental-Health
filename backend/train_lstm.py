# backend/train_lstm.py
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pickle

# --- 1. Parameters ---
# CORRECTED PATHS: Using your exact folder paths
TRAIN_DIR = r'C:\CAPSTONE\capstone prj dataset 1\voice emotion\train' 
TEST_DIR = r'C:\CAPSTONE\capstone prj dataset 1\voice emotion\test'   
VAL_DIR = r'C:\CAPSTONE\capstone prj dataset 1\voice emotion\val'     

# We need to pick a fixed length for audio features
# 174 frames is ~4 seconds of audio. Adjust if most of your files are longer.
MAX_PAD_LEN = 174 
N_MFCC = 40 # Number of MFCC features to extract

EPOCHS = 100
BATCH_SIZE = 32

# --- 2. Feature Extraction Function ---
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # Pad or truncate to a fixed length
        if (mfccs.shape[1] > MAX_PAD_LEN):
            mfccs = mfccs[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
    # We need to transpose the features so shape is (timesteps, features)
    return mfccs.T

# --- 3. Data Loading Function ---
def load_data_from_directory(data_dir):
    features = []
    labels = []
    
    # Get the list of emotions (subdirectories)
    # This checks if the path exists before trying to list files
    if not os.path.exists(data_dir):
        print(f"Error: Path not found: {data_dir}")
        return np.array(features), np.array(labels)
        
    emotions = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Loading data from {data_dir}. Found emotions: {emotions}")

    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        
        for file_name in os.listdir(emotion_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(emotion_dir, file_name)
                mfccs = extract_features(file_path)
                
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)
    
    print(f"Finished loading from {data_dir}. Found {len(features)} files.")
    return np.array(features), np.array(labels)

# --- 4. Load All Data (Train, Val, Test) ---
X_train, y_train = load_data_from_directory(TRAIN_DIR)
X_val, y_val = load_data_from_directory(VAL_DIR)
X_test, y_test = load_data_from_directory(TEST_DIR)

if len(X_train) == 0:
    print(f"Error: No data found in {TRAIN_DIR}. Please check the path.")
else:
    # --- 5. Encode Labels ---
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {label_encoder.classes_}")

    # --- 6. Build the LSTM Model ---
    # Input shape will be (MAX_PAD_LEN, N_MFCC) -> (timesteps, features)
    model = Sequential([
        LSTM(128, input_shape=(MAX_PAD_LEN, N_MFCC), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])

    # --- 7. Compile and Train ---
    model.compile(
        loss='sparse_categorical_crossentropy', # Use sparse because labels are integers
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    model.summary()

    print("Starting speech model training...")
    history = model.fit(
        X_train, y_train_encoded,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val_encoded)
    )

    # --- 8. Save Model and Label Encoder ---
    model.save('speech_emotion_model.h5')
    print("Model saved as speech_emotion_model.h5")

    # Save the label encoder
    with open('speech_label_encoder.pickle', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved as speech_label_encoder.pickle")


    # --- 9. Plot Graphs ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.suptitle('LSTM Training and Validation Metrics')
    plt.show()

    # --- 10. Evaluate on Test Set ---
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    print("Training complete.")