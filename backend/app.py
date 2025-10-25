# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import librosa
import cv2 # You will need to: pip install opencv-python
import base64
import os
import google.generativeai as genai
import io # To handle audio file in memory

# --- 1. Initialize App and CORS ---
app = Flask(__name__)
CORS(app) # Allow requests from your React frontend

# --- 2. Database Configuration ---
# !! REPLACE WITH YOUR OWN MYSQL CREDENTIALS !!
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root', # <-- EDIT THIS
    'database': 'mental_health_db'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

# --- 3. Load AI Models and Encoders ---
print("Loading models... This may take a moment.")
try:
    face_model = load_model('facial_emotion_model.h5')
    text_model = load_model('text_emotion_model.h5')
    speech_model = load_model('speech_emotion_model.h5')

    # Load the text tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Load the speech label encoder
    with open('speech_label_encoder.pickle', 'rb') as handle:
        speech_label_encoder = pickle.load(handle)
        
    # Load the Haar Cascade for face detection
    # This file comes with opencv-python
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Could not find Haar Cascade file at {haar_cascade_path}")
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    print("Models loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")
    # Handle the error appropriately, maybe exit or set a flag
    face_model = None
    text_model = None
    speech_model = None

# --- 4. Gemini API Configuration ---
# !! REPLACE WITH YOUR OWN GEMINI API KEY !!
# (Store this securely, e.g., as an environment variable)
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDNI-UROT2F0sA_dWQ2BIU0Q8TsHZ2uC5U' # <-- EDIT THIS
try:
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini model configured.")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    gemini_model = None


# --- 5. Preprocessing Helper Functions ---

# --- Face Preprocessing ---
def preprocess_face_image(base64_image_data):
    # 1. Decode base64 string
    try:
        img_data = base64.b64decode(base64_image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # Convert to grayscale
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

    # 2. Detect face
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected")
        return None # No face found
        
    (x, y, w, h) = faces[0] # Use first face
    face_roi = img[y:y+h, x:x+w]

    # 3. Resize to model's expected input (48x48)
    try:
        resized_face = cv2.resize(face_roi, (48, 48))
    except Exception as e:
        print(f"Error resizing face: {e}")
        return None
    
    # 4. Rescale and reshape
    processed_img = resized_face.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0) # Add batch dimension
    processed_img = np.expand_dims(processed_img, axis=-1) # Add channel dimension (1 for grayscale)
    return processed_img

# --- Text Preprocessing ---
def preprocess_text(text):
    # Use MAX_LENGTH from train_rnn.py
    MAX_LENGTH = 100 
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    return padded_seq
    
# --- Audio Preprocessing ---
def preprocess_audio(audio_file_storage):
    # Use parameters from train_lstm.py
    MAX_PAD_LEN = 174 
    N_MFCC = 40
    
    try:
        # Load audio file from in-memory file storage
        audio, sample_rate = librosa.load(audio_file_storage, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        
        # Pad or truncate
        if (mfccs.shape[1] > MAX_PAD_LEN):
            mfccs = mfccs[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        # Transpose and add batch dimension
        processed_audio = mfccs.T
        processed_audio = np.expand_dims(processed_audio, axis=0)
        return processed_audio
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


# --- 6. Auth Routes ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'message': 'Username and password are required'}), 400
        
    username = data['username']
    password = data['password']
    
    hashed_password = generate_password_hash(password)
    
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Database connection error'}), 500
        
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except mysql.connector.Error as err:
        if err.errno == 1062: # Duplicate entry
            return jsonify({'message': 'Username already exists'}), 409
        return jsonify({'message': f'Error: {err}'}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'message': 'Username and password are required'}), 400
        
    username = data['username']
    password = data['password']
    
    conn = get_db_connection()
    if conn is None:
        return jsonify({'message': 'Database connection error'}), 500
        
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            # In a real app, return a JWT (JSON Web Token)
            return jsonify({'message': 'Login successful', 'username': user['username']}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    except mysql.connector.Error as err:
        return jsonify({'message': f'Error: {err}'}), 500
    finally:
        cursor.close()
        conn.close()
            
# --- 7. Prediction Routes ---
@app.route('/predict_face', methods=['POST'])
def predict_face():
    if not face_model:
        return jsonify({'error': 'Face model is not loaded'}), 500
        
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data found'}), 400
        
    base64_image = data['image']
    
    processed_image = preprocess_face_image(base64_image)
    if processed_image is None:
        return jsonify({'error': 'No face detected or error in processing'}), 400
        
    prediction = face_model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Make sure this list matches your train_cnn.py classes
    FACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion = FACE_EMOTIONS[predicted_class_index]
    
    return jsonify({'emotion': emotion})

@app.route('/predict_text', methods=['POST'])
def predict_text():
    if not text_model:
        return jsonify({'error': 'Text model is not loaded'}), 500
        
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text data found'}), 400
        
    text = data['text']
    processed_text = preprocess_text(text)
    
    prediction = text_model.predict(processed_text)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # This uses the label encoder from train_rnn.py
    # We must know the classes. Re-run train_rnn.py if unsure.
    # Common classes for this dataset:
    TEXT_EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    # If your model was trained on different labels, you must update this list!
    
    try:
        emotion = TEXT_EMOTIONS[predicted_class_index]
        return jsonify({'emotion': emotion})
    except IndexError:
        return jsonify({'error': 'Model prediction index is out of range. Check TEXT_EMOTIONS list.'}), 500


@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    if not speech_model:
        return jsonify({'error': 'Speech model is not loaded'}), 500
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400
        
    audio_file = request.files['audio']
    
    processed_audio = preprocess_audio(audio_file)
    if processed_audio is None:
        return jsonify({'error': 'Error processing audio file'}), 400
        
    prediction = speech_model.predict(processed_audio)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Use the loaded label encoder to get the emotion string
    emotion = speech_label_encoder.classes_[predicted_class_index]
    
    return jsonify({'emotion': emotion})

# --- 8. Gemini Chatbot Route (UPDATED) ---
@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_model:
        return jsonify({'error': 'Gemini model is not configured'}), 500
        
    data = request.get_json()
    user_message = data.get('message')
    face_emotion = data.get('face_emotion', 'not detected')
    text_emotion = data.get('text_emotion', 'not detected')
    speech_emotion = data.get('speech_emotion', 'not detected')

    # Craft a prompt for Gemini
    prompt = f"""
    You are an empathic and predictive AI assistant for mental health.
    A user is interacting with you. Here is the data collected from them:
    - Facial Emotion Detected: {face_emotion}
    - Text Emotion Detected: {text_emotion}
    - Speech Emotion Detected: {speech_emotion}
    
    The user's latest message is: "{user_message}"
    
    Based on all this information, please provide a warm, empathic, and supportive response.
    If the detected emotions seem negative (like sad, angry, fear, euphoric, surprised), acknowledge this gently.
    Conclude your response by offering one or two simple, actionable tips to help improve their current state of mind (e.g., a breathing exercise, a mindfulness tip, or a suggestion to take a short break).
    Keep the response concise and easy to understand.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        # THIS IS THE NEW LINE TO SHOW US THE REAL ERROR
        print(f"!!!!!!!!!!!!!!! GEMINI API ERROR: {e} !!!!!!!!!!!!!!!") 
        return jsonify({'error': f'Error generating response: {e}'}), 500
# --- 9. Run the App ---
if __name__ == '__main__':
    # You MUST install opencv-python for this to work
    # pip install opencv-python
    if None in [face_model, text_model, speech_model, gemini_model]:
        print("!!! CRITICAL ERROR: One or more models (or Gemini) failed to load. Server will run but predictions will fail. !!!")
    
    print("Starting Flask server at http://localhost:5000")
    app.run(debug=True, port=5000)