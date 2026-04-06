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
import cv2
import base64
import io
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

# --- 1. Initialize App and CORS ---
app = Flask(__name__)
CORS(app)

# --- 2. Database Configuration ---
db_config = {
   'host': os.environ.get("DB_HOST", "localhost"),
    'user': os.environ.get("DB_USER", "root"),
    'password': os.environ.get("DB_PASSWORD", "root"),
    'database': os.environ.get("DB_NAME", "mental_health_db"),
    'port': int(os.environ.get("DB_PORT", 3306))
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    face_model = load_model(os.path.join(BASE_DIR, 'facial_emotion_model.h5'))
    text_model = load_model(os.path.join(BASE_DIR, 'text_emotion_model.h5'))
    speech_model = load_model(os.path.join(BASE_DIR, 'speech_emotion_model.h5'))

    with open(os.path.join(BASE_DIR, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(os.path.join(BASE_DIR, 'speech_label_encoder.pickle'), 'rb') as handle:
        speech_label_encoder = pickle.load(handle)

    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Could not find Haar Cascade file at {haar_cascade_path}")

    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    face_model = text_model = speech_model = None

# --- 4. Groq API Configuration ---
# !! PASTE YOUR NEW GROQ API KEY HERE !!
try:
    client = Groq(
api_key=os.getenv("GROQ_API_KEY"),  
  )
    print("Groq client configured.")
except Exception as e:
    print(f"Error configuring Groq: {e}")
    client = None

# --- 5. Preprocessing Helper Functions ---
# (These are all the same, no changes needed)

# --- Face Preprocessing ---
def preprocess_face_image(base64_image_data):
    try:
        img_data = base64.b64decode(base64_image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected")
        return None
    (x, y, w, h) = faces[0]
    face_roi = img[y:y+h, x:x+w]
    try:
        resized_face = cv2.resize(face_roi, (48, 48))
    except Exception as e:
        print(f"Error resizing face: {e}")
        return None
    processed_img = resized_face.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = np.expand_dims(processed_img, axis=-1)
    return processed_img

# --- Text Preprocessing ---
def preprocess_text(text):
    MAX_LENGTH = 100 
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    return padded_seq
    
# --- Audio Preprocessing ---
def preprocess_audio(audio_file_storage):
    MAX_PAD_LEN = 174 
    N_MFCC = 40
    try:
        audio, sample_rate = librosa.load(audio_file_storage, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        if (mfccs.shape[1] > MAX_PAD_LEN):
            mfccs = mfccs[:, :MAX_PAD_LEN]
        else:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        processed_audio = mfccs.T
        processed_audio = np.expand_dims(processed_audio, axis=0)
        return processed_audio
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# --- 6. Auth Routes ---
# (These are all the same, no changes needed)
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'message': 'Username and password are required'}), 400
    username = data['username']
    password = data['password']
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    if conn is None: return jsonify({'message': 'Database connection error'}), 500
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except mysql.connector.Error as err:
        if err.errno == 1062: return jsonify({'message': 'Username already exists'}), 409
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
    if conn is None: return jsonify({'message': 'Database connection error'}), 500
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            return jsonify({'message': 'Login successful', 'username': user['username']}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    except mysql.connector.Error as err:
        return jsonify({'message': f'Error: {err}'}), 500
    finally:
        cursor.close()
        conn.close()
            
# --- 7. Prediction Routes ---
# (These are all the same, no changes needed)
@app.route('/predict_face', methods=['POST'])
def predict_face():
    if not face_model: return jsonify({'error': 'Face model is not loaded'}), 500
    data = request.get_json()
    if 'image' not in data: return jsonify({'error': 'No image data found'}), 400
    base64_image = data['image']
    processed_image = preprocess_face_image(base64_image)
    if processed_image is None: return jsonify({'error': 'No face detected or error in processing'}), 400
    prediction = face_model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    FACE_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion = FACE_EMOTIONS[predicted_class_index]
    return jsonify({'emotion': emotion})

@app.route('/predict_text', methods=['POST'])
def predict_text():
    if not text_model: return jsonify({'error': 'Text model is not loaded'}), 500
    data = request.get_json()
    if 'text' not in data: return jsonify({'error': 'No text data found'}), 400
    text = data['text']
    processed_text = preprocess_text(text)
    prediction = text_model.predict(processed_text)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    TEXT_EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    try:
        emotion = TEXT_EMOTIONS[predicted_class_index]
        return jsonify({'emotion': emotion})
    except IndexError:
        return jsonify({'error': 'Model prediction index out of range.'}), 500

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    if not speech_model: return jsonify({'error': 'Speech model is not loaded'}), 500
    if 'audio' not in request.files: return jsonify({'error': 'No audio file found'}), 400
    audio_file = request.files['audio']
    processed_audio = preprocess_audio(audio_file)
    if processed_audio is None: return jsonify({'error': 'Error processing audio file'}), 400
    prediction = speech_model.predict(processed_audio)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    emotion = speech_label_encoder.classes_[predicted_class_index]
    return jsonify({'emotion': emotion})

# --- 8. Chatbot Route (UPDATED FOR GROQ) ---
@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({'error': 'Groq client is not configured'}), 500
        
    data = request.get_json()
    user_message = data.get('message')
    face_emotion = data.get('face_emotion', 'not detected')
    text_emotion = data.get('text_emotion', 'not detected')
    speech_emotion = data.get('speech_emotion', 'not detected')

    # Craft the system prompt
    system_prompt = f"""You are an empathic and predictive AI assistant for mental health.
A user is interacting with you. Here is the data collected from them:
- Facial Emotion Detected: {face_emotion}
- Text Emotion Detected: {text_emotion}
- Speech Emotion Detected: {speech_emotion}

Based on all this information, please provide a warm, empathic, and supportive response.
If the detected emotions seem negative (like sad, angry, fear, euphoric, surprised), acknowledge this gently.
Conclude your response by offering one or two simple, actionable tips to help improve their current state of mind (e.g., a breathing exercise, a mindfulness tip, or a suggestion to take a short break).
Keep the response concise and easy to understand."""
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            # --- THIS IS THE FIXED LINE ---
            model="llama-3.1-8b-instant", 
        )
        
        reply = chat_completion.choices[0].message.content
        return jsonify({'reply': reply})
        
    except Exception as e:
        print(f"!!!!!!!!!!!!!!! GROQ API ERROR: {e} !!!!!!!!!!!!!!!") 
        return jsonify({'error': f'Error generating response: {e}'}), 500

# --- 9. Run the App ---
if __name__ == '__main__':
    if None in [face_model, text_model, speech_model, client]:
        print("!!! CRITICAL ERROR: One or more models (or Groq client) failed to load. Server will run but predictions will fail. !!!")
    
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)