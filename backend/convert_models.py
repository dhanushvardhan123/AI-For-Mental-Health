from tensorflow.keras.models import load_model

# Face
model = load_model("facial_emotion_model.h5", compile=False)
model.save("facial_emotion_model.keras")

# Text
model = load_model("text_emotion_model.h5", compile=False)
model.save("text_emotion_model.keras")

# Speech
model = load_model("speech_emotion_model.h5", compile=False)
model.save("speech_emotion_model.keras")

print("✅ Converted to .keras format")