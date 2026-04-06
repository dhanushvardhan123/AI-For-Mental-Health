from tensorflow.keras.models import load_model

# Face model
model = load_model("facial_emotion_model.h5", compile=False)
model.save("facial_emotion_model_new.h5")

# Text model
model = load_model("text_emotion_model.h5", compile=False)
model.save("text_emotion_model_new.h5")

# Speech model
model = load_model("speech_emotion_model.h5", compile=False)
model.save("speech_emotion_model_new.h5")

print("✅ All models converted successfully")