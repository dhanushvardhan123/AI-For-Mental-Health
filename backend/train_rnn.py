# backend/train_rnn.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Dropout
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# --- 1. Parameters ---
VOCAB_SIZE = 10000
MAX_LENGTH = 100
EMBEDDING_DIM = 128
OOV_TOKEN = "<OOV>" # Out-of-vocabulary token
EPOCHS = 20
BATCH_SIZE = 64

# --- 2. Load Data Function ---
def load_data(filepath):
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'emotion'])
    return df['text'].values, df['emotion'].values

# --- 3. Load and Prepare Data ---
train_texts, train_labels = load_data('C:/CAPSTONE/capstone prj dataset 1/text emotion/train.txt')
val_texts, val_labels = load_data('C:/CAPSTONE/capstone prj dataset 1/text emotion/val.txt')
test_texts, test_labels = load_data('C:/CAPSTONE/capstone prj dataset 1/text emotion/test.txt')

# --- 4. Encode Labels (e.g., 'anger' -> 0, 'sadness' -> 1) ---
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

num_classes = len(label_encoder.classes_)

# --- 5. Tokenize and Pad Text ---
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

val_sequences = tokenizer.texts_to_sequences(val_texts)
val_padded = pad_sequences(val_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# --- 6. Build the RNN (LSTM) Model ---
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    SpatialDropout1D(0.3),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5), # This line caused the error
    Dense(num_classes, activation='softmax') # Multi-class classification
])

# --- 7. Compile the Model ---
model.compile(
    loss='sparse_categorical_crossentropy', # Use this since labels are integers
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

# --- 8. Train the Model ---
print("Starting text model training...")
history = model.fit(
    train_padded,
    train_labels_encoded,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(val_padded, val_labels_encoded)
)

# --- 9. Save Model and Tokenizer ---
model.save('text_emotion_model.h5')
# Save the tokenizer so we can reuse it for new predictions
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Model saved as text_emotion_model.h5 and tokenizer.pickle")

# --- 10. Plot Graphs (same as CNN) ---
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

plt.suptitle('RNN Training and Validation Metrics')
plt.show()

# --- 11. Evaluate on Test Set ---
print("Evaluating on test set...")
results = model.evaluate(test_padded, test_labels_encoded)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

print("Training complete.")