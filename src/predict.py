#Used these lines since i'm getting too many logs from tensorflow when utilizing my GPU using CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Fatal errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        # Silence oneDNN optimization logs
import silence_tensorflow.auto

import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load assets once
model = load_model('models/intent_classifier.h5')
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('models/label_encoder.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

def classify_message(text):
    # Process text using the saved tokenizer
    bow_vector = tokenizer.texts_to_matrix([text], mode='tfidf')
    
    # Predict
    prediction = model.predict(bow_vector, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    
    intent = encoder.inverse_transform([class_idx])[0]
    return intent, confidence

# Example usage
msg = "SOS! We are trapped on the roof in Ja-Ela, water is rising!"
intent, score = classify_message(msg)
print(f"Detected Intent: {intent} ({score*100:.1f}%)")