#Used these lines since i'm getting too many logs from tensorflow when utilizing my GPU using CUDA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Fatal errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        # Silence oneDNN optimization logs
import silence_tensorflow.auto


import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from preprocessor import clean_text  # Import your custom logic

# 1. Load Data
df = pd.read_csv("data/crisis_intent_data.csv")

# 2. Apply Stemming Preprocessing
print("Cleaning and Stemming dataset...")
df['message_cleaned'] = df['message'].apply(clean_text)

# 3. Vectorization (BoW)
max_words = 2500
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['message_cleaned'])

# Transform the cleaned text into a TF-IDF Matrix
X = tokenizer.texts_to_matrix(df['message_cleaned'], mode='tfidf')

# 4. Encoding Labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['intent'])

# 5. Build Neural Network (MLP)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(max_words,), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid') #Sigmoid for multi-label classification
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 6. Train
print("Starting training...")
model.fit(X, y, epochs=15, batch_size=32, verbose=1, validation_split=0.2)

# 7. SAVE ASSETS
os.makedirs('models', exist_ok=True)
model.save('models/intent_classifier.h5')

# Save the tokenizer and encoder (essential for predict.py)
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)
with open('models/label_encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle)

print("\nModel and assets saved successfully in 'models/'")