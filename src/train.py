#Used these lines since i'm getting too many logs from tensorflow when utilizing my GPU using CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Fatal errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        # Silence oneDNN optimization logs
import silence_tensorflow.auto


import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv("data/crisis_intent_data.csv")

# 2. Vectorization (BoW)
max_words = 2000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['message'])
X = tokenizer.texts_to_matrix(df['message'], mode='tfidf')

# 3. Encoding Labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['intent']) # Encodes to 0-4

# 4. Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(max_words,), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='sigmoid') #sigmoid for multi-label classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(X, y, epochs=15, batch_size=32, verbose=1)

# 6. SAVE ASSETS
model.save('models/intent_classifier.h5')
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('models/label_encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and Tokenizer saved to models/ folder.")