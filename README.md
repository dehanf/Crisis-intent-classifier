# Intent Classifier - Bag of Words (BoW)

A crisis message intent classification system using Neural Networks and TF-IDF vectorization. This project classifies emergency messages into five categories: Rescue, Medical, Supply, Utility, and Other.

## 🎯 Project Overview

This intent classifier is designed to analyze crisis-related messages (e.g., during natural disasters) and categorize them based on their intent. It uses:
- **Bag of Words (BoW)** with TF-IDF weighting for text representation
- **Neural Network (MLP)** for multi-class classification
- **NLTK Porter Stemmer** for text preprocessing

## 📋 Features

- **5 Intent Categories:**
  - `Rescue` - Emergency situations requiring immediate rescue
  - `Medical` - Medical emergencies and health-related requests
  - `Supply` - Requests for supplies, food, water, etc.
  - `Utility` - Infrastructure updates, road conditions, warnings
  - `Other` - General information and miscellaneous messages

- **Text Preprocessing:**
  - Lowercasing
  - Punctuation removal
  - Word stemming using Porter Stemmer

- **Model Architecture:**
  - Input: TF-IDF vectorized text (max 2500 features)
  - Hidden layers with dropout for regularization
  - Softmax output for multi-class classification

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Intent-classifier-BoW
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) If you need to generate the dataset:
```bash
python src/dataset_creation.py
```

### Training the Model

Run the training script to train the model on the crisis intent dataset:

```bash
python src/train.py
```

This will:
- Load and preprocess the data from `data/crisis_intent_data.csv`
- Train a neural network model
- Save the model and preprocessing artifacts to the `models/` directory

### Making Predictions

Use the prediction script to classify new messages:

```bash
python src/predict.py
```

Or import and use the classification function in your own code:

```python
from predict import classify_message

message = "SOS! We are trapped on the roof in Ja-Ela, water is rising!"
intent, confidence = classify_message(message)
print(f"Detected Intent: {intent} ({confidence*100:.1f}%)")
```

## 📁 Project Structure

```
Intent-classifier-BoW/
│
├── data/
│   └── crisis_intent_data.csv      # Training dataset
│
├── models/
│   ├── intent_classifier.h5        # Trained Keras model
│   ├── tokenizer.pickle             # Fitted tokenizer
│   └── label_encoder.pickle         # Label encoder for intents
│
├── src/
│   ├── dataset_creation.py         # Generate synthetic crisis data
│   ├── preprocessor.py             # Text cleaning and stemming
│   ├── train.py                    # Model training script
│   └── predict.py                  # Prediction script
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 📊 Dataset

The project includes a synthetic dataset generator (`dataset_creation.py`) that creates realistic crisis messages based on templates. The dataset contains 10,000 messages across 5 intent categories.

**Sample Messages:**
- **Rescue:** "SOS: 5 people trapped on a roof in Colombo. Water rising fast."
- **Medical:** "Help! My grandmother is bedridden and the water is knee deep. Location: Gampaha."
- **Supply:** "We need milk powder and diapers for 10 babies at the Kelaniya camp."
- **Utility:** "BREAKING: Water levels in Kelani River (Colombo) have reached 8.5 meters. Warning!"
- **Other:** "Just saw on news that Kandy town is flooded. Hope everyone is safe."

## 🔧 Model Details

- **Vectorization:** TF-IDF with max 2500 features
- **Architecture:** 
  - Dense layer (256 units, ReLU)
  - Dropout (0.3)
  - Dense layer (128 units, ReLU)
  - Output layer (5 units, Softmax)
- **Training:** 15 epochs, batch size 32, 20% validation split
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Label encoding
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations

## 📝 Notes

- The model uses `silence_tensorflow` to suppress verbose TensorFlow logs, especially useful when using GPU/CUDA
- Trained model files are saved in the `models/` directory
- The preprocessing pipeline (tokenizer and label encoder) must be saved and loaded for consistent predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Your Name

## 🙏 Acknowledgments

- Dataset inspired by crisis management scenarios in Sri Lanka
- Built for educational purposes in NLP and deep learning

