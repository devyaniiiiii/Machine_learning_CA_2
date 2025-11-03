# 📰 Fake News Detection for Indian News

## Project Overview

Fake news poses a significant threat to society by spreading misinformation, influencing public opinion, and eroding trust in legitimate news sources. In the context of Indian news, where a large and diverse population relies on various platforms for information, detecting fake news is crucial to maintain a well-informed citizenry and prevent the spread of harmful narratives.

This project develops and compares **machine learning and deep learning models** to classify Indian news articles as either **Real (0)** or **Fake (1)**. We implement three different approaches:
- **Logistic Regression** (baseline ML model)
- **Support Vector Machine (SVM)** (advanced ML model)
- **Long Short-Term Memory (LSTM)** (deep learning model)

**Key Finding:** The LSTM model demonstrated the highest F1 Score and Recall, making it the most effective approach for this fake news detection task.

---

## Table of Contents

1. [Dataset Information](#dataset-information)
2. [Installation & Setup](#installation--setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Development](#model-development)
6. [Results & Comparison](#results--comparison)
7. [Conclusions](#conclusions)
8. [Future Work](#future-work)
9. [References](#references)

---

## Dataset Information

### Dataset: IFND (Indian Fake News Dataset)

- **Source:** [Kaggle - IFND Dataset](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset?resource=download)
- **Total Records:** 45,393 news articles
- **Features:** 9 columns including text content, labels, web sources, and categories
- **File:** `IFND.csv`

### Dataset Structure

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each article |
| `Eng_Trans_Statement` | English translated statement/article text |
| `Label` | Binary label (0 = Real, 1 = Fake) |
| `Web` | Source website of the news article |
| `Category` | News category (e.g., Government, Election, Violence, COVID-19) |

### Class Distribution

The dataset exhibits **significant class imbalance**:
- **Real News (0):** 37,800 articles (83.3%)
- **Fake News (1):** 7,593 articles (16.7%)

This imbalance was addressed using stratified splitting and class weighting during model training.

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.7+ required
# Google Colab (recommended) or local Python environment
```

### Required Libraries

```bash
pip install pandas numpy nltk scikit-learn tensorflow matplotlib seaborn wordcloud joblib
```

### NLTK Data Download

```python
import nltk
nltk.download('stopwords')
```

### Running the Code

1. **Mount Google Drive** (if using Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Update the dataset path**:
```python
path = '/content/IFND.csv'  # Update this path
df = pd.read_csv(path)
```

3. **Run all cells sequentially** in the provided Python notebook

---

## Data Preprocessing

### Text Cleaning Pipeline

The raw text from the `Eng_Trans_Statement` column underwent comprehensive preprocessing:

#### Step 1: Basic Cleaning
```python
def clean(s):
    s = str(s).lower()                    # Convert to lowercase
    s = html.unescape(s)                  # Decode HTML entities
    s = re.sub(r'http\S+', '', s)         # Remove URLs
    s = re.sub(r'[^a-z0-9 ]', ' ', s)     # Remove special characters
    s = re.sub(r'\s+', ' ', s).strip()    # Normalize whitespace
    return s
```

#### Step 2: Stop Word Removal
```python
# Remove common English stop words
text_stop = ' '.join([w for w in text_clean.split() if w not in STOPWORDS])
```

### Generated Columns

- **`text_clean`**: Text after basic cleaning (Steps 1-5)
- **`text_stop`**: Text after stop word removal (all steps)
- **`Label`**: Binary label (0 = Real, 1 = Fake)

### Train-Test Split

- **Training Set:** 80% (36,314 articles)
- **Testing Set:** 20% (9,079 articles)
- **Method:** Stratified split to maintain class distribution
- **Random State:** 42 (for reproducibility)

---

## Exploratory Data Analysis

### Key Findings

#### 1. Label Distribution

<img width="1189" height="590" alt="Image" src="https://github.com/user-attachments/assets/09abbe49-9fed-4c43-9789-ff4c7ab0e9d4" />

**Observation:** Significant class imbalance with real news dominating the dataset. This necessitates careful evaluation using metrics beyond accuracy.

<img width="1169" height="913" alt="Image" src="https://github.com/user-attachments/assets/e9e61ee0-fd41-4fb7-9214-1b0d026ebe85" />

#### 2. Top News Sources

Most frequent web sources:
1. **TRIBUNEINDIA** - Highest contributor
2. **THEPRINT** - Second highest
3. **THESTATESMAN** - Third highest

**Note:** Some naming inconsistencies detected (e.g., 'THESTATESMAN' vs 'THESTATEMAN')

#### 3. Category Distribution

Top categories by frequency:
1. **GOVERNMENT** - Most common
2. **ELECTION** - Second most
3. **VIOLENCE** - Third most
4. **COVID-19** - Fourth most

#### 4. Text Length Analysis

- **Average word count:** ~150 words per article
- **Character count range:** Wide distribution indicating diverse article lengths
- Articles vary significantly in length, requiring padding/truncation for LSTM

#### 5. Word Cloud & N-gram Analysis

**Top Bigrams in Fake News:**
- Phrases often related to sensational claims
- More emotional and urgent language patterns

<img width="1239" height="676" alt="Image" src="https://github.com/user-attachments/assets/f73e1a1d-efd6-4c6e-8539-034341a52900" />

**Top Bigrams in Real News:**
- More factual and neutral terminology
- References to official sources and verified information

---

## Model Development

### Model 1: Logistic Regression

#### Approach
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features:** 20,000 (later increased to 40,000)
- **N-gram Range:** Unigrams and bigrams (1,2)
- **Class Weight:** Balanced to handle class imbalance
- **Solver:** 'lbfgs' with max 2000 iterations

#### Implementation
```python
tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)
```

#### Rationale

<img width="765" height="621" alt="Image" src="https://github.com/user-attachments/assets/4810680f-6c8a-410d-94c5-b1ba7b908550" />

Logistic Regression serves as a strong baseline for text classification, offering interpretability and computational efficiency.

---

### Model 2: Support Vector Machine (SVM)

#### Approach
- **Kernel:** Linear (effective for high-dimensional text data)
- **Vectorization:** Same TF-IDF features as Logistic Regression
- **Class Weight:** Balanced
- **Calibration:** CalibratedClassifierCV for probability estimates

#### Implementation
```python
svc = LinearSVC(class_weight='balanced', max_iter=5000)
calibrated_svc = CalibratedClassifierCV(svc)
calibrated_svc.fit(X_train_tfidf, y_train)
```

#### Rationale

<img width="802" height="616" alt="image" src="https://github.com/user-attachments/assets/0ce83c81-a6db-4ebd-bf85-d136cf0d63d8" />

SVMs excel at finding optimal decision boundaries in high-dimensional spaces, often outperforming simpler linear models.

---

### Model 3: LSTM (Long Short-Term Memory)

#### Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Embedding                   (None, 200, 100)          3,000,000 
Bidirectional(LSTM)         (None, 256)               234,496   
Dropout(0.5)                (None, 256)               0         
Dense(64, relu)             (None, 64)                16,448    
Dropout(0.3)                (None, 64)                0         
Dense(1, sigmoid)           (None, 1)                 65        
=================================================================
Total params: 3,251,009
Trainable params: 3,251,009
```

#### Configuration
- **Vocabulary Size:** 30,000 words
- **Embedding Dimension:** 100
- **Sequence Length:** 200 tokens (padded/truncated)
- **LSTM Units:** 128 (bidirectional = 256 total)
- **Dropout:** 0.5 (after LSTM), 0.3 (after Dense)
- **Optimizer:** Adam
- **Loss:** Binary crossentropy
- **Early Stopping:** Patience of 2 epochs on validation loss

#### Implementation
```python
tokenizer = Tokenizer(num_words=30000, oov_token='<OOV>')
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)

model = Sequential([
    Embedding(30000, 100, input_length=200),
    Bidirectional(LSTM(128)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

#### Rationale

<img width="755" height="610" alt="Image" src="https://github.com/user-attachments/assets/0db9a8ff-ba28-4e90-ab14-40e21080a5d4" />

LSTMs capture sequential dependencies and contextual information in text, making them ideal for understanding nuanced language patterns that distinguish fake from real news.

---

## Results & Comparison

### Performance Metrics

<img width="846" height="547" alt="Image" src="https://github.com/user-attachments/assets/82d6dc10-9e1b-480b-8f73-3c9f39fc8661" />

| Metric     | Logistic Regression | SVM   | LSTM  |
|------------|---------------------|-------|-------|
| **Accuracy**   | 0.959               | 0.964 | **0.964** |
| **F1 Score**   | 0.863               | 0.883 | **0.889** |
| **Precision**  | **0.974**           | 0.965 | 0.924 |
| **Recall**     | 0.776               | 0.814 | **0.856** |

### Detailed Analysis

#### Logistic Regression
- ✅ **Strengths:** Highest precision (97.4%), fast training, interpretable
- ❌ **Weaknesses:** Lowest recall (77.6%), misses ~22% of fake news
- 📊 **Use Case:** When false positives must be minimized

#### Support Vector Machine
- ✅ **Strengths:** Balanced performance, improved recall over LR
- ❌ **Weaknesses:** Computationally expensive, slightly lower precision than LR
- 📊 **Use Case:** Good all-around performer for production systems

#### LSTM (Best Overall)
- ✅ **Strengths:** 
  - **Highest F1 Score (0.889)** - Best balance of precision and recall
  - **Highest Recall (0.856)** - Catches 85.6% of fake news
  - Captures contextual and sequential information
- ❌ **Weaknesses:** 
  - Lower precision (92.4%) than LR and SVM
  - Slower training time
  - Requires more computational resources
- 📊 **Use Case:** **Recommended for fake news detection** where catching misinformation is critical

<img width="790" height="717" alt="Image" src="https://github.com/user-attachments/assets/3690c971-69a1-40b9-b87f-4127adb7c252" />

### Key Insights

#### Why Recall Matters Most
In fake news detection, **high recall is critical** because:
- Undetected fake news can spread rapidly and cause harm
- The cost of missing fake news > cost of false alarms
- Users can verify flagged content, but can't identify unflagged fake news

#### Model Selection Recommendation
**LSTM is the recommended model** because:
1. Achieves the best F1 Score (0.889)
2. Highest recall (85.6%) catches more fake news
3. Still maintains reasonable precision (92.4%)
4. Captures semantic and contextual patterns better than traditional ML

---

## Confusion Matrix Analysis

### Logistic Regression
```
                Predicted
              Real    Fake
Actual Real   7514     46
       Fake    340   1179
```
- False Negatives: 340 (missed fake news)
- False Positives: 46 (incorrectly flagged real news)

### SVM
```
                Predicted
              Real    Fake
Actual Real   7500     60
       Fake    282   1237
```
- False Negatives: 282 (missed fake news) ✓ Better
- False Positives: 60 (incorrectly flagged real news)

### LSTM
```
                Predicted
              Real    Fake
Actual Real   7444    116
       Fake    218   1301
```
- False Negatives: 218 (missed fake news) ✓ **Best**
- False Positives: 116 (incorrectly flagged real news)

**Analysis:** LSTM significantly reduces false negatives (missed fake news) from 340 → 218, a 36% improvement over Logistic Regression.

---

## Conclusions

### Summary of Findings

1. **LSTM Outperforms Traditional ML Models**
   - Achieved the highest F1 Score (0.889) and Recall (0.856)
   - Better at capturing contextual nuances in news text
   - More effective at identifying fake news instances

2. **Class Imbalance Requires Special Handling**
   - Stratified splitting ensured representative train/test sets
   - Class weighting improved minority class (fake news) detection
   - F1 Score and Recall are more informative than accuracy

3. **Text Preprocessing is Critical**
   - HTML cleaning, URL removal, and stop word elimination improved model performance
   - TF-IDF vectorization captured important term frequencies
   - Sequence padding enabled LSTM to process variable-length text

4. **Trade-offs Between Models**
   - **Logistic Regression:** Fast, interpretable, high precision but lower recall
   - **SVM:** Balanced performance, good for production
   - **LSTM:** Best overall performance but requires more resources

### Real-World Implications

For deployment in a fake news detection system:
- **Recommended:** LSTM for its superior recall and F1 score
- **Alternative:** SVM for faster inference with acceptable performance
- **Not Recommended:** Logistic Regression alone (too many false negatives)

### Practical Deployment Considerations

1. **Real-time Detection:** SVM offers faster inference
2. **Batch Processing:** LSTM can process larger volumes with GPU acceleration
3. **Hybrid Approach:** Use SVM for first-pass filtering, LSTM for final verification
4. **Human Review:** Flag borderline cases (confidence < 0.7) for manual review

---

## Future Work

### Recommended Improvements

#### 1. Address Class Imbalance More Explicitly
- **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic fake news examples
- **Undersampling** real news to balance the dataset
- **Cost-sensitive learning** with custom loss functions

#### 2. Advanced Deep Learning Models
- **BERT (Bidirectional Encoder Representations from Transformers)**
  - Pre-trained on large corpora
  - Superior contextual understanding
- **RoBERTa** or **DistilBERT** for efficiency
- **XLNet** for capturing bidirectional context

#### 3. Multi-modal Features
- **Source Credibility Scores:** Incorporate website reputation
- **Author Information:** Track author history and credibility
- **Metadata Features:** Publication date, article length, social shares
- **Network Analysis:** Propagation patterns on social media

#### 4. Explainable AI (XAI)
- **LIME (Local Interpretable Model-agnostic Explanations)** to explain predictions
- **SHAP (SHapley Additive exPlanations)** for feature importance
- **Attention Visualization** to highlight influential words/phrases
- Build trust by showing users WHY an article was flagged

#### 5. Multilingual Support
- Extend to Hindi, Tamil, Telugu, and other Indian languages
- Use multilingual BERT (mBERT) or XLM-R
- Address code-mixing (Hinglish) in Indian social media

#### 6. Real-time Deployment
- **Model Optimization:** Quantization, pruning for faster inference
- **API Development:** RESTful API for integration with news platforms
- **Browser Extension:** Real-time fact-checking while browsing
- **Continuous Learning:** Update model with new fake news patterns

#### 7. Domain-Specific Models
- Separate models for different categories (Politics, Health, Finance)
- Fine-tune on category-specific data for better performance

---

## Technical Specifications

### Hardware Requirements
- **Minimum:** 8GB RAM, CPU (for ML models)
- **Recommended:** 16GB RAM, GPU (for LSTM training)
- **Cloud Option:** Google Colab with GPU runtime (free)

### Training Time
- **Logistic Regression:** ~2-3 minutes
- **SVM:** ~5-10 minutes
- **LSTM:** ~15-30 minutes (3-6 epochs with early stopping)

### Model Sizes
- **Logistic Regression:** ~160 MB (TF-IDF + model)
- **SVM:** ~180 MB (TF-IDF + model)
- **LSTM:** ~12 MB (tokenizer + model weights)

---

## References

### Dataset
- **IFND Dataset:** [Kaggle - Indian Fake News Dataset](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset?resource=download)

### Libraries & Frameworks
- **Pandas:** [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **NumPy:** [https://numpy.org/](https://numpy.org/)
- **NLTK:** [https://www.nltk.org/](https://www.nltk.org/)
- **scikit-learn:** [https://scikit-learn.org/](https://scikit-learn.org/)
- **TensorFlow/Keras:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Matplotlib:** [https://matplotlib.org/](https://matplotlib.org/)
- **Seaborn:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

### Research Papers
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases."
- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers."

---

## Project Structure

```
fake-news-detection/
│
├── fakenews_india_colab_(2).py    # Main Python notebook
├── IFND.csv                       # Dataset (not included, download separately)
├── README.md                      # This file
│
├── saved_models/                  # Saved model files
│   ├── tfidf_vectorizer.joblib    # TF-IDF vectorizer
│   ├── tokenizer.joblib           # LSTM tokenizer
│   └── lstm_model.h5              # Trained LSTM model
│
└── visualizations/                # Generated plots
    ├── model_performance_comparison.png
    ├── model_performance_radar_chart.png
    └── confusion_matrices.png
```

---

## Usage Instructions

### 1. Training Models

```python
# Load and preprocess data
df = pd.read_csv('IFND.csv')
df['text_clean'] = df['Eng_Trans_Statement'].apply(clean)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_stop'], df['Label'], test_size=0.2, stratify=df['Label']
)

# Train LSTM
model.fit(X_train_seq, y_train, validation_split=0.1, epochs=6)
```

### 2. Making Predictions

```python
# Predict on new text
new_text = ["Breaking: Shocking revelation about..."]
new_clean = [clean(text) for text in new_text]
new_seq = pad_sequences(tokenizer.texts_to_sequences(new_clean), maxlen=200)
prediction = model.predict(new_seq)

# Interpret result
label = "Fake" if prediction[0] > 0.5 else "Real"
confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
print(f"Prediction: {label} (Confidence: {confidence:.2%})")
```

### 3. Evaluating Custom Data

```python
# Evaluate on custom dataset
y_pred = (model.predict(X_test_seq) > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## License

This project is for educational and research purposes. Please cite appropriately when using this code or dataset.

---

## Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures (Transformer-based)
- Multilingual support
- Explainability features
- Real-time deployment pipeline
- Web application interface

---

## Contact & Acknowledgments

**Dataset Source:** IFND Dataset contributors on Kaggle  
**Framework:** TensorFlow, scikit-learn, NLTK  
**Environment:** Google Colab

---

## Appendix: Hyperparameter Tuning

### Logistic Regression
- `max_iter`: 1000 → 2000 (improved convergence)
- `class_weight`: 'balanced' (handled imbalance)
- `solver`: 'lbfgs' (default, works well)

### SVM
- `kernel`: 'linear' (best for text)
- `class_weight`: 'balanced'
- `max_iter`: 5000 (increased for convergence)

### LSTM
- `vocabulary_size`: 20000 → 30000 → 40000 (tested)
- `embedding_dim`: 100 (good balance)
- `lstm_units`: 128 (256 with bidirectional)
- `dropout`: 0.5, 0.3 (prevents overfitting)
- `batch_size`: 64 → 128 (faster training)
- `epochs`: 3 → 6 with early stopping

---

**Project Status:** ✅ Completed  
**Last Updated:** 2025  
**Version:** 1.0
