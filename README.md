as3edrfghy# Fake News Detection for Indian News

Fake news poses a significant threat to society by spreading misinformation, influencing public opinion, and eroding trust in legitimate news sources. In the context of Indian news, where a large and diverse population relies on various platforms for information, detecting fake news is crucial to maintain a well-informed citizenry and prevent the spread of harmful narratives. This project aims to address this challenge by developing machine learning and deep learning models to classify Indian news as either real or fake. We explore various approaches, including Logistic Regression, SVM, and LSTM. Based on our experiments, the LSTM model demonstrated the highest accuracy in identifying fake news.

## Dataset and Preprocessing

The dataset used in this project is named `IFND.csv`. It contains information about Indian news articles and their corresponding labels (Real or Fake).

The dataset has the following dimensions:
- Number of rows: 45393
- Number of columns: 9

The dataset was loaded from the path `/content/IFND.csv`.

### Text Preprocessing

The 'Statement' column, containing the news text, underwent several preprocessing steps to prepare it for model training:
1.  **Lowercase Conversion:** All text was converted to lowercase.
2.  **HTML Entity Removal:** HTML entities were decoded.
3.  **URL Removal:** URLs starting with 'http' or 'https' were removed.
4.  **Non-alphanumeric Character Removal:** Characters other than lowercase letters, numbers, and spaces were removed.
5.  **Whitespace Standardization:** Multiple spaces were replaced with a single space, and leading/trailing spaces were removed.
6.  **Stop Word Removal:** Common English stop words (e.g., 'the', 'a', 'is') were removed using the `nltk.corpus.stopwords` list.

These steps resulted in two new columns:
-   `text_clean`: Contains the text after steps 1-5.
-   `text_stop`: Contains the text after all six steps, including stop word removal.

### Label Mapping

The 'Label' column, which originally contained 'Fake' and 'True' values, was converted into a binary numerical format:
-   'Fake' was mapped to the integer value 1.
-   'True' was mapped to the integer value 0.

Missing values were dropped, and the index was reset.

### Train-Test Split

The preprocessed text data (`text_stop`) and the numerical labels (`Label`) were split into training and testing sets.
-   80% of the data was allocated to the training set (`X_train`, `y_train`).
-   20% of the data was allocated to the testing set (`X_test`, `y_test`).

The split was performed using `sklearn.model_selection.train_test_split` with `stratify=y` to ensure that the proportion of fake and real news is the same in both the training and testing sets, addressing the class imbalance observed in the 'Label' column. A `random_state` of 42 was used for reproducibility.

## Methods

### Machine Learning Models

Two traditional machine learning models were employed as baseline classifiers for the text data:

*   **Logistic Regression:** Chosen for its simplicity and effectiveness as a linear model for binary classification tasks like fake news detection. It provides a good starting point for evaluating the problem's complexity.
*   **Support Vector Machine (SVM):** A powerful and versatile model that works well for classification tasks by finding the optimal hyperplane to separate classes. The linear kernel was used, which is often effective for text classification with high-dimensional feature spaces like TF-IDF.

These models were chosen to establish performance benchmarks against which the deep learning model could be compared.

### Deep Learning Model

A deep learning model based on the Long Short-Term Memory (LSTM) architecture was implemented to capture sequential dependencies in the text data.

*   **Long Short-Term Memory (LSTM):** LSTMs are a type of recurrent neural network (RNN) particularly well-suited for processing sequential data like text. They can learn and remember long-term dependencies in the input sequence, which is crucial for understanding the context and nuances in news articles that might indicate their authenticity. The model architecture included an Embedding layer to represent words as dense vectors, an LSTM layer to process the sequence, a Dropout layer for regularization, and a final Dense layer with a sigmoid activation for binary classification.

## Experiments and Results Summary
After preprocessing the data and splitting it into training and testing sets, three different models were trained and evaluated on the task of classifying news articles as real (0) or fake (1): Logistic Regression, Support Vector Machine (SVM), and a Long Short-Term Memory (LSTM) deep learning model.

The performance of each model was assessed using standard classification metrics: Accuracy, F1 Score, Precision, and Recall. The results are summarized below and visualized in the accompanying bar plot:

<img width="989" height="590" alt="download" src="https://github.com/user-attachments/assets/f16613e3-b313-44a5-bdd9-c4321cdabf0e" />

| Metric     | Logistic Regression | SVM   | LSTM  |
|------------|---------------------|-------|-------|
| Accuracy   | 0.959               | 0.964 | 0.964 |
| F1 Score   | 0.863               | 0.883 | 0.889 |
| Precision  | 0.974               | 0.965 | 0.924 |
| Recall     | 0.776               | 0.814 | 0.856 |

*(Note: Metrics are rounded to three decimal places for clarity.)*
As visually represented in the bar plot above, all three models achieved high accuracy, with Logistic Regression at approximately 95.9%, SVM at 96.4%, and LSTM also at 96.4%. While accuracy is a useful metric, it can be misleading in the presence of class imbalance, which is present in our dataset (more real news than fake news). Therefore, the F1 Score, which is the harmonic mean of Precision and Recall, provides a more balanced evaluation, especially for the minority class (fake news).
*   **Logistic Regression:** This model performed well as a baseline, showing high precision but lower recall. This suggests it is good at identifying real news (high precision on the positive class, which is fake news, means it doesn't label many real news as fake), but it misses a significant portion of the actual fake news (lower recall).
*   **SVM:** The SVM model showed improved performance over Logistic Regression across all metrics, particularly in F1 Score and Recall. It demonstrates a better balance between Precision and Recall compared to Logistic Regression.
*   **LSTM:** The LSTM model achieved the highest F1 Score and Recall among the three models. This indicates that the deep learning approach is more effective at identifying a larger proportion of the actual fake news instances while maintaining a reasonable level of precision. The ability of LSTM networks to capture sequential context in text likely contributes to their better performance on this task.
In the context of fake news detection, maximizing Recall is often critical to minimize the number of fake news articles that go undetected. While Precision is also important to avoid flagging real news as fake, a higher Recall ensures that more harmful misinformation is caught. Based on the F1 Score and Recall metrics, the LSTM model appears to be the most effective for this specific fake news detection problem, demonstrating the best balance between identifying fake news instances and minimizing false positives compared to the other models. The visual comparison plot clearly illustrates these differences in performance across the metrics.

## Conclusion
This project successfully explored the application of traditional machine learning models (Logistic Regression, SVM) and a deep learning model (LSTM) for fake news detection in the context of Indian news. The data analysis revealed a significant class imbalance, highlighting the importance of metrics beyond simple accuracy, such as F1 Score and Recall, for a comprehensive evaluation.
The experiments showed that while all models achieved high accuracy, the LSTM model demonstrated superior performance in terms of F1 Score and Recall. This indicates that the LSTM, with its ability to capture sequential dependencies in text, is more effective at identifying fake news instances, which is crucial for minimizing the spread of misinformation. The traditional ML models, while providing strong baselines, were slightly less effective at recalling fake news compared to the LSTM.
Key takeaways from this project include the importance of robust text preprocessing, the impact of class imbalance on model evaluation, and the potential of deep learning architectures like LSTMs for complex natural language processing tasks such as fake news detection.

Future work could involve addressing the class imbalance more explicitly using techniques like oversampling or undersampling, experimenting with more advanced transformer-based models (e.g., BERT), incorporating other features like author information or publication source credibility, and exploring explainable AI techniques to understand why models classify certain news as fake.

## References

*   **Dataset Source:** The dataset `IFND.csv` is from the Fake News Detection for Indian News repository/source (https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset?resource=download).
*   **Libraries Used:**
    *   [Pandas](https://pandas.pydata.org/)
    *   [NumPy](https://numpy.org/)
    *   [nltk](https://www.nltk.org/)
    *   [scikit-learn](https://scikit-learn.org/stable/)
    *   [TensorFlow](https://www.tensorflow.org/)
    *   [Matplotlib](https://matplotlib.org/)
    *   [html](https://docs.python.org/3/library/html.html)
    *   [re](https://docs.python.org/3/library/re.html)
