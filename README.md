TextFilter: SMS Spam Classification Model

TextFilter is a machine learning project that classifies SMS messages as either “Spam” or “Real” (ham). This project aims to improve the efficiency of message filtering, ensuring users experience a cleaner, spam-free inbox.

Project Overview

With the increasing number of spam messages, it is crucial to have an automated system to distinguish spam messages from real ones. SpamShield leverages Natural Language Processing (NLP) and machine learning techniques to achieve high accuracy in spam detection.

Features
- Preprocesses SMS messages using text-cleaning techniques like tokenization, lemmatization, and stopword removal.
- Implements various machine learning algorithms for classification, including:
- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machines (SVM)
- Provides accuracy, precision, recall, and F1-score metrics to evaluate model performance.
- Exports a trained model for deployment.


 Tech Stack
- Programming Language: Python
- Libraries/Frameworks:
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Pandas, NumPy
- Matplotlib, Seaborn
- Flask (optional, for API deployment)
- Development Environment: Jupyter Notebook, PyCharm

 Dataset

The project uses the SMS Spam Collection Dataset. This dataset contains a collection of 5,572 SMS messages labeled as:
- Ham: Non-spam messages
- Spam: Unwanted, promotional, or malicious messages.

Data Preprocessing
- Lowercasing all text
- Removing punctuation, numbers, and special characters
- Removing stopwords
- Stemming/Lemmatization


 Model Development

The following steps were followed to build and evaluate the spam classification model:
1. Data Cleaning: Clean and preprocess the SMS messages.
2. Feature Extraction:
3. Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
4. Model Training: Train multiple models, including Logistic Regression, Naive Bayes, and Random Forest.
5. Evaluation: Use metrics like accuracy, precision, recall, and F1-score to evaluate the models.
6. Model Export: Save the best-performing model for deployment using joblib or pickle.
