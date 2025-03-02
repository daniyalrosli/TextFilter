TextFilter: SMS Spam Classification Model

TextFilter is a machine learning project that classifies SMS messages as either “Spam” or “Real” (ham). This project aims to improve the efficiency of message filtering, ensuring users experience a cleaner, spam-free inbox.

Project Overview

With the increasing number of spam messages, it is crucial to have an automated system to distinguish spam messages from real ones. TextFilter leverages Natural Language Processing (NLP) and machine learning techniques to achieve high accuracy in spam detection.


Features:

- Preprocesses SMS messages using text-cleaning techniques like tokenization, lemmatization, and stopword removal.

- Implements various machine learning algorithms for classification, including:

- Logistic Regression

- Naive Bayes

- Random Forest

- Support Vector Machines (SVM)

- Provides accuracy, precision, recall, and F1-score metrics to evaluate model performance.

- Exports a trained model for deployment.

Tech Stack:

- Programming Language: Python

Libraries/Frameworks:

- Scikit-learn

- NLTK (Natural Language Toolkit)

- Pandas, NumPy

- Matplotlib, Seaborn

- FastAPI (for API deployment)

- Development Environment: Jupyter Notebook


Dataset

The project uses the SMS Spam Collection Dataset, containing 5,572 SMS messages labeled as:

- Ham: Non-spam messages

- Spam: Unwanted, promotional, or malicious messages.

Data Preprocessing:

- Lowercasing all text

- Removing punctuation, numbers, and special characters

- Removing stopwords

- Stemming/Lemmatization

Model Development

The following steps were followed to build and evaluate the spam classification model:

- Data Cleaning: Clean and preprocess the SMS messages.

 Feature Extraction: Convert text into numerical features using:

- CountVectorizer

- TF-IDF (Term Frequency-Inverse Document Frequency)

- Model Training: Train multiple models, including Logistic Regression, Naive Bayes, and Random Forest.

- Evaluation: Use accuracy, precision, recall, and F1-score to assess model performance.

- Model Export: Save the best-performing model for deployment using joblib.


Results

The models were evaluated based on accuracy scores using both CountVectorizer and TF-IDF.

Model

Accuracy (CountVectorizer)

Accuracy (TF-IDF)

Logistic Regression

0.9754

0.9721

Naive Bayes

0.9839 (Best)

0.9732

Random Forest

0.9705

0.9658

SVM

0.9792

0.9785 (Best)

Best Models:

Naive Bayes performed best with CountVectorizer (Accuracy: 98.39%)

SVM performed best with TF-IDF (Accuracy: 97.85%)
