import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASETS
# =========================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

fake_df = pd.read_csv("../data/Fake.csv")
true_df = pd.read_csv("../data/True.csv")

# =========================
# ADD LABELS
# =========================

fake_df["label"] = 0
true_df["label"] = 1

# =========================
# COMBINE DATASETS
# =========================

df = pd.concat([fake_df, true_df])

# =========================
# KEEP REQUIRED COLUMNS
# =========================

df = df[["text", "label"]]

# =========================
# CHECK DATA
# =========================

print(df.head())

# =========================
# DATASET DISTRIBUTION
# =========================

plt.figure(figsize=(6,4))

sns.countplot(
    x='label',
    data=df
)

plt.xticks([0,1], ['Fake', 'Real'])

plt.title("Fake vs Real News Distribution")

plt.savefig("../outputs/news_distribution.png")

plt.show()

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    text = " ".join(words)

    return text

# =========================
# APPLY CLEANING
# =========================

df["text"] = df["text"].apply(clean_text)

# =========================
# WORD COUNT ANALYSIS
# =========================

df["word_count"] = df["text"].apply(
    lambda x: len(x.split())
)

fake_avg = df[df["label"] == 0]["word_count"].mean()

real_avg = df[df["label"] == 1]["word_count"].mean()

print("\nAverage Fake News Word Count:", round(fake_avg, 2))

print("Average Real News Word Count:", round(real_avg, 2))

# =========================
# FEATURES AND TARGET
# =========================

X = df["text"]

y = df["label"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression()

model.fit(X_train_vec, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test_vec)

# =========================
# ACCURACY
# =========================

acc = accuracy_score(y_test, y_pred) * 100

print(f"\nAccuracy: {acc:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("../outputs/confusion_matrix.png")

plt.show()

# =========================
# TEST CUSTOM NEWS
# =========================

news = input("\nEnter News Article:\n")

news_clean = clean_text(news)

news_vector = vectorizer.transform([news_clean])

prediction = model.predict(news_vector)

confidence = model.predict_proba(news_vector)

confidence_score = max(confidence[0]) * 100

if prediction[0] == 0:
    print(f"\nPrediction: FAKE NEWS ({confidence_score:.2f}% confidence)")
else:
    print(f"\nPrediction: REAL NEWS ({confidence_score:.2f}% confidence)")