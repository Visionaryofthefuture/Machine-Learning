import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

df = pd.read_csv("/home/soumyajit-das/Downloads/fake_or_real_news.csv")
df['label'] = df['label'].apply(lambda x : 0 if x == "REAL" else 1)
df = df.drop("id", axis = 1)
vectorizer = TfidfVectorizer(stop_words = "english" , max_df = 0.7)
X,y = df["text"] , df["label"]
X_train , X_test , Y_train , Y_test = train_test_split(X,y, test_size = 0.2)
X_train_vectorizer =  vectorizer.fit_transform(X_train)
X_test_vectorizer = vectorizer.transform(X_test)
classifier = LinearSVC()
print(X_train_vectorizer)
classifier.fit(X_train_vectorizer, Y_train)
