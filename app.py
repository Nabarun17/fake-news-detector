import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')

df = pd.read_csv('train.csv')
df['content'] = df['author'] + " " + df['title']
df['content'] = df['content'].fillna('')

ps = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return " ".join(content)

df['content'] = df['content'].apply(stemming)
x = df['content'].values
y = df['label'].values

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

st.title('Fake News Detector')

input_text = st.text_input('Enter news article:')

if st.button('Predict'):
    input_processed = stemming(input_text)
    input_vectorized = vectorizer.transform([input_processed])
    prediction = model.predict(input_vectorized)
    if prediction[0] == 1:
        st.write("This news article is likely **FAKE**.")
    else:
        st.write("This news article is likely **REAL**.")

if st.button('Show Model Accuracy'):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2%}")
