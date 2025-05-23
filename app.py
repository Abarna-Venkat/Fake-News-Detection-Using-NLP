import streamlit as st
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load
import nltk

# Load model and vectorizer
model = load('model.joblib')
tfidf = load('tfidf.joblib')

# Preprocessing
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load original datasets for download
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake['label'] = 0
real['label'] = 1
data = pd.concat([fake, real], axis=0).reset_index(drop=True)

# --- UI Starts Here ---

st.title("📰 Fake News Detection with NLP")

# ⬇️ Dataset download button
st.download_button(
    label="📥 Download Combined News Dataset (CSV)",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='news_dataset.csv',
    mime='text/csv'
)

# 🧪 Sample news examples
st.subheader("Try a Sample News Article:")

sample_news = {
    "🟢 Real News Sample": "NASA has announced a new mission to send a rover to explore the icy moons of Jupiter by 2030.",
    "🔴 Fake News Sample": "The moon is actually a hologram created by NASA to cover alien bases."
}

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Real News Sample"):
        st.session_state["user_input"] = sample_news["🟢 Real News Sample"]

with col2:
    if st.button("🔴 Fake News Sample"):
        st.session_state["user_input"] = sample_news["🔴 Fake News Sample"]

# ✏️ Input area
user_input = st.text_area("✏️ Enter a news article or headline", value=st.session_state.get("user_input", ""))

# 🔍 Detect button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.success("🟢 This is Real News!")
        else:
            st.error("🔴 This is Fake News!")
