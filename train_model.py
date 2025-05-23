import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake['label'] = 0
real['label'] = 1

data = pd.concat([fake, real]).reset_index(drop=True)
data['clean_text'] = data['text'].apply(clean_text)

X = data['clean_text']
y = data['label']

tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
dump(model, 'model.joblib')
dump(tfidf, 'tfidf.joblib')
print("✅ Model and vectorizer saved!")
