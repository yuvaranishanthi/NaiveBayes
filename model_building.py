import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# 1. Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Preprocess text
df['message'] = df['message'].str.lower().str.strip()

# Encode labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 4. Vectorize text (✅ keep stopwords so banking patterns remain)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  
# ngram_range=(1,2) → uses single words + 2-word phrases ("your account")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Naive Bayes model
model = MultinomialNB(alpha=0.3)  # ✅ Lower alpha = more sensitive to rare spam words
model.fit(X_train_tfidf, y_train)

# 6. Save model & vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

# 7. Evaluate
acc = model.score(X_test_tfidf, y_test)
print(f"✅ Model trained successfully — Accuracy: {acc:.2%}")

# 8. Quick test
test_msg = "Congrats! your account has been credited with 1000000"
test_vec = vectorizer.transform([test_msg.lower()])
pred = model.predict(test_vec)[0]
prob = model.predict_proba(test_vec)[0][1] * 100
print(f"Test Message: {test_msg}")
print(f"Prediction: {'Spam' if pred == 1 else 'Ham'} ({prob:.2f}% spam probability)")

