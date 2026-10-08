import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("TRAINING FAKE NEWS DETECTOR")
print("=" * 60)

# ── Load ONLY Fake.csv + True.csv (proven 99% accuracy) ──────
print("\nLoading Fake.csv and True.csv...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

fake_df['label'] = 0
true_df['label'] = 1

# Combine title + text
fake_df['combined'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
true_df['combined'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')

df = pd.concat([fake_df[['combined','label']], true_df[['combined','label']]], ignore_index=True)
df = df[df['combined'].str.len() > 50].dropna(subset=['combined'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total: {len(df)} | Real: {sum(df['label']==1)} | Fake: {sum(df['label']==0)}")

X_train, X_test, y_train, y_test = train_test_split(
    df['combined'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")
print("\nTraining model... (2-5 mins)")

model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Fake','Real']))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fake_news_model.pkl')
print("✅ Model saved! Restart with: python app.py")
