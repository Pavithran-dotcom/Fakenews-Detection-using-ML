import pandas as pd

print("Converting Kaggle dataset...")

# Load from data folder
fake_df = pd.read_csv('data/Fake.csv')  # Note: data/ prefix
true_df = pd.read_csv('data/True.csv')

print(f"Loaded {len(fake_df)} fake articles")
print(f"Loaded {len(true_df)} real articles")

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine title + text if separate columns
if 'title' in fake_df.columns:
    fake_df['text'] = fake_df['title'] + ' ' + fake_df['text']
    true_df['text'] = true_df['title'] + ' ' + true_df['text']

# Keep only needed columns
fake_df = fake_df[['text', 'label']]
true_df = true_df[['text', 'label']]

# Combine and shuffle
combined = pd.concat([fake_df, true_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save back to data folder
combined.to_csv('data/news_dataset.csv', index=False)

print(f"\n✅ Success! Saved {len(combined)} articles")
print(f"   - Fake: {len(fake_df)}")
print(f"   - Real: {len(true_df)}")