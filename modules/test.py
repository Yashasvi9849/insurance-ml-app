import pandas as pd
df = pd.read_csv("data/processed/extracted_features.csv")

print("Columns:", df.columns.tolist())
print("Non-null count:", df['approved_benefit_amount'].notnull().sum() if 'approved_benefit_amount' in df.columns else "Column missing")
print(df.head(3))
