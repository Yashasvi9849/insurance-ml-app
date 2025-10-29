import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DIR, MODELS_DIR

print("\n" + "="*60)
print("SIMPLE MODEL TRAINING (3 SAMPLES)")
print("="*60 + "\n")

csv_path = PROCESSED_DIR / "extracted_features_fixed.csv"

if not csv_path.exists():
    print(f"❌ Error: {csv_path} not found")
    print("Run: python quick_fix_target.py first")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} samples\n")

print("Target values:")
print(df[['filename', 'approved_benefit_amount']])

target = 'approved_benefit_amount'

exclude_cols = [
    target, 'filename', 'tracking_number', 'pm_explanation', 
    'hold_reason', 'missing_fields', 'extraction_confidence',
    'claim_date', 'approval_date', 'lease_start_date', 
    'lease_end_date', 'move_out_date'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].select_dtypes(include=[np.number])
y = df[target]

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {len(X)}")

X = X.fillna(X.median())
X = X.fillna(0)

print("\n" + "="*60)
print("TRAINING MODEL")
print("-"*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=3,
    random_state=42
)

model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("✅ Training complete\n")
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"\nPredictions vs Actual:")
for i, (actual, pred, fname) in enumerate(zip(y, y_pred, df['filename'])):
    error_pct = abs(actual - pred) / actual * 100
    print(f"\n{fname}")
    print(f"  Actual:    ${actual:,.2f}")
    print(f"  Predicted: ${pred:,.2f}")
    print(f"  Error:     ${abs(actual-pred):,.2f} ({error_pct:.1f}%)")

print(f"\n" + "="*60)
print("OVERALL METRICS")
print("="*60)
print(f"RMSE:     ${rmse:,.2f}")
print(f"MAE:      ${mae:,.2f}")
print(f"R2 Score: {r2:.4f}")

avg_amount = y.mean()
error_pct = (rmse / avg_amount) * 100
print(f"\nError as % of average claim: {error_pct:.1f}%")

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Top 10)")
print("="*60)
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_imp.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODELS_DIR / "simple_model.pkl"
joblib.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, model_path)

print("\n" + "="*60)
print("MODEL SAVED")
print("="*60)
print(f"Location: {model_path}")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if r2 < 0:
    print("⚠️  R2 Score is negative")
    print("   This is EXPECTED with only 3 samples")
    print("   The model memorizes rather than learns patterns")
elif r2 > 0.99:
    print("⚠️  R2 Score is very high")
    print("   This is overfitting - model memorized the 3 samples")
else:
    print("✅ Model trained successfully")

print("\n💡 With only 3 samples:")
print("   • Model has memorized these specific examples")
print("   • Predictions on new data will be unreliable")
print("   • Need 20-50 samples for production use")

print("\n✅ However, the PIPELINE WORKS!")
print("   • Extraction ✅")
print("   • Feature engineering ✅")
print("   • Model training ✅")
print("   • Model saving ✅")

print("\n📈 Next steps:")
print("   1. Add more 'Move Out Calculation' PDFs")
print("   2. Re-run extraction: python test_extraction.py")
print("   3. Re-run this training with more data")
print("   4. Test predictions: python predict_new.py <path_to_pdf>")

print("\n" + "="*60 + "\n")