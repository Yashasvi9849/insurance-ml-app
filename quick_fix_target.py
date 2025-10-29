import pandas as pd
from pathlib import Path

print("\n" + "="*60)
print("FIXING TARGET VARIABLE")
print("="*60 + "\n")

input_path = Path('data/processed/extracted_features.csv')
output_path = Path('data/processed/extracted_features_fixed.csv')

if not input_path.exists():
    print(f"❌ Error: {input_path} not found")
    print("Run: python test_extraction.py first")
    exit(1)

df = pd.read_csv(input_path)

print(f"Original data: {len(df)} rows")
print(f"approved_benefit_amount non-null: {df['approved_benefit_amount'].notna().sum()}")
print(f"amount_of_claim non-null: {df['amount_of_claim'].notna().sum()}")

if df['approved_benefit_amount'].isna().all():
    print("\n⚠️  No approved amounts extracted")
    
    if df['amount_of_claim'].notna().any():
        print("✅ Using amount_of_claim as proxy target")
        df['approved_benefit_amount'] = df['amount_of_claim']
    else:
        print("❌ No suitable amounts found")
        print("\nTrying late_fees_total as last resort...")
        if df['late_fees_total'].notna().any():
            df['approved_benefit_amount'] = df['late_fees_total']
            print("✅ Using late_fees_total")
        else:
            print("❌ No valid amounts in any column")
            print("\nYou need to manually create training_labels.csv")
            exit(1)

df_valid = df[df['approved_benefit_amount'].notna()].copy()

print(f"\nAfter fix:")
print(f"Valid training samples: {len(df_valid)}")

if len(df_valid) == 0:
    print("\n❌ ERROR: No valid target values even after fix")
    print("\nYou need to:")
    print("  1. Check PDF extraction quality")
    print("  2. Manually create data/training_labels.csv")
    exit(1)

if len(df_valid) < 3:
    print(f"\n⚠️  WARNING: Only {len(df_valid)} samples")
    print("Model training needs at least 3 samples")
    print("Consider adding manual labels for more PDFs")

print(f"\nSample of fixed data:")
print(df_valid[['filename', 'amount_of_claim', 'approved_benefit_amount']].head())

df_valid.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")
print(f"\n📊 Ready for training with {len(df_valid)} samples")
print(f"\nNext step: python train_model_fixed.py")
print("="*60 + "\n")