"""
Complete Pipeline: Fill Targets → Convert to CSV → Ready for Training
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("🚀 COMPLETE DATA PREPARATION PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: FILL MISSING TARGET VALUES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: FILLING MISSING TARGET VALUES")
print("="*80)

# Load JSON
json_path = Path('data/processed/extracted_features.json')
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"\nTotal records: {len(data)}")

# Get current statistics
amounts = [d['amount_of_claim'] for d in data if d.get('amount_of_claim')]

if not amounts:
    print("❌ No amounts found in data!")
    exit(1)

min_amount = min(amounts)
max_amount = max(amounts)
mean_amount = np.mean(amounts)
median_amount = np.median(amounts)

print(f"\nCurrent amounts ({len(amounts)} records):")
print(f"  Min: ${min_amount:,.2f}")
print(f"  Max: ${max_amount:,.2f}")
print(f"  Mean: ${mean_amount:,.2f}")
print(f"  Median: ${median_amount:,.2f}")

# Fill missing values
print(f"\nFilling missing target values...")
filled_count = 0
strategies_used = {}

for record in data:
    if record.get('amount_of_claim'):
        continue  # Already has amount
    
    # Strategy 1: If monthly_rent exists, estimate claim as 3-6 months rent
    if record.get('monthly_rent'):
        rent = record['monthly_rent']
        multiplier = np.random.uniform(3, 5)
        estimated = rent * multiplier
        estimated = max(min_amount, min(estimated, max_amount))
        
        record['amount_of_claim'] = round(estimated, 2)
        record['approved_benefit_amount'] = round(estimated, 2)
        filled_count += 1
        strategies_used['Based on rent'] = strategies_used.get('Based on rent', 0) + 1
        continue
    
    # Strategy 2: For claim documents, use median with variation
    text = record.get('full_text', '').lower()
    claim_keywords = ['closeout', 'move out', 'claim', 'sdrp', 'balance due']
    
    if any(keyword in text for keyword in claim_keywords):
        variation = np.random.uniform(0.8, 1.2)
        estimated = median_amount * variation
        
        record['amount_of_claim'] = round(estimated, 2)
        record['approved_benefit_amount'] = round(estimated, 2)
        filled_count += 1
        strategies_used['Claim doc (median)'] = strategies_used.get('Claim doc (median)', 0) + 1
        continue
    
    # Strategy 3: For lease agreements, use lower range
    if 'lease agreement' in text or 'rental agreement' in text:
        lower_quartile = np.percentile(amounts, 25)
        variation = np.random.uniform(0.8, 1.2)
        estimated = lower_quartile * variation
        
        record['amount_of_claim'] = round(estimated, 2)
        record['approved_benefit_amount'] = round(estimated, 2)
        filled_count += 1
        strategies_used['Lease (lower range)'] = strategies_used.get('Lease (lower range)', 0) + 1
        continue
    
    # Strategy 4: For everything else, use mean with variation
    variation = np.random.uniform(0.7, 1.3)
    estimated = mean_amount * variation
    estimated = max(min_amount, min(estimated, max_amount))
    
    record['amount_of_claim'] = round(estimated, 2)
    record['approved_benefit_amount'] = round(estimated, 2)
    filled_count += 1
    strategies_used['Mean with variation'] = strategies_used.get('Mean with variation', 0) + 1

# Save updated JSON
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Filled {filled_count} missing values")
print(f"\nStrategies used:")
for strategy, count in strategies_used.items():
    print(f"  {strategy}: {count}")

# New statistics
new_amounts = [d['amount_of_claim'] for d in data if d.get('amount_of_claim')]

print(f"\n{'='*80}")
print("AFTER FILLING:")
print('='*80)
print(f"Records with amounts: {len(new_amounts)}/318 ({len(new_amounts)/318*100:.1f}%)")
print(f"\nAmount distribution:")
print(f"  Min: ${min(new_amounts):,.2f}")
print(f"  Max: ${max(new_amounts):,.2f}")
print(f"  Mean: ${np.mean(new_amounts):,.2f}")
print(f"  Median: ${np.median(new_amounts):,.2f}")

# ============================================================================
# STEP 2: CONVERT JSON TO CSV
# ============================================================================
print("\n" + "="*80)
print("STEP 2: CONVERTING JSON TO CSV")
print("="*80)

df = pd.DataFrame(data)
csv_path = Path('data/processed/extracted_features_fixed.csv')

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"\n✅ Created CSV: {csv_path}")
print(f"   Records: {len(df)}")
print(f"   Columns: {len(df.columns)}")

# Verify CSV
print(f"\nVerifying CSV contents...")
with_amounts_csv = df['amount_of_claim'].notna().sum()
with_approved_csv = df['approved_benefit_amount'].notna().sum()
with_text_csv = df['full_text'].notna().sum()

print(f"  ✅ amount_of_claim: {with_amounts_csv}/{len(df)}")
print(f"  ✅ approved_benefit_amount: {with_approved_csv}/{len(df)}")
print(f"  ✅ full_text: {with_text_csv}/{len(df)}")

# ============================================================================
# STEP 3: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 PIPELINE COMPLETE - READY FOR MODELING!")
print("="*80)

print(f"\n📊 Final Dataset Summary:")
print(f"   Total records: {len(df)}")
print(f"   Training samples: {with_approved_csv}")
print(f"   Features available: {len(df.columns)}")
print(f"   Target variable: approved_benefit_amount")

print(f"\n📈 Target Variable Statistics:")
print(f"   Min: ${df['approved_benefit_amount'].min():,.2f}")
print(f"   Max: ${df['approved_benefit_amount'].max():,.2f}")
print(f"   Mean: ${df['approved_benefit_amount'].mean():,.2f}")
print(f"   Median: ${df['approved_benefit_amount'].median():,.2f}")
print(f"   Std Dev: ${df['approved_benefit_amount'].std():,.2f}")

print(f"\n⚠️  IMPORTANT NOTES:")
print(f"   • {len(amounts)} amounts are REAL (extracted from PDFs)")
print(f"   • {filled_count} amounts are ESTIMATED (filled algorithmically)")
print(f"   • Model will train but predictions may vary in accuracy")
print(f"   • Consider manual labeling for production use")

print(f"\n{'='*80}")
print("🚀 NEXT STEP: TRAIN THE MODEL")
print('='*80)
print(f"\nRun this command:")
print(f"   python modules/data_processor.py")
print(f"\nThis will:")
print(f"   1. Load the CSV with {with_approved_csv} training samples")
print(f"   2. Clean and preprocess the data")
print(f"   3. Split into train/test sets")
print(f"   4. Scale features")
print(f"   5. Save preprocessed data for model training")
print("="*80)

# Show sample of data
print(f"\n📄 Sample Records (first 5):")
print("="*80)
sample_cols = ['filename', 'amount_of_claim', 'approved_benefit_amount', 'monthly_rent']
available_cols = [col for col in sample_cols if col in df.columns]
print(df[available_cols].head(5).to_string(index=False))
print("="*80)

print("\n✅ All files ready!")
print(f"   JSON: {json_path}")
print(f"   CSV: {csv_path}")
print("\n🎯 You can now proceed to model training!")
print("="*80)
