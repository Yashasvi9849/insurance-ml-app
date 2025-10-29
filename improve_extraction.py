import json
import re
from pathlib import Path

def extract_amounts_flexible(text):
    """More flexible amount extraction"""
    amounts = []
    
    # Pattern 1: Total Balance Due (original)
    pattern1 = r'Total\s+Balance\s+Due:\s*\$?\s*([\d,]+\.?\d*)'
    matches = re.findall(pattern1, text, re.IGNORECASE)
    amounts.extend([float(m.replace(',', '')) for m in matches])
    
    # Pattern 2: Total Amount, Total Due, Amount Due
    pattern2 = r'(?:Total\s+)?(?:Amount|Balance|Due):\s*\$\s*([\d,]+\.?\d*)'
    matches = re.findall(pattern2, text, re.IGNORECASE)
    amounts.extend([float(m.replace(',', '')) for m in matches])
    
    # Pattern 3: Claim Amount
    pattern3 = r'Claim\s+Amount:\s*\$?\s*([\d,]+\.?\d*)'
    matches = re.findall(pattern3, text, re.IGNORECASE)
    amounts.extend([float(m.replace(',', '')) for m in matches])
    
    # Pattern 4: Dollar amounts on lines with "total" or "balance"
    pattern4 = r'(?:total|balance|amount|due).*?\$\s*([\d,]+\.\d{2})'
    matches = re.findall(pattern4, text, re.IGNORECASE)
    amounts.extend([float(m.replace(',', '')) for m in matches])
    
    # Return the largest reasonable amount (likely the total)
    valid_amounts = [a for a in amounts if 100 < a < 50000]  # Filter reasonable claim amounts
    return max(valid_amounts) if valid_amounts else None

# Load your JSON
json_path = Path('data/processed/extracted_features.json')
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"Processing {len(data)} records...")
print("="*70)

improved = 0
for record in data:
    # If amount_of_claim is missing or null
    if not record.get('amount_of_claim'):
        # Try to extract from full_text if available
        if 'full_text' in record and record['full_text']:
            amount = extract_amounts_flexible(record['full_text'])
            if amount:
                record['amount_of_claim'] = amount
                record['approved_benefit_amount'] = amount
                improved += 1
                print(f"✅ {record.get('filename', 'unknown')[:50]}: ${amount:.2f}")

print("="*70)
print(f"Improved: {improved} records")
print(f"Original with amounts: 6")
print(f"New total: {improved + 6}")

# Save back
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Updated JSON saved")
print("\nNow run:")
print("  1. python -c 'import pandas as pd, json; df=pd.DataFrame(json.load(open(\"data/processed/extracted_features.json\"))); df.to_csv(\"data/processed/extracted_features_fixed.csv\", index=False)'")
print("  2. python modules/data_processor.py")
