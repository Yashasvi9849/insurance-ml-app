import json
import re
from pathlib import Path

def extract_amounts_flexible(text):
    """Extract amounts with multiple fallback patterns"""
    
    results = {
        'method': None,
        'amount': None,
        'confidence': 'low'
    }
    
    # Pattern 1: Total Balance Due (highest confidence)
    match = re.search(r'Total\s+Balance\s+Due[:\s]*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        results['amount'] = float(match.group(1).replace(',', ''))
        results['method'] = 'Total Balance Due'
        results['confidence'] = 'high'
        return results
    
    # Pattern 2: Move Out Calculation (claim-specific)
    if 'move out calculation' in text.lower():
        match = re.search(r'Total[:\s]*\$\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if match:
            results['amount'] = float(match.group(1).replace(',', ''))
            results['method'] = 'Move Out Total'
            results['confidence'] = 'high'
            return results
    
    # Pattern 3: Claim Amount
    match = re.search(r'Claim\s+Amount[:\s]*\$\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
    if match:
        results['amount'] = float(match.group(1).replace(',', ''))
        results['method'] = 'Claim Amount'
        results['confidence'] = 'high'
        return results
    
    # Pattern 4: Balance Due
    match = re.search(r'Balance\s+Due[:\s]*\$\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
    if match:
        results['amount'] = float(match.group(1).replace(',', ''))
        results['method'] = 'Balance Due'
        results['confidence'] = 'medium'
        return results
    
    # Pattern 5: Amount Due
    match = re.search(r'Amount\s+Due[:\s]*\$\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
    if match:
        results['amount'] = float(match.group(1).replace(',', ''))
        results['method'] = 'Amount Due'
        results['confidence'] = 'medium'
        return results
    
    # Pattern 6: Find largest reasonable amount (lowest confidence)
    all_amounts = re.findall(r'\$\s*([\d,]+\.\d{2})', text)
    if all_amounts:
        amounts = [float(a.replace(',', '')) for a in all_amounts]
        # Filter to reasonable claim amounts
        valid = [a for a in amounts if 100 < a < 50000]
        if valid:
            results['amount'] = max(valid)
            results['method'] = 'Largest amount (guess)'
            results['confidence'] = 'low'
            return results
    
    return results

def classify_document(text, filename):
    """Determine document type"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    # Check for claim-related documents
    claim_indicators = ['move out calculation', 'claim', 'closeout', 'balance due', 'sdrp']
    is_claim_doc = any(indicator in text_lower or indicator in filename_lower for indicator in claim_indicators)
    
    # Check for lease agreements
    lease_indicators = ['lease agreement', 'rental agreement', 'residential lease']
    is_lease = any(indicator in text_lower for indicator in lease_indicators)
    
    if is_claim_doc:
        return 'CLAIM_DOCUMENT'
    elif is_lease:
        return 'LEASE_AGREEMENT'
    elif 'letter' in filename_lower or 'mail' in filename_lower:
        return 'LETTER'
    else:
        return 'UNKNOWN'

# Load JSON
json_path = Path('data/processed/extracted_features.json')
with open(json_path, 'r') as f:
    data = json.load(f)

print("="*70)
print("🔍 ANALYZING FIRST 15 PDFs")
print("="*70)

for i, record in enumerate(data[:15], 1):
    filename = record['filename']
    text = record.get('full_text', '')
    
    print(f"\n{i}. {filename}")
    print("-" * 70)
    
    # Classify document
    doc_type = classify_document(text, filename)
    print(f"   📄 Document Type: {doc_type}")
    
    # Try to extract amount
    extraction = extract_amounts_flexible(text)
    
    if extraction['amount']:
        print(f"   💰 Amount Found: ${extraction['amount']:,.2f}")
        print(f"   🎯 Method: {extraction['method']}")
        print(f"   📊 Confidence: {extraction['confidence']}")
    else:
        print(f"   ❌ No amount found")
        
        # Show what's in the document
        if doc_type == 'LEASE_AGREEMENT':
            print(f"   ℹ️  This is a lease agreement - amounts not expected")
        elif doc_type == 'LETTER':
            print(f"   ℹ️  This is a letter - may not contain claim amounts")
        else:
            # Show dollar amounts found (for debugging)
            dollar_lines = [line.strip() for line in text.split('\n') if '$' in line][:2]
            if dollar_lines:
                print(f"   💡 Sample lines with $:")
                for line in dollar_lines:
                    print(f"      {line[:70]}")
            else:
                print(f"   ℹ️  No dollar amounts found in text")
    
    # Check what's currently in the record
    current_amount = record.get('amount_of_claim')
    if current_amount:
        print(f"   ✅ Already extracted: ${current_amount:.2f}")

# Summary
print(f"\n{'='*70}")
print("📊 SUMMARY OF FIRST 15 PDFs:")
print('='*70)

doc_types = {}
extractable = 0
claim_docs = 0

for record in data[:15]:
    text = record.get('full_text', '')
    doc_type = classify_document(text, record['filename'])
    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    if doc_type == 'CLAIM_DOCUMENT':
        claim_docs += 1
    
    extraction = extract_amounts_flexible(text)
    if extraction['amount']:
        extractable += 1

print(f"\nDocument types found:")
for dtype, count in doc_types.items():
    print(f"   {dtype}: {count}")

print(f"\nClaim documents (should have amounts): {claim_docs}")
print(f"Documents where amounts were found: {extractable}")

if extractable < claim_docs:
    print(f"\n⚠️  Found {claim_docs} claim docs but only extracted {extractable} amounts")
    print(f"   → Need better extraction patterns")
elif extractable > claim_docs:
    print(f"\n✅ Good! Found amounts in some non-claim docs too")
else:
    print(f"\n✅ Perfect! Extracted from all claim documents")

print("="*70)
