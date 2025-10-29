import re
from modules.pdf_extractor import PDFExtractor

def test_extraction_patterns(pdf_path):
    
    print("\n" + "="*60)
    print(f"TESTING: {pdf_path.split('/')[-1]}")
    print("="*60)
    
    extractor = PDFExtractor()
    doc = extractor.extract_pdf(pdf_path)
    
    if not doc:
        print("❌ Failed to extract PDF")
        return
    
    text = doc.full_text
    
    print(f"\n📄 Extracted {len(text)} characters")
    print(f"📄 Pages: {doc.page_count}")
    
    print("\n--- CURRENCY AMOUNTS FOUND ---")
    currency_patterns = [
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(?:Total|Amount|Balance|Due|Rent|Fee|Charge|Cost|Payment)[:\s]+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
    ]
    
    amounts_found = []
    for pattern in currency_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount_str = match.group(1) if match.lastindex else match.group(0)
                amount_str = amount_str.replace('$', '').replace(',', '').strip()
                amount = float(amount_str)
                
                context_start = max(0, match.start() - 60)
                context_end = min(len(text), match.end() + 60)
                context = text[context_start:context_end]
                
                amounts_found.append((amount, context))
            except:
                pass
    
    amounts_found = list(set(amounts_found))
    amounts_found.sort(key=lambda x: x[0], reverse=True)
    
    for amount, context in amounts_found[:15]:
        print(f"\n💰 ${amount:,.2f}")
        print(f"   Context: ...{context.strip()}...")
    
    print("\n--- DATES FOUND ---")
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
    ]
    
    dates_found = set()
    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            dates_found.add(match.group(0))
    
    for date in list(dates_found)[:10]:
        print(f"📅 {date}")
    
    print("\n--- KEY PHRASES ---")
    key_phrases = [
        'total', 'balance', 'due', 'amount', 'rent', 'approved',
        'benefit', 'claim', 'eviction', 'lease', 'move out', 'tenant'
    ]
    
    for phrase in key_phrases:
        count = text.lower().count(phrase)
        if count > 0:
            print(f"'{phrase}': found {count} times")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    
    pdfs = [
        'data/raw_pdfs/SDRP Move Out Calculation - 7721 Suzanne.pdf',
        'data/raw_pdfs/2013 Wishing Well move out calculation.pdf',
        'data/raw_pdfs/241 yoakum tenant closeout (1).pdf'
    ]
    
    for pdf in pdfs:
        test_extraction_patterns(pdf)