# This is created for test pipeline 1
import pdfplumber
import re
import numpy as np

def extract_features_from_pdf(pdf_path: str) -> dict:
    """Extract structured info from claim PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.lower() + "\n"

    # Extract dollar amounts
    amounts = [float(a.replace(",", "")) for a in re.findall(r"\$?\s*([0-9]{2,6}(?:\.[0-9]{2})?)", text)]
    avg_amount = np.mean(amounts) if amounts else 0

    # Detect claim type keywords
    if "move out" in text or "lease end" in text:
        claim_type = "move_out"
    elif "evict" in text or "termination" in text:
        claim_type = "eviction"
    else:
        claim_type = "unknown"

    # Count damage-related words
    damage_keywords = ["damage", "repair", "replace", "clean", "fix", "broken", "stain"]
    damage_count = sum(text.count(word) for word in damage_keywords)

    # Heuristic refund eligibility (optional rule-based logic)
    refund_eligible = 1 if "security deposit" in text or avg_amount < 2000 else 0

    return {
        "avg_amount_in_pdf": avg_amount,
        "damage_keyword_count": damage_count,
        "claim_type": claim_type,
        "refund_eligible": refund_eligible
    }
