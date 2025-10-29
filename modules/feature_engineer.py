import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dateutil import parser as date_parser
from dataclasses import dataclass, asdict
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ExtractionPatterns, FeatureConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaimFeatures:
    filename: str
    full_text: Optional[str] = None
    tracking_number: Optional[str] = None
    
    claim_date: Optional[str] = None
    approval_date: Optional[str] = None
    lease_start_date: Optional[str] = None
    lease_end_date: Optional[str] = None
    move_out_date: Optional[str] = None
    
    monthly_rent: Optional[float] = None
    amount_of_claim: Optional[float] = None
    max_benefit: Optional[float] = None
    approved_benefit_amount: Optional[float] = None
    
    lease_duration_days: Optional[int] = None
    days_to_moveout: Optional[int] = None
    days_claim_to_approval: Optional[int] = None
    
    late_fees_total: Optional[float] = None
    repair_costs_total: Optional[float] = None
    legal_fees_total: Optional[float] = None
    admin_fees_total: Optional[float] = None
    
    termination_type: Optional[str] = None
    lease_state: Optional[str] = None
    lease_city: Optional[str] = None
    tenant_contacted: Optional[str] = None
    collection_status: Optional[str] = None
    is_2nd_tenant: Optional[bool] = None
    is_3rd_tenant: Optional[bool] = None
    property_management_company: Optional[str] = None
    
    pm_explanation: Optional[str] = None
    hold_reason: Optional[str] = None
    
    extraction_confidence: float = 1.0
    missing_fields: List[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureEngineer:
    
    def __init__(self):
        self.patterns = ExtractionPatterns()
        self.feature_config = FeatureConfig()
        
    def extract_features(self, document_text: str, filename: str) -> ClaimFeatures:
        logger.info(f"Extracting features from: {filename}")
        
        features = ClaimFeatures(filename=filename, full_text=document_text)
        
        features = self._extract_dates(document_text, features)
        features = self._extract_amounts_improved(document_text, features)
        features = self._extract_address_info(document_text, features)
        features = self._extract_categorical(document_text, features)
        features = self._extract_text_features(document_text, features)
        features = self._extract_from_tables(document_text, features)
        
        features = self._calculate_derived_features(features)
        features.missing_fields = self._identify_missing_fields(features)
        
        return features
    
    def _extract_amounts_improved(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        
        total_balance_pattern = r'Total\s+Balance\s+Due:\s*\$?\s*([\d,]+\.?\d*)'
        match = re.search(total_balance_pattern, text, re.IGNORECASE)
        if match:
            try:
                amount_str = match.group(1).replace(',', '')
                features.approved_benefit_amount = float(amount_str)
                features.amount_of_claim = float(amount_str)
                logger.info(f"Found Total Balance Due: ${features.approved_benefit_amount:.2f}")
            except Exception as e:
                logger.debug(f"Failed to parse Total Balance Due: {e}")
        
        rent_pattern = r'(\d{2}/\d{2}/\d{4})\s+Rent\s+\$?\s*([\d,]+\.?\d*)'
        rent_matches = re.findall(rent_pattern, text, re.IGNORECASE)
        if rent_matches:
            rent_amounts = [float(m[1].replace(',', '')) for m in rent_matches]
            features.monthly_rent = max(rent_amounts)
            logger.info(f"Found Monthly Rent: ${features.monthly_rent:.2f}")
        
        late_fee_pattern = r'Late\s+Fee(?:\s*:\s*Late\s+Fee\s*\(Manual\))?\s+\$?\s*([\d,]+\.?\d*)'
        late_matches = re.findall(late_fee_pattern, text, re.IGNORECASE)
        if late_matches:
            features.late_fees_total = sum(float(m.replace(',', '')) for m in late_matches)
            logger.info(f"Found Late Fees Total: ${features.late_fees_total:.2f}")
        
        repair_pattern = r'(?:Move-out\s+)?Repair\s+(?:\d+\s+)?\$?\s*([\d,]+\.?\d*)'
        repair_matches = re.findall(repair_pattern, text, re.IGNORECASE)
        if repair_matches:
            features.repair_costs_total = sum(float(m.replace(',', '')) for m in repair_matches)
            logger.info(f"Found Repair Costs: ${features.repair_costs_total:.2f}")
        
        legal_pattern = r'Legal\s+and\s+Professional\s+\$?\s*([\d,]+\.?\d*)'
        legal_matches = re.findall(legal_pattern, text, re.IGNORECASE)
        if legal_matches:
            features.legal_fees_total = sum(float(m.replace(',', '')) for m in legal_matches)
            logger.info(f"Found Legal Fees: ${features.legal_fees_total:.2f}")
        
        admin_pattern = r'Administrative\s+Fee\s+\$?\s*([\d,]+\.?\d*)'
        admin_matches = re.findall(admin_pattern, text, re.IGNORECASE)
        if admin_matches:
            features.admin_fees_total = sum(float(m.replace(',', '')) for m in admin_matches)
            logger.info(f"Found Admin Fees: ${features.admin_fees_total:.2f}")
        
        return features
    
    def _extract_dates(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        
        header_date = re.search(r'^(\d{2}/\d{2}/\d{4})', text, re.MULTILINE)
        if header_date:
            try:
                parsed = date_parser.parse(header_date.group(1))
                features.claim_date = parsed.strftime('%Y-%m-%d')
                logger.info(f"Found Claim Date: {features.claim_date}")
            except:
                pass
        
        move_out_calc_match = re.search(r'Move\s+Out\s+Calculation', text, re.IGNORECASE)
        if move_out_calc_match and features.claim_date:
            features.move_out_date = features.claim_date
        
        date_desc_pattern = r'(\d{2}/\d{2}/\d{4})\s+([A-Za-z\s]+)'
        date_matches = re.findall(date_desc_pattern, text)
        
        if date_matches and not features.lease_start_date:
            first_date_str = date_matches[0][0]
            try:
                parsed = date_parser.parse(first_date_str)
                features.lease_start_date = parsed.strftime('%Y-%m-%d')
            except:
                pass
        
        return features
    
    def _extract_address_info(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        
        address_pattern = r'(\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Boulevard|Blvd))\s+([A-Za-z\s]+)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)'
        
        match = re.search(address_pattern, text, re.IGNORECASE)
        if match:
            features.lease_city = match.group(2).strip()
            features.lease_state = match.group(3).upper()
            logger.info(f"Found Address: {features.lease_city}, {features.lease_state}")
        
        return features
    
    def _extract_categorical(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        text_lower = text.lower()
        
        if 'move out' in text_lower or 'move-out' in text_lower:
            features.termination_type = 'move_out'
        elif 'eviction' in text_lower:
            features.termination_type = 'eviction'
        elif 'lease break' in text_lower:
            features.termination_type = 'lease_break'
        
        if 'legal' in text_lower or 'attorney' in text_lower:
            features.collection_status = 'Legal Action'
        else:
            features.collection_status = 'Pending'
        
        features.tenant_contacted = 'Yes'
        
        if ' and all other residents' in text_lower or 'and ' in text_lower:
            features.is_2nd_tenant = True
        else:
            features.is_2nd_tenant = False
        
        features.is_3rd_tenant = False
        
        company_match = re.search(r'^([A-Z][A-Za-z\s&]+Property\s+Management)', text, re.MULTILINE)
        if company_match:
            features.property_management_company = company_match.group(1).strip()
            logger.info(f"Found Company: {features.property_management_company}")
        
        return features
    
    def _extract_text_features(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        return features
    
    def _extract_from_tables(self, text: str, features: ClaimFeatures) -> ClaimFeatures:
        return features
    
    def _calculate_derived_features(self, features: ClaimFeatures) -> ClaimFeatures:
        try:
            if features.lease_start_date and features.move_out_date:
                start = datetime.strptime(features.lease_start_date, '%Y-%m-%d')
                moveout = datetime.strptime(features.move_out_date, '%Y-%m-%d')
                features.days_to_moveout = (moveout - start).days
            
            if features.claim_date and features.approval_date:
                claim = datetime.strptime(features.claim_date, '%Y-%m-%d')
                approval = datetime.strptime(features.approval_date, '%Y-%m-%d')
                features.days_claim_to_approval = (approval - claim).days
        
        except Exception as e:
            logger.debug(f"Error calculating derived features: {e}")
        
        return features
    
    def _identify_missing_fields(self, features: ClaimFeatures) -> List[str]:
        missing = []
        
        important_fields = [
            'monthly_rent', 'amount_of_claim', 'approved_benefit_amount',
            'termination_type', 'claim_date', 'lease_state', 'property_management_company'
        ]
        
        for field in important_fields:
            if getattr(features, field) is None:
                missing.append(field)
        
        return missing
    
    def batch_extract_features(self, documents: List[Dict], output_file: str = None) -> List[ClaimFeatures]:
        logger.info(f"Extracting features from {len(documents)} documents")
        logger.info("=" * 60)
        
        all_features = []
        
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get('full_text', '')
                filename = doc.get('filename', 'unknown')
            else:
                text = doc.full_text
                filename = doc.filename
            
            features = self.extract_features(text, filename)
            all_features.append(features)
            
            if len(features.missing_fields) > 3:
                logger.warning(f"{filename}: Missing {len(features.missing_fields)} fields")
            else:
                logger.info(f"{filename}: Extracted successfully")
        
        if output_file:
            self._save_features(all_features, output_file)
        
        logger.info("=" * 60)
        logger.info(f"Feature extraction complete: {len(all_features)} documents")
        
        return all_features
    
    def _save_features(self, features: List[ClaimFeatures], output_file: str):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features_dict = [f.to_dict() for f in features]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(features_dict, f, indent=2, default=str)
        
        logger.info(f"Features saved to: {output_path}")


if __name__ == "__main__":
    from pdf_extractor import PDFExtractor
    
    extractor = PDFExtractor()
    documents = extractor.batch_extract(
        pdf_directory="data/raw_pdfs",
        output_directory="data/processed/extracted_text"
    )
    
    engineer = FeatureEngineer()
    features = engineer.batch_extract_features(
        documents=documents,
        output_file="data/processed/extracted_features.json"
    )
    
    if features:
        print("\n" + "=" * 60)
        print("SAMPLE FEATURES (First Document)")
        print("=" * 60)
        sample = features[0]
        print(f"Filename: {sample.filename}")
        print(f"Monthly Rent: ${sample.monthly_rent}")
        print(f"Amount of Claim: ${sample.amount_of_claim}")
        print(f"Approved Amount: ${sample.approved_benefit_amount}")
        print(f"Late Fees: ${sample.late_fees_total}")
        print(f"Repair Costs: ${sample.repair_costs_total}")
        print(f"Termination Type: {sample.termination_type}")
        print(f"State: {sample.lease_state}")
        print(f"Missing Fields: {sample.missing_fields}")
        print("=" * 60)