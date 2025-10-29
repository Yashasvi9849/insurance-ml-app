import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).parent))

from modules.pdf_extractor import PDFExtractor
from modules.feature_engineer import FeatureEngineer
from config import MODELS_DIR


def predict_single_pdf(pdf_path: str):
    
    print("\n" + "="*60)
    print("CLAIM AMOUNT PREDICTION - NEW PDF")
    print("="*60 + "\n")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        return
    
    print(f"📄 Processing: {pdf_path.name}\n")
    
    print("STEP 1: Extracting text from PDF...")
    print("-" * 60)
    
    extractor = PDFExtractor()
    document = extractor.extract_pdf(str(pdf_path))
    
    if not document:
        print("❌ Error: Failed to extract text from PDF")
        return
    
    print(f"✅ Extracted {len(document.full_text)} characters")
    print(f"   Method: {document.extraction_method}\n")
    
    print("STEP 2: Extracting features...")
    print("-" * 60)
    
    engineer = FeatureEngineer()
    features = engineer.extract_features(document.full_text, pdf_path.name)
    
    print(f"✅ Extracted features:")
    print(f"   Monthly Rent: ${features.monthly_rent}" if features.monthly_rent else "   Monthly Rent: Not found")
    print(f"   Amount of Claim: ${features.amount_of_claim}" if features.amount_of_claim else "   Amount of Claim: Not found")
    print(f"   Late Fees: ${features.late_fees_total}" if features.late_fees_total else "   Late Fees: Not found")
    print(f"   Repair Costs: ${features.repair_costs_total}" if features.repair_costs_total else "   Repair Costs: Not found")
    print(f"   State: {features.lease_state}" if features.lease_state else "   State: Not found")
    print(f"   Termination Type: {features.termination_type}" if features.termination_type else "   Termination Type: Not found")
    
    if len(features.missing_fields) > 5:
        print(f"\n⚠️  Warning: {len(features.missing_fields)} fields missing")
        print("   Prediction may be less accurate")
    
    print("\nSTEP 3: Loading trained model...")
    print("-" * 60)
    
    model_path = MODELS_DIR / "simple_model.pkl"
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train a model first: python train_simple.py")
        return
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    expected_features = model_data['features']
    
    print(f"✅ Model loaded successfully\n")
    
    print("STEP 4: Preparing features for prediction...")
    print("-" * 60)
    
    features_dict = features.to_dict()
    df = pd.DataFrame([features_dict])
    
    exclude_cols = [
        'filename', 'tracking_number', 'pm_explanation', 
        'hold_reason', 'missing_fields', 'extraction_confidence',
        'claim_date', 'approval_date', 'lease_start_date', 
        'lease_end_date', 'move_out_date', 'approved_benefit_amount'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[expected_features]
    
    X = X.fillna(X.median())
    X = X.fillna(0)
    
    X_scaled = scaler.transform(X)
    
    print(f"✅ Features prepared\n")
    
    print("STEP 5: Making prediction...")
    print("-" * 60)
    
    prediction = model.predict(X_scaled)[0]
    
    print("\n" + "="*60)
    print("🎯 PREDICTION RESULT")
    print("="*60)
    print(f"\n💰 Predicted Approved Benefit Amount: ${prediction:,.2f}\n")
    
    if features.amount_of_claim:
        diff = prediction - features.amount_of_claim
        diff_pct = (diff / features.amount_of_claim) * 100
        print(f"📊 Comparison:")
        print(f"   Amount of Claim:  ${features.amount_of_claim:,.2f}")
        print(f"   Predicted Amount: ${prediction:,.2f}")
        print(f"   Difference:       ${diff:,.2f} ({diff_pct:+.1f}%)")
    
    confidence_level = "Low"
    if len(features.missing_fields) < 3:
        confidence_level = "High"
    elif len(features.missing_fields) < 5:
        confidence_level = "Medium"
    
    print(f"\n📈 Confidence Level: {confidence_level}")
    print(f"   (Based on {len(features.missing_fields)} missing fields)")
    
    print("\n" + "="*60)
    print("⚠️  IMPORTANT NOTES")
    print("="*60)
    print("• This model was trained on only 3 samples")
    print("• Predictions should be treated as ESTIMATES only")
    print("• For production use, train with 20-50+ samples")
    print("• Always verify predictions with actual claim data")
    
    print("\n" + "="*60)
    print("📝 EXTRACTED DATA SUMMARY")
    print("="*60)
    
    print(f"\nDocument: {pdf_path.name}")
    print(f"Property: {features.lease_city}, {features.lease_state}" if features.lease_city else "Property: Unknown")
    print(f"Company: {features.property_management_company}" if features.property_management_company else "Company: Unknown")
    
    if features.late_fees_total or features.repair_costs_total or features.legal_fees_total:
        print(f"\nItemized Costs:")
        if features.late_fees_total:
            print(f"  Late Fees:    ${features.late_fees_total:,.2f}")
        if features.repair_costs_total:
            print(f"  Repairs:      ${features.repair_costs_total:,.2f}")
        if features.legal_fees_total:
            print(f"  Legal Fees:   ${features.legal_fees_total:,.2f}")
        if features.admin_fees_total:
            print(f"  Admin Fees:   ${features.admin_fees_total:,.2f}")
    
    print("\n" + "="*60 + "\n")
    
    save_option = input("💾 Save prediction to CSV? (y/n): ").strip().lower()
    if save_option == 'y':
        output_path = Path("data/processed/predictions.csv")
        
        prediction_df = pd.DataFrame([{
            'filename': pdf_path.name,
            'predicted_amount': prediction,
            'actual_amount_of_claim': features.amount_of_claim,
            'confidence': confidence_level,
            'timestamp': pd.Timestamp.now().isoformat()
        }])
        
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            prediction_df = pd.concat([existing_df, prediction_df], ignore_index=True)
        
        prediction_df.to_csv(output_path, index=False)
        print(f"\n✅ Prediction saved to: {output_path}\n")


def main():
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("CLAIM AMOUNT PREDICTOR")
        print("="*60)
        print("\nUsage:")
        print("  python predict_new.py <path_to_pdf>")
        print("\nExamples:")
        print("  python predict_new.py data/raw_pdfs/claim_001.pdf")
        print("  python predict_new.py ~/Downloads/new_claim.pdf")
        print("\n" + "="*60 + "\n")
        return
    
    pdf_path = sys.argv[1]
    predict_single_pdf(pdf_path)


if __name__ == "__main__":
    main()