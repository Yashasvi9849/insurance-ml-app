from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from modules.pdf_parser import extract_features_from_pdf
from modules.predict_logic import predict_benefit

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_claim(
    file: UploadFile = File(...),
    max_benefit: float = Form(...),
    termination_type: str = Form(...),
    additional_reqs: str = Form("")
):
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Extract features safely
    pdf_features = extract_features_from_pdf(tmp_path) or {}
    claim_type = pdf_features.get("claim_type", "move_out")
    claim_type_num = 1 if claim_type.lower() == "eviction" else 0

    # Combine all features
    combined_features = {
        "max_benefit": max_benefit,
        "termination_type": 1 if termination_type.lower() == "eviction" else 0,
        "avg_amount_in_pdf": pdf_features.get("avg_amount_in_pdf", 0),
        "damage_keyword_count": pdf_features.get("damage_keyword_count", 0),
        "refund_eligible": int(pdf_features.get("refund_eligible", 0)),
        "claim_type": claim_type_num
    }

    # Make prediction safely
    try:
        predicted_benefit = predict_benefit(combined_features)
    except Exception as e:
        import traceback
        print("Prediction failed:", traceback.format_exc())
        return {"error": f"Prediction failed: {str(e)}"}

    # Return clean JSON response
    # Return clean JSON response
    return {
    "status": "success",
    "predicted_benefit": float(predicted_benefit),
    "claim_type": claim_type,
    "refund_eligible": bool(pdf_features.get("refund_eligible", 0))
}

