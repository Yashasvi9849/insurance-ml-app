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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    pdf_features = extract_features_from_pdf(tmp_path)
    claim_type_num = 1 if pdf_features["claim_type"] == "eviction" else 0

    combined_features = {
        "max_benefit": max_benefit,
        "termination_type": 1 if termination_type.lower() == "eviction" else 0,
        "avg_amount_in_pdf": pdf_features["avg_amount_in_pdf"],
        "damage_keyword_count": pdf_features["damage_keyword_count"],
        "refund_eligible": pdf_features["refund_eligible"],
        "claim_type": claim_type_num
    }

    predicted_benefit = predict_benefit(combined_features)

    return {
        "predicted_benefit": predicted_benefit,
        "claim_type": pdf_features["claim_type"],
        "refund_eligible": bool(pdf_features["refund_eligible"])
    }
