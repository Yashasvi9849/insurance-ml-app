# verify_model.py
import joblib

model_path = "models/simple_model.pkl"
obj = joblib.load(model_path)

print(f"✅ Loaded type: {type(obj)}")

if isinstance(obj, dict):
    print(f"Keys in pickle: {list(obj.keys())}")

    for key, val in obj.items():
        print(f"\n🔹 {key}: {type(val)}")
