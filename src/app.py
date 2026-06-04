import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
import torch
import mlflow
from mlflow.tracking import MlflowClient

BASE_MODEL = "HooshvareLab/bert-fa-base-uncased"
ID2LABEL = {0: "O", 1: "ENTITY"}
REGISTERED_MODEL_NAME = "Persian-NER-LoRA" 

tokenizer = None
model = None

def load_production_model():
    global tokenizer, model
    print("🌍 Connecting to DagsHub Model Registry...")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    if not dagshub_token:
        raise ValueError("DAGSHUB_TOKEN is not set!")
        
    os.environ["MLFLOW_TRACKING_USERNAME"] = "amiresbati52"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri("https://dagshub.com/amiresbati52/NER-LoRA-MLOps.mlflow")
    
    try:
        client = MlflowClient()
        print(f"🔍 Searching for '{REGISTERED_MODEL_NAME}' in Production...")
        
        # روش مدرن و بدون هشدار زرد رنگ برای گرفتن نسخه پروداکشن
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        
        if not prod_versions:
            print("⚠️ No Production stage found. Defaulting to version 1...")
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/1"
        else:
            version_num = prod_versions[0].version
            print(f"✅ Found Version: {version_num} | Downloading...")
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version_num}"

        # 1. دانلود در پوشه لوکال
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
        
        # 2. رادار جستجو: پیدا کردن مسیر دقیق فایل adapter_config.json
        adapter_dir = local_dir
        for root, dirs, files in os.walk(local_dir):
            if "adapter_config.json" in files:
                adapter_dir = root
                break
                
        print(f"🎯 Adapter files located at: {adapter_dir}")
        
        # 3. لود کردن مدل‌ها با مسیر دقیق
        new_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model_hf = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, num_labels=2)
        
        new_model = PeftModel.from_pretrained(base_model_hf, adapter_dir)
        new_model.eval() 
        
        tokenizer = new_tokenizer
        model = new_model
        print("🚀 Production Model is LIVE and ready to serve!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_production_model()
    yield
    print("🛑 Shutting down and clearing memory...")
    global model, tokenizer
    del model, tokenizer

app = FastAPI(title="Persian NER Production API", version="2.1", lifespan=lifespan)

class InferenceRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_ner(request: InferenceRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is loading...")
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    results = []
    for token, pred_id in zip(tokens, predictions):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            results.append({"word": token.replace("##", ""), "entity": ID2LABEL.get(pred_id, "O")})
    return {"text": request.text, "entities": results}

@app.post("/refresh-model")
def refresh_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(load_production_model)
    return {"status": "success", "message": "Downloading new model in background..."}