
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
import torch

# ساخت اپلیکیشن
app = FastAPI(
    title="Persian NER API",
    description="API for detecting Named Entities in Persian text using ParsBERT and LoRA",
    version="1.0.0"
)

# --- بارگذاری مدل از Hugging Face ---
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
PEFT_MODEL_ID = "amirhosseinesbati/parsbert-ner-lora-binary" # <--- نام کاربری خود را اینجا بنویسید

print("Loading Tokenizer and Base Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Loading LoRA Weights from Hugging Face Hub...")
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)

# انتقال مدل به GPU در صورت وجود
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # قرار دادن مدل در حالت تست
print("Model is ready to serve!")

# --- تعریف ساختار ورودی و خروجی API ---
class TextInput(BaseModel):
    text: str

class EntityOutput(BaseModel):
    word: str
    is_entity: int
    label: str

# --- ساخت Endpoint برای پیش‌بینی ---
@app.post("/predict", response_model=list[EntityOutput])
def predict_ner(input_data: TextInput):
    inputs = tokenizer(input_data.text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    predicted_labels = predictions[0].cpu().numpy()
    
    results = []
    for token, label_id in zip(tokens, predicted_labels):
        # حذف توکن‌های ویژه سیستم
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            results.append(
                EntityOutput(
                    word=token,
                    is_entity=int(label_id),
                    label="ENTITY" if label_id == 1 else "NON_ENTITY"
                )
            )
            
    return results

# ایجاد یک مسیر ساده برای تست سلامت API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Persian NER API. Go to /docs for Swagger UI."}