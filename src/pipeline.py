import os
import mlflow
import dagshub
from zenml import step, pipeline
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ایمپورت متغیرهای خودت
from config import *
from data_loader import load_clean_dataset, get_tokenized_dataset

# ==========================================
# تعریف گام‌های پایپ‌لاین (Steps)
# ==========================================

@step
def load_data_step() -> DatasetDict:
    """گام اول: لود کردن داده‌های تمیز شده از پوشه محلی (که توسط DVC مدیریت می‌شود)"""
    print("Loading data from DVC storage...")
    raw_datasets = load_clean_dataset(DATA_FOLDER)
    return raw_datasets

@step
def tokenize_data_step(raw_datasets: DatasetDict) -> DatasetDict:
    """گام دوم: توکنایز کردن داده‌ها"""
    print("Tokenizing data...")
    tokenized_datasets, _ = get_tokenized_dataset(raw_datasets)
    return tokenized_datasets

@step
def train_model_step(tokenized_datasets: DatasetDict):
    """گام سوم: آموزش مدل و لاگ کردن در DagsHub"""
    print("Initializing Model and LoRA...")
    
    # 1. تنظیمات اتصال به DagsHub (فاز 1)
    
    dagshub.init(repo_owner='amiresbati52', repo_name='NER-LoRA-MLOps', mlflow=True)
    
    mlflow.set_experiment("NER_LoRA_ZenML_Pipeline")

    # 2. آماده‌سازی مدل و توکنایزر
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, modules_to_save=["classifier"]
    )
    peft_model = get_peft_model(model, peft_config)
    
    # 3. تنظیمات آموزش (مناسب برای 1660 Ti فعلی شما)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=True,
        report_to="mlflow",           # اتصال جادویی به MLflow
        run_name="ZenML_Local_Run",
        fp16=True                     # استفاده بهینه از GPU
    )
    
    # تابع متریک‌ها (همان تابع خودت)
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=2)
        true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[l for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        flat_preds = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]
        return {
            "f1": f1_score(flat_labels, flat_preds, zero_division=0),
            "accuracy": accuracy_score(flat_labels, flat_preds),
        }

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting Training in ZenML Pipeline...")
    trainer.train()
    
    print("💾 Saving the best model...")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print("Step Finished!")

# ==========================================
# تعریف ارکستراسیون (Pipeline)
# ==========================================

@pipeline
def ner_training_pipeline():
    """اینجا مشخص می‌کنیم گام‌ها با چه ترتیبی اجرا شوند"""
    raw_data = load_data_step()
    tokenized_data = tokenize_data_step(raw_data)
    train_model_step(tokenized_data)

if __name__ == "__main__":
    # اجرای پایپ‌لاین
    ner_training_pipeline()