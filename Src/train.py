import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
import mlflow.pytorch
import os
import dagshub

# وارد کردن متغیرها و توابع از فایل‌های خودمان
from config import *
from data_loader import load_clean_dataset, get_tokenized_dataset

# اتصال به سرور MLflow داگزهاب

dagshub.init(repo_owner='amiresbati52', repo_name='NER-LoRA-MLOps', mlflow=True)
mlflow.set_experiment("ParsBERT_NER_Experiment")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)
    
    true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[l for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    return {
        "precision": precision_score(flat_labels, flat_preds, zero_division=0),
        "recall": recall_score(flat_labels, flat_preds, zero_division=0),
        "f1": f1_score(flat_labels, flat_preds, zero_division=0),
        "accuracy": accuracy_score(flat_labels, flat_preds),
    }

def main():
    print("Loading data...")
    raw_datasets = load_clean_dataset(DATA_FOLDER)
    tokenized_datasets, tokenizer = get_tokenized_dataset(raw_datasets)
    
    print("Initializing Model and LoRA...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)
    
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, modules_to_save=["classifier"]
    )
    peft_model = get_peft_model(model, peft_config)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to="mlflow",
        run_name="Vast_Test"

    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("🚀 Starting Training...")


    # ثبت پارامترها به صورت اتوماتیک
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("lora_r", LORA_R)



    trainer.train()


    # ثبت متریک‌ها
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # ذخیره مدل در MLflow (بدون نیاز به ذخیره دستی در پوشه!)
    mlflow.pytorch.log_model(peft_model, "best_model")
    
    print("💾 Saving the best model...")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print("Training Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()