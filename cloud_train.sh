#!/bin/bash
set -e
echo "🚀 Starting MLOps Cloud Pipeline..."

# --- 2. تابع خود-تخریبی برای جلوگیری از هزینه اضافی ---
cleanup() {
    echo "🛑 Job finished or failed. Destroying this Vast.ai instance..."
    pip install vastai
    
    # استخراج فقط اعداد از نام کانتینر
    INSTANCE_ID=${VAST_CONTAINERLABEL//[!0-9]/}
    
    echo "💣 Target Instance ID to destroy: $INSTANCE_ID"
    
    # نابود کردن ماشین با آیدی خالص + پرچم -y برای تایید خودکار (بدون پرسش)
    vastai destroy instance $INSTANCE_ID -y --api-key $VAST_API_KEY
}
trap cleanup EXIT

# ۱. آپدیت پایتون و پیپ
python -m pip install --upgrade pip 
python -m pip install vastai

۲. نصب مستقیم، به‌روز و بدون دردسر پکیج‌های اصلی (بدون requirements.txt)
استفاده از --upgrade باعث می‌شود جدیدترین نسخه transformers نصب شود و باگ PyTorch 2.4 حل شود
خط نصب را در فایل cloud_train.sh به این خط تغییر دهید:
python -m pip install mlflow dagshub "zenml[server]" "dvc[s3]" datasets "transformers==4.44.2" "accelerate==0.34.2" "peft==0.12.0" huggingface_hub scikit-learn evaluate seqeval sqlalchemy-utils
# ۳. تنظیمات DVC با ترفند python -m (برای دور زدن خطای command not found)
echo "⚙️ Setting up DVC..."
python -m dvc remote remove origin 2>/dev/null || true
python -m dvc remote add -d origin s3://dvc
python -m dvc remote modify origin endpointurl https://dagshub.com/amiresbati52/NER-LoRA-MLOps.s3

export AWS_ACCESS_KEY_ID=$DAGSHUB_TOKEN
export AWS_SECRET_ACCESS_KEY=$DAGSHUB_TOKEN


# (برای اینکه در مرحله بعد MLflow هم بتواند لاگ‌ها را به داگزهاب بفرستد، این دو متغیر را هم همینجا اضافه کنید عالی می‌شود)
export MLFLOW_TRACKING_USERNAME="amiresbati52"
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN

# این دو خط را به بخش export های فایل cloud_train.sh اضافه کنید:
export DAGSHUB_USER_TOKEN=$DAGSHUB_TOKEN
export MLFLOW_TRACKING_URI="https://dagshub.com/amiresbati52/NER-LoRA-MLOps.mlflow"

echo "📥 Pulling data from DVC..."
python -m dvc pull -r origin

echo "🧠 Running ZenML Training Pipeline..."
# ۴. اجرای خط لوله
python src/pipeline.py

echo "✅ Training Completed Successfully!"