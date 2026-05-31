#!/bin/bash
echo "🚀 Starting MLOps Cloud Pipeline..."

# 1. نصب ابزارهای مورد نیاز
pip install -r requirements.txt
pip install dvc dvc-s3 zenml vastai

# 2. تنظیم دسترسی DVC به DagsHub (بدون نیاز به پسورد دستی)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user amiresbati52
dvc remote modify origin --local password b7bbea51b36793e3cb81403c0e5e277969cd23ba

# 3. دانلود داده‌ها از کلود
echo "📦 Pulling data from DVC..."
dvc pull

# 4. اجرای پایپ‌لاین آموزش
echo "🧠 Running ZenML Training Pipeline..."
python src/pipeline.py

echo "✅ Training Completed Successfully!"

# (اختیاری) 5. خاموش کردن خودکار سرور پس از اتمام کار برای جلوگیری از هزینه اضافی!
# اگر می‌خواهی بعد از آموزش سرور نابود شود، علامت # خط زیر را بردار و آیدی سرور را ست کن
# vastai destroy instance $VAST_CONTAINERLABEL