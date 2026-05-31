#!/bin/bash
echo "🚀 Starting MLOps Cloud Pipeline..."

# 1. نصب ابزارهای مورد نیاز
pip install -r requirements.txt
pip install dvc dvc-s3 zenml vastai

# 2. تنظیم دسترسی DVC به DagsHub (بدون نیاز به پسورد دستی)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user amiresbati52
dvc remote modify origin --local password 9755f1691ad9f4e384f6131973699590f6dc63b0

# 3. دانلود داده‌ها از کلود
echo "📦 Pulling data from DVC..."
dvc pull

# 4. اجرای پایپ‌لاین آموزش
echo "🧠 Running ZenML Training Pipeline..."
python Src/pipeline.py

echo "✅ Training Completed Successfully!"

# (اختیاری) 5. خاموش کردن خودکار سرور پس از اتمام کار برای جلوگیری از هزینه اضافی!
# اگر می‌خواهی بعد از آموزش سرور نابود شود، علامت # خط زیر را بردار و آیدی سرور را ست کن
# vastai destroy instance $VAST_CONTAINERLABEL