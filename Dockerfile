# استفاده از یک نسخه سبک پایتون
FROM python:3.10-slim

# تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

# کپی کردن فایل نیازمندی‌ها و نصب آن‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن سورس کدهای پروژه
COPY src/ /app/src/

# باز کردن پورت 8000 برای ارتباط شبکه
EXPOSE 8000

# دستوری که هنگام اجرای کانتینر ران می‌شود
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]