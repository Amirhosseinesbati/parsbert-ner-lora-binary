FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY src/app.py .

# راز موفقیت: نصب تمام کتابخانه‌ها در یک خط + معرفی منبع CPU برای پایتورچ 
RUN pip install --no-cache-dir torch>=2.6.0 fastapi uvicorn transformers peft pydantic mlflow dagshub --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]