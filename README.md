# 🇮🇷 Persian Named Entity Recognition (NER) with LoRA & MLOps

![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)
![Model](https://img.shields.io/badge/Model-ParsBERT-orange.svg)
![PEFT](https://img.shields.io/badge/Optimization-LoRA-red.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Registry-yellow)](https://huggingface.co/[YOUR_USERNAME]/parsbert-ner-lora-binary)

یک پروژه کامل و استاندارد برای تشخیص موجودیت‌های نامدار (Named Entity Recognition) در متون زبان فارسی. این مدل به صورت باینری (موجودیت یا غیرموجودیت) عمل کرده و با استفاده از معماری **ParsBERT** و روش بهینه‌سازی پارامترهای کارآمد **LoRA (PEFT)** آموزش دیده است.

پروژه پیش رو صرفاً یک مدل تحقیقاتی نیست، بلکه بر اساس استانداردهای **MLOps** ساختاردهی شده و آماده استقرار (Deployment) در محیط‌های Production از طریق **FastAPI** و **Docker** می‌باشد.

---

## 📑 فهرست مطالب

- [معماری سیستم](#-معماری-سیستم)
- [عملکرد و نتایج ارزیابی](#-عملکرد-و-نتایج-ارزیابی)
- [ساختار پروژه](#-ساختار-پروژه)
- [اجرای سریع (Quick Start)](#-اجرای-سریع-quick-start)
- [استفاده از API](#-استفاده-از-api)
- [آموزش مجدد مدل](#-آموزش-مجدد-مدل)

---

## 🧠 معماری سیستم

1. **مدل پایه:** `HooshvareLab/bert-fa-base-uncased` (ParsBERT)
2. **تکنیک فاین‌تیونینگ:** روش LoRA با کاهش چشمگیر پارامترهای آموزش‌پذیر (Rank=16, Alpha=32)
3. **Model Registry:** ذخیره و فراخوانی مستقیم وزن‌ها از Hugging Face Hub
4. **سرویس‌دهی (Serving):** پیاده‌سازی RESTful API با استفاده از FastAPI
5. **کانتینرسازی:** آماده‌سازی ایمیج سبک با Docker برای اجرای مستقل از پلتفرم

---

## 📊 عملکرد و نتایج ارزیابی

مدل به مدت ۱۰ دوره (Epoch) روی دیتاست استاندارد آموزش دیده است. با وجود استفاده از LoRA و فریز بودن وزن‌های اصلی، مدل به نتایج خیره‌کننده‌ای دست یافته است:

| Metric | Score (Test Data) |
|--------|-------------------|
| **Precision** | 94.2% |
| **Recall** | 95.4% |
| **F1-Score** | **94.8%** |
| **Accuracy** | 99.7% |

*نکته: دلیل عدم بروز پدیده Overfitting، استفاده از تنظیمات بهینه `weight_decay=0.01` و `lora_dropout=0.1` بوده است.*

---

## 📂 ساختار پروژه

```text
Persian-Binary-NER/
│
├── src/
│   ├── app.py             # کدهای سرویس‌دهی FastAPI
│   ├── config.py          # تنظیمات متمرکز سیستم (هایپرپارامترها)
│   ├── data_loader.py     # پیش‌پردازش دیتاست و Token Alignment
│   └── train.py           # پایپ‌لاین آموزش با Trainer و LoRA
│
├── data/                  # (در گیتهاب نادیده گرفته شده) محل فایل‌های txt
├── requirements.txt       # نیازمندی‌های پایتون
├── Dockerfile             # اسکریپت ساخت کانتینر
└── README.md              # مستندات پروژه
```

---

## 🚀 اجرای سریع (Quick Start)

### روش اول: با استفاده از Docker (پیشنهادی)

سریع‌ترین راه برای اجرای API، استفاده از داکر است. بدون نیاز به نصب پایتون و کتابخانه‌ها:

```bash
git clone https://github.com/[YOUR_USERNAME]/Persian-Binary-NER.git
cd Persian-Binary-NER

# ساخت ایمیج
docker build -t persian-ner-api .

# اجرای کانتینر در پورت 8000
docker run -p 8000:8000 persian-ner-api
```

### روش دوم: اجرای محلی با پایتون

```bash
git clone https://github.com/[YOUR_USERNAME]/Persian-Binary-NER.git
cd Persian-Binary-NER

pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

پس از اجرا، برای تست رابط گرافیکی API به آدرس زیر مراجعه کنید:

```text
http://localhost:8000/docs
```

---

## 🌐 استفاده از API

هنگامی که سرور در حال اجراست، می‌توانید متون خود را در قالب JSON به آن ارسال کنید:

### درخواست (CURL)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "پروفسور مجید سمیعی در شهر رشت متولد شد."
}'
```

### پاسخ سامانه (Response)

```json
[
  { "word": "پروفسور", "is_entity": 0, "label": "NON_ENTITY" },
  { "word": "مجید", "is_entity": 1, "label": "ENTITY" },
  { "word": "سمیعی", "is_entity": 1, "label": "ENTITY" },
  { "word": "در", "is_entity": 0, "label": "NON_ENTITY" },
  { "word": "شهر", "is_entity": 0, "label": "NON_ENTITY" },
  { "word": "رشت", "is_entity": 1, "label": "ENTITY" },
  ...
]
```

---

## 🛠 آموزش مجدد مدل (Training Pipeline)

اگر مایلید مدل را با داده‌های خودتان آموزش دهید:

1. فایل‌های متنی (txt) با فرمت استاندارد (کلمه و برچسب 0/1) را در پوشه `data/` قرار دهید.
2. دستور زیر را اجرا کنید:

```bash
python src/train.py
```

مدل به صورت خودکار پارامترهای `config.py` را خوانده و بهترین وزن‌ها را در پایان ذخیره می‌کند.

---

توسعه داده شده با ❤️ برای ارتقاء پردازش زبان طبیعی فارسی