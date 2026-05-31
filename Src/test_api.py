import requests

# اگر از روش اول (SSH) استفاده کردید:
API_URL = "http://localhost:8000/predict"

# اگر از روش دوم (Ngrok) استفاده کردید، لینک ngrok را بگذارید:
# API_URL = "https://YOUR_NGROK_URL.ngrok-free.app/predict"

data = {
    "text": "  علی به همراه حسن در تهران به دانشگاه شریف رفتند ولی محمد همراه آنها نرفت !"
}

print("ارسال درخواست به سرور GPU...")
response = requests.post(API_URL, json=data)

if response.status_code == 200:
    results = response.json()
    for item in results:
        # چاپ کلماتی که به عنوان موجودیت (Entity) شناخته شده‌اند
        if item["is_entity"] == 1:
            print(f"کلمه: {item['word']}  --> تشخیص: {item['label']}")
else:
    print(f"خطا: {response.status_code}")
    print(response.text)