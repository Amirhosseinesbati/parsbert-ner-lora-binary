
# تنظیمات کلی پروژه
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
DATA_FOLDER = "./data" # مسیر پوشه‌ای که فایل‌های txt در آن هستند
OUTPUT_DIR = "./saved_model"

# هایپرپارامترهای مدل و آموزش
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 10
WEIGHT_DECAY = 0.01

# تنظیمات LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query", "value"]

# تنظیمات برچسب‌ها
ID2LABEL = {0: "O", 1: "B-ENTITY"}
LABEL2ID = {"O": 0, "B-ENTITY": 1}