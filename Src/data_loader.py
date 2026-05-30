import os
import glob
import subprocess
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN

def download_and_extract_data(target_folder):
    """
    بررسی می‌کند که آیا دیتا موجود است یا خیر. اگر نبود، آن را دانلود و استخراج می‌کند.
    """
    # بررسی وجود فایل‌های متنی در پوشه هدف
    if os.path.exists(target_folder) and len(glob.glob(os.path.join(target_folder, '*.txt'))) > 0:
        print(f"✅ داده‌ها از قبل در پوشه '{target_folder}' موجود هستند. (بدون نیاز به دانلود مجدد)")
        return

    print("📥 در حال دانلود دیتاست از Dropbox (لطفاً صبور باشید)...")
    os.makedirs(target_folder, exist_ok=True)
    
    # لینک مستقیم دانلود (dl=1)
    url = "https://www.dropbox.com/scl/fi/ln64xjqlwd4665u9sp2bq/Full_Dataset.rar?rlkey=g5aa4zefy2xpj98n3x8ihjcp2&st=bg907w5p&dl=1"
    rar_path = "Full_Dataset.rar"
    
    try:
        # دانلود فایل
        subprocess.run(f"wget -O {rar_path} '{url}'", shell=True, check=True)
        
        print("📦 در حال استخراج فایل‌های دیتاست...")
        # استخراج فایل‌ها مستقیماً داخل پوشه هدف (بدون ساخت پوشه‌های تو در تو)
        subprocess.run(f"unrar e {rar_path} {target_folder}/", shell=True, check=True)
        
        # پاک کردن فایل فشرده برای خلوت شدن فضای سرور
        if os.path.exists(rar_path):
            os.remove(rar_path)
            
        print("🎉 دانلود و استخراج دیتاست با موفقیت انجام شد!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ خطایی در فرآیند دانلود یا استخراج رخ داد: {e}")
        print("❗ راهنمایی: احتمالاً پکیج unrar روی سرور نصب نیست.")
        print("لطفاً این دستور را در ترمینال اجرا کنید: apt-get update && apt-get install unrar -y")
        exit(1)

def load_clean_dataset(folder_path):
    # --- ابتدا مطمئن می‌شویم دیتا دانلود شده است ---
    download_and_extract_data(folder_path)
    # -----------------------------------------------

    all_tokens, all_ner_tags = [], []
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if len(file_list) == 0:
        raise ValueError(f"هیچ فایل .txt در مسیر {folder_path} پیدا نشد!")
        
    for file_path in file_list:
        if "glove" in file_path.lower(): continue
        tokens, tags = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(int(parts[-1]))
        if tokens:
            all_tokens.append(tokens)
            all_ner_tags.append(tags)
            
    dataset = Dataset.from_dict({'tokens': all_tokens, 'ner_tags': all_ner_tags})
    print(f"📊 دیتاست با {len(dataset)} جمله با موفقیت ساخته شد.")
    
    # تقسیم‌بندی
    train_testvalid = dataset.train_test_split(test_size=0.3, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    return DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

def get_tokenized_dataset(dataset_dict):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_and_align(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=MAX_LEN)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None: label_ids.append(-100)
                elif word_idx != previous_word_idx: label_ids.append(label[word_idx])
                else: label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dataset_dict.map(tokenize_and_align, batched=True), tokenizer