import os
import glob
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN

def load_clean_dataset(folder_path):
    all_tokens, all_ner_tags = [], []
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
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