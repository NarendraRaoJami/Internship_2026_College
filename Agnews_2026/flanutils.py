import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# Memory clearing
# ==========================================
def clear_gpu_memory():
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# Config
# ==========================================
class BaseConfig:
    SEED = 42
    MAX_LENGTH = 256
    BATCH_SIZE = 4
    LR = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(BaseConfig.SEED)
np.random.seed(BaseConfig.SEED)

# ==========================================
# Load AG NEWS Dataset
# ==========================================
def load_data(train_path="agnewstrain.csv", test_path="agnewstest.csv"):
    if not os.path.exists(train_path):
        print("⚠️ Creating dummy AG NEWS datasets (files missing)")
        def make_dummy(size):
            df = pd.DataFrame({
                'label': np.random.choice([1, 2, 3, 4], size),
                'title': ["Dummy Title"] * size,
                'description': ["This is a dummy description of a news article."] * size
            })
            df['text'] = df['title'] + " - " + df['description']
            return df
        return make_dummy(1000), make_dummy(200)
    
    train_df = pd.read_csv(train_path, header=None, names=['label', 'title', 'description'])
    test_df = pd.read_csv(test_path, header=None, names=['label', 'title', 'description'])
    
    train_df['text'] = train_df['title'].fillna('') + " - " + train_df['description'].fillna('')
    test_df['text'] = test_df['title'].fillna('') + " - " + test_df['description'].fillna('')
    
    return train_df, test_df

# ==========================================
# Prepare Dataset
# ==========================================
# def prepare_dataset_agnews(train_df, test_df):
#     label_mapping = {1: "world", 2: "sports", 3: "business", 4: "sci/tech"}
    
#     def format_dataframe(df):
#         data = []
#         for _, row in df.iterrows():
#             text_snippet = str(row['text'])[:300] 
#             prompt = (f"Classify the following news article into one of four categories: "
#                       f"world, sports, business, or sci/tech. "
#                       f"Article: '{text_snippet}'. Predict category:")
#             data.append({
#                 "input_text": prompt,
#                 "target_text": label_mapping.get(row['label'], "world"),
#                 "type": "ag_news"
#             })
#         return pd.DataFrame(data)

#     formatted_train = format_dataframe(train_df)
#     formatted_test = format_dataframe(test_df)

#     # 10% validation split
#     train_final, val_final = train_test_split(
#         formatted_train, test_size=0.10, stratify=formatted_train['target_text'], random_state=BaseConfig.SEED
#     )
    
#     return train_final, val_final, formatted_test

def prepare_dataset_agnews(train_df, test_df):
    # Support both int and string keys just in case
    label_mapping = {
        1: "world", 2: "sports", 3: "business", 4: "sci/tech",
        "1": "world", "2": "sports", "3": "business", "4": "sci/tech"
    }
    
    def format_dataframe(df):
        data = []
        for _, row in df.iterrows():
            # Skip header row if it accidentally got loaded as data
            if str(row['label']).lower() in ['class index', 'label']:
                continue
                
            text_snippet = str(row['text'])[:300] 
            prompt = (f"Classify the following news article into one of four categories: "
                      f"world, sports, business, or sci/tech. "
                      f"Article: '{text_snippet}'. Predict category:")
            data.append({
                "input_text": prompt,
                "target_text": label_mapping.get(row['label'], "unknown"), # Change default to catch errors
                "type": "ag_news"
            })
        return pd.DataFrame(data)
    
    formatted_train = format_dataframe(train_df)
    formatted_test = format_dataframe(test_df)

    # 10% validation split
    train_final, val_final = train_test_split(
        formatted_train, test_size=0.10, stratify=formatted_train['target_text'], random_state=BaseConfig.SEED
    )
    
    return train_final, val_final, formatted_test

# ==========================================
# Dataset Class (UPDATED FOR SEQ2SEQ)
# ==========================================
class AGNewsSeq2SeqDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256, target_max_length=16):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize encoder inputs (the prompt)
        model_inputs = self.tokenizer(
            row['input_text'],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize decoder targets (the answer)
        labels = self.tokenizer(
            row['target_text'],
            max_length=self.target_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = model_inputs.input_ids.squeeze()
        attention_mask = model_inputs.attention_mask.squeeze()
        label_ids = labels.input_ids.squeeze()

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}

# ==========================================
# Pruning & Distillation
# ==========================================
def task_weighted_prune(model, dataloader, prune_ratio, device):
    model.eval()
    model.zero_grad()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        # break
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.grad is not None and param.dim() > 1:
                importance = torch.abs(param * param.grad)
                k = int(prune_ratio * importance.numel())
                if k > 0:
                    threshold = torch.kthvalue(importance.view(-1), k).values
                    mask = importance > threshold
                    param.data.mul_(mask)
    return model

def contrastive_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    return nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1)
    ) * (temperature ** 2)

# ==========================================
# Evaluation (UPDATED FOR SEQ2SEQ)
# ==========================================
def evaluate_seq2seq(model, dataloader, tokenizer, device, model_path=None):
    model.eval()
    preds, gts = [], []
    label_map = {"world": 0, "sports": 1, "business": 2, "sci/tech": 3}
    total_time, total_samples = 0, 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            for i in range(input_ids.size(0)):
                start_time = time.time()
                
                # T5 generates the target directly from the encoder input
                gen_out = model.generate(
                    input_ids=input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id
                )
                total_time += (time.time() - start_time)
                total_samples += 1

                gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True).strip().lower()
                
                # Extract ground truth, ignoring the -100 padding tokens
                valid_label_tokens = labels[i][labels[i] != -100]
                gt_text = tokenizer.decode(valid_label_tokens, skip_special_tokens=True).strip().lower()
                
                preds.append(gen_text)
                gts.append(gt_text)

    y_true = [label_map.get(g, 0) for g in gts]
    y_pred = [label_map.get(p, 0) for p in preds] # Defaults to 0 (world) if generation is gibberish
    
    model_size = os.path.getsize(model_path) / (1024**2) if model_path and os.path.exists(model_path) else sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "latency_sec_per_sample": total_time / total_samples if total_samples > 0 else 0,
        "model_size_mb": model_size
    }