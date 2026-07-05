import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

# Updated imports matching the new Seq2Seq logic
from flanutils import (
    BaseConfig, load_data, prepare_dataset_agnews, AGNewsSeq2SeqDataset,
    task_weighted_prune, contrastive_distillation_loss, evaluate_seq2seq, clear_gpu_memory
)

class FlanConfig(BaseConfig):
    MODEL_NAME = "google/flan-t5-xl"
    BATCH_SIZE = 4 # T5-small is tiny, you can usually push batch size up safely
    EPOCHS_STAGE1 = 3
    EPOCHS_PRUNE_FT = 2 
    DISTILL_EPOCHS = 3
    PRUNE_RATIO = 0.3
    GRAD_ACCUM = 4 
    CLIP_NORM = 1.0
    LR = 5e-5
    ARTIFACTS_DIR = "artifacts-flan-t5-xl-agnews"

def train_step(model, batch, optimizer, device):
    """
    Performs a single training step using BF16 Mixed Precision.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / FlanConfig.GRAD_ACCUM

    loss.backward()
    return loss.item() * FlanConfig.GRAD_ACCUM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Running AG-NEWS Pipeline on {device} (BF16 Mode)\n")
    os.makedirs(FlanConfig.ARTIFACTS_DIR, exist_ok=True)

    # ========== DATA SETUP ==========
    train_raw, test_raw = load_data(train_path="agnewstrain.csv", test_path="agnewstest.csv")
    train_df, val_df, test_df = prepare_dataset_agnews(train_raw, test_raw)
    
    tokenizer = AutoTokenizer.from_pretrained(FlanConfig.MODEL_NAME)
    # Note: T5 already has a proper pad_token, so we do NOT set pad_token = eos_token

    train_loader = DataLoader(AGNewsSeq2SeqDataset(train_df, tokenizer), batch_size=FlanConfig.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(AGNewsSeq2SeqDataset(test_df, tokenizer), batch_size=1)
    metrics_history = []

    # ========== STAGE 1: TEACHER TRAINING ==========
    print("📘 Stage 1: Teacher Training (AG-NEWS)")
    # Swap to AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        FlanConfig.MODEL_NAME, 
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=FlanConfig.LR)

    for epoch in range(FlanConfig.EPOCHS_STAGE1):
        model.train()
        for i, batch in enumerate(train_loader):
            loss_val = train_step(model, batch, optimizer, device)
            
            if (i + 1) % FlanConfig.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), FlanConfig.CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()
                
            if i % 20 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss {loss_val:.4f}")

    teacher_path = os.path.join(FlanConfig.ARTIFACTS_DIR, "agnews_teacher.pt")
    torch.save(model.state_dict(), teacher_path)
    metrics_history.append({**evaluate_seq2seq(model, test_loader, tokenizer, device, teacher_path), "stage": "Teacher"})
    
    # ========== STAGE 2: PRUNING & FINE-TUNING ==========
    print("✂️ Stage 2: Pruning")
    model = task_weighted_prune(model, train_loader, FlanConfig.PRUNE_RATIO, device)
    
    print("🔧 Fine-tuning Pruned Model...")
    optimizer_pruned = torch.optim.AdamW(model.parameters(), lr=FlanConfig.LR)

    for epoch in range(FlanConfig.EPOCHS_PRUNE_FT):
        model.train()
        for i, batch in enumerate(train_loader):
            loss_val = train_step(model, batch, optimizer_pruned, device)
            
            if (i + 1) % FlanConfig.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), FlanConfig.CLIP_NORM)
                optimizer_pruned.step()
                optimizer_pruned.zero_grad()

    pruned_path = os.path.join(FlanConfig.ARTIFACTS_DIR, "agnews_pruned.pt")
    torch.save(model.state_dict(), pruned_path)
    metrics_history.append({**evaluate_seq2seq(model, test_loader, tokenizer, device, pruned_path), "stage": "Pruned"})
    
    student = model
    del model
    clear_gpu_memory()

    # ========== STAGE 3: DISTILLATION ==========
    print("🧠 Stage 3: Distillation")
    teacher = AutoModelForSeq2SeqLM.from_pretrained(FlanConfig.MODEL_NAME, dtype=torch.bfloat16).to(device)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()

    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=FlanConfig.LR)

    for epoch in range(FlanConfig.DISTILL_EPOCHS):
        student.train()
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Crucial change: Pass labels to teacher so it evaluates decoder logits properly
                    teacher_logits = teacher(input_ids=ids, attention_mask=mask, labels=labels).logits

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                student_out = student(input_ids=ids, attention_mask=mask, labels=labels)
                d_loss = contrastive_distillation_loss(student_out.logits, teacher_logits)
                loss = (d_loss + student_out.loss) / FlanConfig.GRAD_ACCUM

            loss.backward()
            
            if (i + 1) % FlanConfig.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), FlanConfig.CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

    distilled_path = os.path.join(FlanConfig.ARTIFACTS_DIR, "agnews_distilled.pt")
    torch.save(student.state_dict(), distilled_path)
    metrics_history.append({**evaluate_seq2seq(student, test_loader, tokenizer, device, distilled_path), "stage": "Distilled"})
    
    del teacher, student
    clear_gpu_memory()

    # ========== STAGE 4: QUANTIZATION (CPU) ==========
    print("⚡ Stage 4: Quantization")
    # Swap to AutoModelForSeq2SeqLM here as well
    model = AutoModelForSeq2SeqLM.from_pretrained(FlanConfig.MODEL_NAME, dtype=torch.float32)
    model.load_state_dict(torch.load(distilled_path, map_location="cpu"))
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    quant_path = os.path.join(FlanConfig.ARTIFACTS_DIR, "agnews_quantized.pt")
    torch.save(quantized_model.state_dict(), quant_path)
    metrics_history.append({**evaluate_seq2seq(quantized_model, test_loader, tokenizer, "cpu", quant_path), "stage": "Quantized"})

    # Final Report
    report_df = pd.DataFrame(metrics_history)
    report_df.to_csv(os.path.join(FlanConfig.ARTIFACTS_DIR, "agnews_metrics.csv"), index=False)
    print("\n✅ AG-NEWS Pipeline Complete\n", report_df)

if __name__ == "__main__":
    main()