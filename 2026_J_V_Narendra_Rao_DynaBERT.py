import os, sys, time, subprocess, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODEL_DIR    = os.path.expanduser("~/glue_finetune_models")
OUTPUT_DIR   = os.path.expanduser("~/glue_finetune_output")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glue_finetune_results.csv")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Fixed config, per spec ──────────────────────────────────────────────────
SEED           = 42
OPTIMIZER      = "AdamW"
WEIGHT_DECAY   = 0.01
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5
WARMUP_RATIO   = 0.10
EPOCHS         = 5
MAX_SEQ_LENGTH = 128


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── GLUE task definitions (mirrors ternarybert_run.py TASKS) ────────────────
TASKS = {
    "sst2": dict(repo_name="sst-2", glue_cfg="sst2", output_mode="classification",
                 fields=("sentence", None),       num_labels=2, label_mode="raw01"),
    "qnli": dict(repo_name="qnli",  glue_cfg="qnli", output_mode="classification",
                 fields=("question", "sentence"), num_labels=2, label_mode="class_names"),
    "qqp":  dict(repo_name="qqp",   glue_cfg="qqp",  output_mode="classification",
                 fields=("question1", "question2"), num_labels=2, label_mode="raw01"),
    "wnli": dict(repo_name="wnli",  glue_cfg="wnli", output_mode="classification",
                 fields=("sentence1", "sentence2"), num_labels=2, label_mode="raw01"),
    "mnli": dict(repo_name="mnli",  glue_cfg="mnli", output_mode="classification",
                 fields=("premise", "hypothesis"),  num_labels=3, label_mode="class_names"),
    "mrpc": dict(repo_name="mrpc",  glue_cfg="mrpc", output_mode="classification",
                 fields=("sentence1", "sentence2"), num_labels=2, label_mode="raw01"),
    "stsb": dict(repo_name="sts-b", glue_cfg="stsb", output_mode="regression",
                 fields=("sentence1", "sentence2"), num_labels=1, label_mode="regression"),
    "rte":  dict(repo_name="rte",   glue_cfg="rte",  output_mode="classification",
                 fields=("sentence1", "sentence2"), num_labels=2, label_mode="class_names"),
    "cola": dict(repo_name="cola",  glue_cfg="cola", output_mode="classification",
                 fields=("sentence", None),       num_labels=2, label_mode="raw01"),
}

TASK_ORDER = ["wnli", "rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]
TASK_OVERRIDES = {
    "wnli": dict(learning_rate=1e-5, use_class_weights=False, init_from_task="mnli"),
}

MODEL_REGISTRY = {
    "bert-base":    dict(display_name="BERT-base",    hf_checkpoint="bert-base-uncased"),
    "roberta-base": dict(display_name="RoBERTa-Base", hf_checkpoint="roberta-base"),
    "electra-base": dict(display_name="ELECTRA-Base", hf_checkpoint="google/electra-base-discriminator"),
}
MODEL_ORDER = ["bert-base", "roberta-base", "electra-base"]


def load_glue_examples(task_key, split, max_examples=None):
    cfg = TASKS[task_key]
    ds = load_dataset("glue", cfg["glue_cfg"])
    hf_split = "validation_matched" if (task_key == "mnli" and split == "validation") else split
    data = ds[hf_split]
    if max_examples:
        data = data.select(range(min(max_examples, len(data))))
    label_names = data.features["label"].names if cfg["label_mode"] == "class_names" else None
    f1, f2 = cfg["fields"]
    texts_a, texts_b, labels = [], [], []
    for row in data:
        texts_a.append(row[f1])
        texts_b.append(row[f2] if f2 else None)
        if cfg["label_mode"] == "regression":
            labels.append(float(row["label"]))
        else:
            labels.append(int(row["label"]))
    return texts_a, texts_b, labels


class GlueDataset(Dataset):
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_seq_length, output_mode):
        self.texts_a, self.texts_b, self.labels = texts_a, texts_b, labels
        self.tokenizer, self.max_seq_length, self.output_mode = tokenizer, max_seq_length, output_mode

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts_a[idx], self.texts_b[idx], truncation=True,
                             padding="max_length", max_length=self.max_seq_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        dtype = torch.float if self.output_mode == "regression" else torch.long
        item["labels"] = torch.tensor(self.labels[idx], dtype=dtype)
        return item


def full_eval_metrics(model, eval_loader, output_mode, num_labels):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = (torch.argmax(logits, dim=-1) if output_mode == "classification"
                     else logits.squeeze(-1)).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    preds, labels = np.concatenate(all_preds), np.concatenate(all_labels)

    if output_mode == "regression":
        return {"accuracy": float(pearsonr(preds, labels)[0]),
                "spearman": float(spearmanr(preds, labels)[0]),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "mcc": None}

    avg = "macro" if num_labels > 2 else "binary"
    return {
        "accuracy":  float((preds == labels).mean()),
        "spearman":  None,
        "precision": float(precision_score(labels, preds, average=avg, zero_division=0)),
        "recall":    float(recall_score(labels, preds, average=avg, zero_division=0)),
        "f1":        float(f1_score(labels, preds, average=avg, zero_division=0)),
        "mcc":       float(matthews_corrcoef(labels, preds)) if num_labels == 2 else None,
    }


def primary_metric(task_key, m):
    if task_key == "cola" and m.get("mcc") is not None:
        return m["mcc"]
    if task_key in ("mrpc", "qqp"):
        return (m["accuracy"] + m["f1"]) / 2
    return m["accuracy"]


def fine_tune_and_eval(task_key, model_key, max_train_examples=None, max_eval_examples=None):
    set_seed(SEED)
    cfg = TASKS[task_key]
    model_info = MODEL_REGISTRY[model_key]
    hf_checkpoint = model_info["hf_checkpoint"]
    output_mode, num_labels = cfg["output_mode"], cfg["num_labels"]

    overrides = TASK_OVERRIDES.get(task_key, {})
    lr = overrides.get("learning_rate", LEARNING_RATE)
    use_class_weights = overrides.get("use_class_weights", False)

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint)

    train_a, train_b, train_y = load_glue_examples(task_key, "train", max_train_examples)
    dev_a, dev_b, dev_y       = load_glue_examples(task_key, "validation", max_eval_examples)

    class_weights = None
    if use_class_weights and output_mode == "classification":
        counts = np.bincount(train_y, minlength=num_labels)
        counts = np.where(counts == 0, 1, counts)
        class_weights = torch.tensor(len(train_y) / (len(counts) * counts), dtype=torch.float).to(DEVICE)
        print(f"  [{task_key}/{model_key}] using class-weighted loss: {class_weights.tolist()}")

    train_ds = GlueDataset(train_a, train_b, train_y, tokenizer, MAX_SEQ_LENGTH, output_mode)
    dev_ds   = GlueDataset(dev_a, dev_b, dev_y, tokenizer, MAX_SEQ_LENGTH, output_mode)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader  = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    problem_type = "regression" if output_mode == "regression" else "single_label_classification"

    init_task = overrides.get("init_from_task")
    init_dir = None
    if init_task:
        candidate_dir = os.path.join(OUTPUT_DIR, model_key, TASKS[init_task]["repo_name"])
        if os.path.exists(candidate_dir):
            init_dir = candidate_dir
        else:
            print(f"  [{task_key}/{model_key}] WARNING: STILTs source '{init_task}' checkpoint not found "
                  f"at {candidate_dir} -- falling back to raw pretrained weights. Run {init_task} for "
                  f"{model_key} first to enable transfer.")

    if init_dir:
        print(f"  [{task_key}/{model_key}] initializing from {init_task}-finetuned checkpoint at {init_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            init_dir, num_labels=num_labels, problem_type=problem_type,
            ignore_mismatched_sizes=True).to(DEVICE)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_checkpoint, num_labels=num_labels, problem_type=problem_type).to(DEVICE)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    total_steps  = max(1, EPOCHS * len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_metric, best_state, global_step = -1e9, None, 0
    eval_every = max(50, len(train_loader) // 4)

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if class_weights is not None:
                fwd_kwargs = {k: v for k, v in batch.items() if k != "labels"}
                logits = model(**fwd_kwargs).logits
                loss = torch.nn.functional.cross_entropy(logits, batch["labels"], weight=class_weights)
            else:
                out = model(**batch)
                loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if step % 200 == 0:
                print(f"  [{task_key}/{model_key}] epoch {epoch+1}/{EPOCHS} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f}")

            if global_step % eval_every == 0:
                m = full_eval_metrics(model, eval_loader, output_mode, num_labels)
                pm = primary_metric(task_key, m)
                print(f"  [{task_key}/{model_key}] step {global_step}/{total_steps}  primary_metric={pm:.4f}")
                if pm > best_metric:
                    best_metric = pm
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()

    final_m = full_eval_metrics(model, eval_loader, output_mode, num_labels)
    if primary_metric(task_key, final_m) > best_metric:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metric = primary_metric(task_key, final_m)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    final_metrics = full_eval_metrics(model, eval_loader, output_mode, num_labels)

    save_dir = os.path.join(OUTPUT_DIR, model_key, cfg["repo_name"])
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return model, tokenizer, final_metrics, save_dir


# ── Benchmarking (latency / throughput / memory / energy) ──────────────────
def get_gpu_power_watts():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def model_disk_size_mb(path_dir):
    total = 0
    for fname in os.listdir(path_dir):
        fpath = os.path.join(path_dir, fname)
        if os.path.isfile(fpath) and (fname.endswith(".bin") or fname.endswith(".safetensors")):
            total += os.path.getsize(fpath)
    return total / 1e6


def benchmark_model(model, tokenizer, max_seq_length, n_warmup=10, n_runs=50):
    model.eval()
    dummy_text = "this is a short sentence used purely for latency benchmarking"
    enc = tokenizer(dummy_text, truncation=True, padding="max_length", max_length=max_seq_length,
                    return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        for _ in range(n_warmup):
            model(**enc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        power_samples, start = [], time.perf_counter()
        for _ in range(n_runs):
            model(**enc)
            p = get_gpu_power_watts()
            if p is not None:
                power_samples.append(p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n_runs) * 1000
    avg_power  = float(np.mean(power_samples)) if power_samples else None
    return latency_ms, 1000.0 / latency_ms, (avg_power * latency_ms if avg_power else None)


# ── Checkpointing ────────────────────────────────────────────────────────────
COLUMNS = ["task_key", "model_key", "Model name", "Memory Size", "Latency", "Accuracy",
           "Spearman", "Precision", "Recall", "F1-score", "Throughput", "Energy Consumption"]


def load_checkpoint():
    if not os.path.exists(RESULTS_PATH):
        return {}
    df = pd.read_csv(RESULTS_PATH)
    result = {}
    for _, row in df.iterrows():
        key = (row["task_key"], row.get("model_key", "bert-base"))
        result[key] = row.drop(["task_key", "model_key"], errors="ignore").to_dict()
    return result


def save_checkpoint(results):
    disk_results = load_checkpoint()
    disk_results.update(results)
    rows = [{"task_key": tk, "model_key": mk, **v} for (tk, mk), v in disk_results.items()]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(RESULTS_PATH, index=False)
    print(f"  Checkpoint saved -> {RESULTS_PATH}")
    return disk_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=TASK_ORDER)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=MODEL_ORDER)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 200 train / 100 eval examples (NOT paper-comparable)")
    args = parser.parse_args()

    if args.quick:
        print("WARNING: --quick mode active. Results will NOT be accuracy-representative.")

    set_seed(SEED)
    results = load_checkpoint()

    for model_key in args.models:
        for task_key in args.tasks:
            key = (task_key, model_key)
            if key in results:
                print(f"Skipping {task_key}/{model_key} (already in checkpoint)")
                continue

            max_train = 200 if args.quick else None
            max_eval  = 100 if args.quick else None

            try:
                print(f"\n{'='*60}")
                print(f"  {task_key.upper()} / {model_key} | full fine-tuning (epochs={EPOCHS})")
                print(f"{'='*60}")
                model, tokenizer, metrics, out_dir = fine_tune_and_eval(
                    task_key, model_key, max_train_examples=max_train, max_eval_examples=max_eval)

                print(f"\n  {task_key.upper()} / {model_key} | benchmarking")
                latency_ms, throughput, energy_mj = benchmark_model(model, tokenizer, MAX_SEQ_LENGTH)
                mem_mb = model_disk_size_mb(out_dir)

                def pct(v):
                    return None if v is None or (isinstance(v, float) and v != v) else round(v * 100, 2)

                results[key] = {
                    "Model name":         MODEL_REGISTRY[model_key]["display_name"],
                    "Memory Size":        round(mem_mb, 2),
                    "Latency":            round(latency_ms, 3),
                    "Accuracy":           pct(metrics["accuracy"]),
                    "Spearman":           pct(metrics.get("spearman")),
                    "Precision":          pct(metrics["precision"]),
                    "Recall":             pct(metrics["recall"]),
                    "F1-score":           pct(metrics["f1"]),
                    "Throughput":         round(throughput, 2),
                    "Energy Consumption": round(energy_mj, 3) if energy_mj is not None else None,
                }
                results = save_checkpoint(results)
                print(f"\n  {task_key.upper()}/{model_key} DONE: {results[key]}\n")

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n!!! {task_key}/{model_key} FAILED: {e}")
                import traceback; traceback.print_exc()
                continue

    # ── Final results table ──────────────────────────────────────────────────
    DATASET_ORDER = ["sst2", "qnli", "qqp", "wnli", "mnli", "mrpc", "stsb", "rte", "cola"]
    SHEET_NAME = {"sst2": "SST2", "qnli": "QNLI", "qqp": "QQP", "wnli": "WNLI", "mnli": "MNLI",
                 "mrpc": "MRPC", "stsb": "STS-B", "rte": "RTE", "cola": "COLA"}

    rows = []
    for model_key in args.models:
        for tk in DATASET_ORDER:
            key = (tk, model_key)
            if key in results:
                rows.append({"Dataset": SHEET_NAME[tk], **results[key]})
    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        csv_path = os.path.expanduser("~/GLUE_finetune_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
