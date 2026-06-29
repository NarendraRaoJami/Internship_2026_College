import os, sys, time, copy, random, subprocess, argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.optim.lr_scheduler import LambdaLR

from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from datasets import load_dataset

REPO_DIR = os.path.expanduser("~/Pretrained-Language-Model")
if not os.path.exists(REPO_DIR):
    print("Cloning TernaryBERT repo...")
    os.system(f"git clone --depth 1 --filter=blob:none --sparse https://github.com/huawei-noah/Pretrained-Language-Model.git {REPO_DIR}")
    os.system(f"cd {REPO_DIR} && git sparse-checkout set TernaryBERT")

TB_DIR = os.path.join(REPO_DIR, "TernaryBERT")
sys.path.insert(0, TB_DIR)
os.chdir(TB_DIR)
print("Working dir:", os.getcwd())

from transformer import (BertForSequenceClassification, BertConfig, BertTokenizer,
                         BertAdam, WEIGHTS_NAME, CONFIG_NAME)
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformers import (BertTokenizer as HFBertTokenizer,
                          BertForSequenceClassification as HFBertForSequenceClassification)
from utils_glue import (
    InputExample, convert_examples_to_features,
    MrpcProcessor, MnliProcessor, ColaProcessor, Sst2Processor,
    StsbProcessor, QqpProcessor, QnliProcessor, RteProcessor, WnliProcessor,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODEL_DIR  = os.path.expanduser("~/ternarybert_models")   
OUTPUT_DIR = os.path.expanduser("~/ternarybert_output")   
RESULTS_PATH = os.path.expanduser("~/ternarybert_results.csv")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = {
    "sst2": dict(repo_name="sst-2", glue_cfg="sst2", processor=Sst2Processor(), output_mode="classification",
                 max_seq_length=64,  batch_size=32, eval_step=200,  fields=("sentence",  None),         label_mode="raw01"),
    "qnli": dict(repo_name="qnli",  glue_cfg="qnli", processor=QnliProcessor(), output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=1000, fields=("question",  "sentence"),    label_mode="class_names"),
    "qqp":  dict(repo_name="qqp",   glue_cfg="qqp",  processor=QqpProcessor(),  output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=1000, fields=("question1", "question2"),   label_mode="raw01"),
    "wnli": dict(repo_name="wnli",  glue_cfg="wnli", processor=WnliProcessor(), output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=50,   fields=("sentence1", "sentence2"),   label_mode="raw01"),
    "mnli": dict(repo_name="mnli",  glue_cfg="mnli", processor=MnliProcessor(), output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=1000, fields=("premise",   "hypothesis"),  label_mode="class_names"),
    "mrpc": dict(repo_name="mrpc",  glue_cfg="mrpc", processor=MrpcProcessor(), output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=200,  fields=("sentence1", "sentence2"),   label_mode="raw01"),
    "stsb": dict(repo_name="sts-b", glue_cfg="stsb", processor=StsbProcessor(), output_mode="regression",
                 max_seq_length=128, batch_size=32, eval_step=50,   fields=("sentence1", "sentence2"),   label_mode="regression"),
    "rte":  dict(repo_name="rte",   glue_cfg="rte",  processor=RteProcessor(),  output_mode="classification",
                 max_seq_length=128, batch_size=32, eval_step=100,  fields=("sentence1", "sentence2"),   label_mode="class_names"),
    "cola": dict(repo_name="cola",  glue_cfg="cola", processor=ColaProcessor(), output_mode="classification",
                 max_seq_length=64,  batch_size=16, eval_step=50,   fields=("sentence",  None),          label_mode="raw01"),
}

TASK_EPOCHS = {
    "sst2": 3, "qnli": 3, "qqp": 3, "mnli": 3,
    "wnli": 5, "rte":  5, "mrpc": 5, "stsb": 5, "cola": 5,
}

TASK_ORDER = ["wnli", "rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]

def load_glue_examples(task_key, split, max_examples=None):
    cfg = TASKS[task_key]
    ds = load_dataset("glue", cfg["glue_cfg"])
    hf_split = "validation_matched" if (task_key == "mnli" and split == "validation") else split
    data = ds[hf_split]
    if max_examples:
        data = data.select(range(min(max_examples, len(data))))
    label_names = data.features["label"].names if cfg["label_mode"] == "class_names" else None
    f1, f2 = cfg["fields"]
    examples = []
    for i, row in enumerate(data):
        text_a = row[f1]
        text_b = row[f2] if f2 else None
        if cfg["label_mode"] == "regression":
            label = row["label"]
        elif cfg["label_mode"] == "class_names":
            label = label_names[row["label"]]
        else:
            label = str(row["label"])
        examples.append(InputExample(guid=f"{split}-{i}", text_a=text_a, text_b=text_b, label=label))
    return examples


def get_tensor_data(output_mode, features):
    label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.float if output_mode == "regression" else torch.long,
    )
    input_ids   = torch.tensor([f.input_ids   for f in features], dtype=torch.long)
    input_mask  = torch.tensor([f.input_mask  for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    seq_lengths = torch.tensor([f.seq_length  for f in features], dtype=torch.long)
    return torch.utils.data.TensorDataset(input_ids, input_mask, segment_ids, label_ids, seq_lengths), label_ids


class _HFGlueDataset(Dataset):
    def __init__(self, examples, label_map, max_seq_length, tokenizer, output_mode):
        self.examples, self.label_map = examples, label_map
        self.max_seq_length, self.tokenizer, self.output_mode = max_seq_length, tokenizer, output_mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex.text_a, ex.text_b, truncation=True, padding="max_length",
                             max_length=self.max_seq_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.output_mode == "classification":
            item["labels"] = torch.tensor(self.label_map[ex.label], dtype=torch.long)
        else:
            item["labels"] = torch.tensor(float(ex.label), dtype=torch.float)
        return item

def fine_tune_teacher(task_key, epochs=3, lr=2e-5, max_train_examples=None):
    cfg = TASKS[task_key]
    label_list = cfg["processor"].get_labels()
    num_labels = 1 if cfg["output_mode"] == "regression" else len(label_list)
    label_map  = {l: i for i, l in enumerate(label_list)} if cfg["output_mode"] == "classification" else None

    hf_tok = HFBertTokenizer.from_pretrained("bert-base-uncased")
    train_examples = load_glue_examples(task_key, "train", max_examples=max_train_examples)
    train_ds = _HFGlueDataset(train_examples, label_map, cfg["max_seq_length"], hf_tok, cfg["output_mode"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    problem_type = "regression" if cfg["output_mode"] == "regression" else "single_label_classification"
    model = HFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, problem_type=problem_type).to(DEVICE)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    total_steps  = max(1, epochs * len(train_loader))
    warmup_steps = int(0.1 * total_steps)

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    for ep in range(epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if step % 200 == 0:
                print(f"  [{task_key}] teacher epoch {ep+1}/{epochs}  step {step}/{len(train_loader)}  loss {out.loss.item():.4f}")

    save_dir = os.path.join(MODEL_DIR, cfg["repo_name"])
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    model.config.to_json_file(os.path.join(save_dir, "config.json"))
    hf_tok.save_vocabulary(save_dir)
    print(f"  [{task_key}] teacher saved to {save_dir}")
    return save_dir

def predict_all(model, eval_dataloader, output_mode):
    model.eval()
    all_preds, all_labels = [], []
    for batch in eval_dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, _ = batch
        with torch.no_grad():
            logits, _, _ = model(input_ids, segment_ids, input_mask)
        preds = (torch.argmax(logits, dim=-1) if output_mode == "classification"
                 else logits.squeeze(-1)).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(label_ids.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def full_eval_metrics(model, output_mode, num_labels, eval_dataloader):
    preds, labels = predict_all(model, eval_dataloader, output_mode)
    if output_mode == "regression":
        return {"accuracy": pearsonr(preds, labels)[0],
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    avg = "macro" if num_labels > 2 else "binary"
    return {
        "accuracy":  float((preds == labels).mean()),
        "precision": float(precision_score(labels, preds, average=avg, zero_division=0)),
        "recall":    float(recall_score(labels, preds, average=avg, zero_division=0)),
        "f1":        float(f1_score(labels, preds, average=avg, zero_division=0)),
        "mcc":       float(matthews_corrcoef(labels, preds)) if num_labels == 2 else None,
    }


def primary_metric(repo_task, m):
    if repo_task == "cola" and m.get("mcc") is not None:
        return m["mcc"]
    if repo_task in ("mrpc", "qqp"):
        return (m["accuracy"] + m["f1"]) / 2
    return m["accuracy"]


def soft_cross_entropy(predicts, targets):
    return (-torch.nn.functional.softmax(targets, dim=-1) *
             torch.nn.functional.log_softmax(predicts, dim=-1)).mean()

def run_quant_glue(task_key, epochs=3, lr=2e-5, weight_bits=2, input_bits=8, clip_val=2.5,
                   max_train_examples=None, max_eval_examples=None):
    cfg = TASKS[task_key]
    repo_task, output_mode = cfg["repo_name"], cfg["output_mode"]
    label_list = cfg["processor"].get_labels()
    num_labels = 1 if output_mode == "regression" else len(label_list)
    max_seq_length, batch_size, eval_step = cfg["max_seq_length"], cfg["batch_size"], cfg["eval_step"]

    task_model_dir = os.path.join(MODEL_DIR, repo_task)
    tokenizer = BertTokenizer.from_pretrained(task_model_dir, do_lower_case=True)

    train_examples = load_glue_examples(task_key, "train", max_examples=max_train_examples)
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, output_mode)
    train_data, _  = get_tensor_data(output_mode, train_features)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    dev_examples = load_glue_examples(task_key, "validation", max_examples=max_eval_examples)
    dev_features = convert_examples_to_features(dev_examples, label_list, max_seq_length, tokenizer, output_mode)
    eval_data, _ = get_tensor_data(output_mode, dev_features)
    eval_dataloader = DataLoader(eval_data, sampler=SequentialSampler(eval_data), batch_size=batch_size)

    teacher_model = BertForSequenceClassification.from_pretrained(task_model_dir, num_labels=num_labels).to(DEVICE)
    teacher_model.eval()

    student_config = BertConfig.from_pretrained(task_model_dir, quantize_act=True,
                                                weight_bits=weight_bits, input_bits=input_bits, clip_val=clip_val)
    student_model  = QuantBertForSequenceClassification.from_pretrained(
        task_model_dir, config=student_config, num_labels=num_labels).to(DEVICE)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in student_model.named_parameters() if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    num_train_steps = max(1, int(len(train_features) / batch_size) * epochs)
    optimizer = BertAdam(optimizer_grouped_parameters, schedule="warmup_linear",
                         lr=lr, warmup=0.1, t_total=num_train_steps)
    loss_mse = MSELoss()

    best_metric, best_state, global_step = -1e9, None, 0

    for epoch in range(epochs):
        student_model.train()
        for batch in train_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch

            student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

            cls_loss = (soft_cross_entropy(student_logits, teacher_logits) if output_mode == "classification"
                        else loss_mse(student_logits, teacher_logits))

            att_loss = sum(
                loss_mse(torch.where(s <= -1e2, torch.zeros_like(s), s),
                         torch.where(t <= -1e2, torch.zeros_like(t), t))
                for s, t in zip(student_atts, teacher_atts)
            )
            rep_loss = sum(loss_mse(s, t) for s, t in zip(student_reps, teacher_reps))

            loss = cls_loss + att_loss + rep_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            if global_step % eval_step == 0:
                m  = full_eval_metrics(student_model, output_mode, num_labels, eval_dataloader)
                pm = primary_metric(repo_task, m)
                print(f"  [{task_key}] step {global_step}/{num_train_steps}  primary_metric={pm:.4f}")
                if pm > best_metric:
                    best_metric, best_state = pm, copy.deepcopy(student_model.state_dict())
                student_model.train()

    final_m = full_eval_metrics(student_model, output_mode, num_labels, eval_dataloader)
    if primary_metric(repo_task, final_m) > best_metric:
        best_state = copy.deepcopy(student_model.state_dict())
    if best_state:
        student_model.load_state_dict(best_state)

    quant_model = copy.deepcopy(student_model)
    for _, module in quant_model.named_modules():
        if hasattr(module, "weight_quantizer"):
            module.weight.data = module.weight_quantizer.apply(
                module.weight, module.weight_clip_val, module.weight_bits, True)

    task_output_dir = os.path.join(OUTPUT_DIR, repo_task, "quant")
    os.makedirs(task_output_dir, exist_ok=True)
    torch.save(quant_model.state_dict(), os.path.join(task_output_dir, WEIGHTS_NAME))
    quant_model.config.to_json_file(os.path.join(task_output_dir, CONFIG_NAME))
    tokenizer.save_vocabulary(task_output_dir)

    final_metrics = full_eval_metrics(quant_model, output_mode, num_labels, eval_dataloader)
    return quant_model, tokenizer, final_metrics, task_output_dir

def get_gpu_power_watts():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def model_disk_size_mb(path_dir):
    return os.path.getsize(os.path.join(path_dir, WEIGHTS_NAME)) / 1e6


def benchmark_model(model, tokenizer, max_seq_length, n_warmup=10, n_runs=50):
    model.eval()
    dummy = [InputExample(guid="bench-0",
                          text_a="this is a short sentence used purely for latency benchmarking",
                          text_b=None, label=None)]
    feats = convert_examples_to_features(dummy, ["0", "1"], max_seq_length, tokenizer, "classification")
    input_ids   = torch.tensor([feats[0].input_ids],   dtype=torch.long, device=DEVICE)
    input_mask  = torch.tensor([feats[0].input_mask],  dtype=torch.long, device=DEVICE)
    segment_ids = torch.tensor([feats[0].segment_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, segment_ids, input_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        power_samples, start = [], time.perf_counter()
        for _ in range(n_runs):
            model(input_ids, segment_ids, input_mask)
            p = get_gpu_power_watts()
            if p is not None:
                power_samples.append(p)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n_runs) * 1000
    avg_power  = float(np.mean(power_samples)) if power_samples else None
    return latency_ms, 1000.0 / latency_ms, (avg_power * latency_ms if avg_power else None)

RESULTS_PATH = os.path.expanduser("~/ternarybert_results.csv")

COLUMNS = ["task_key", "Model name", "Memory Size", "Latency", "Accuracy",
           "Bits", "Precision", "Recall", "F1-score", "Throughput", "Energy Consumption"]

def load_checkpoint():
    """Returns dict of {task_key: row_dict} from CSV, or empty dict if no file."""
    if not os.path.exists(RESULTS_PATH):
        return {}
    df = pd.read_csv(RESULTS_PATH)
    return {row["task_key"]: row.drop("task_key").to_dict() for _, row in df.iterrows()}

def save_checkpoint(results):
    rows = [{"task_key": k, **v} for k, v in results.items()]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(RESULTS_PATH, index=False)
    print(f"  Checkpoint saved → {RESULTS_PATH}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  nargs="+", choices=list(TASKS), default=TASK_ORDER,
                        help="Tasks to run (default: all in paper order)")
    parser.add_argument("--quick",  action="store_true",
                        help="Quick test: 200 train / 100 eval examples (NOT paper-comparable)")
    args = parser.parse_args()

    if args.quick:
        print("WARNING: --quick mode active. Results will NOT match the paper.")

    results = load_checkpoint()

    for task_key in args.tasks:
        if task_key in results:
            print(f"Skipping {task_key} (already in checkpoint)")
            continue

        cfg      = TASKS[task_key]
        n_epochs = TASK_EPOCHS[task_key]
        max_train = 200 if args.quick else None
        max_eval  = 100 if args.quick else None

        try:
            print(f"\n{'='*60}")
            print(f"  {task_key.upper()} | stage 1: fine-tuning fp32 teacher (epochs={n_epochs})")
            print(f"{'='*60}")
            fine_tune_teacher(task_key, epochs=n_epochs, lr=2e-5, max_train_examples=max_train)

            print(f"\n{'='*60}")
            print(f"  {task_key.upper()} | stage 2: W2A8 quantization-aware distillation (epochs={n_epochs})")
            print(f"{'='*60}")
            quant_model, tokenizer, metrics, out_dir = run_quant_glue(
                task_key, epochs=n_epochs, lr=2e-5, weight_bits=2, input_bits=8,
                max_train_examples=max_train, max_eval_examples=max_eval)

            print(f"\n  {task_key.upper()} | stage 3: benchmarking")
            latency_ms, throughput, energy_mj = benchmark_model(quant_model, tokenizer, cfg["max_seq_length"])
            mem_mb = model_disk_size_mb(out_dir)

            def pct(v):
                return None if v is None or (isinstance(v, float) and v != v) else round(v * 100, 2)

            results[task_key] = {
                "Model name":         "BERT-base",
                "Memory Size":        round(mem_mb, 2),
                "Latency":            round(latency_ms, 3),
                "Accuracy":           pct(metrics["accuracy"]),
                "Bits":               2,
                "Precision":          pct(metrics["precision"]),
                "Recall":             pct(metrics["recall"]),
                "F1-score":           pct(metrics["f1"]),
                "Throughput":         round(throughput, 2),
                "Energy Consumption": round(energy_mj, 3) if energy_mj is not None else None,
            }
            save_checkpoint(results)
            print(f"\n  {task_key.upper()} DONE: {results[task_key]}\n")

        except Exception as e:
            print(f"\n!!! {task_key} FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

    # ── Final results table ────────────────────────────────────────────────────
    DATASET_ORDER = ["sst2", "qnli", "qqp", "wnli", "mnli", "mrpc", "stsb", "rte", "cola"]
    SHEET_NAME    = {"sst2": "SST2", "qnli": "QNLI", "qqp": "QQP", "wnli": "WNLI", "mnli": "MNLI",
                     "mrpc": "MRPC", "stsb": "STS-B", "rte": "RTE", "cola": "COLA"}

    rows = [{"Dataset": SHEET_NAME[tk], **results[tk]} for tk in DATASET_ORDER if tk in results]
    if rows:
        df = pd.DataFrame(rows)
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        csv_path = os.path.expanduser("~/TernaryBERT_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()