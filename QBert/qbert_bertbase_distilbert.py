SCRIPT_VERSION = "v2-fix-distilbert-stsb-checkpoint-2026-07-26"

import os, gc, copy, time, warnings, traceback
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

print(f"System Log: Running script version: {SCRIPT_VERSION}")

BASE_DIR = "/media/shared/Devshree/edgellm3/glue_pruning/qbert"
MASTER_CSV = f"{BASE_DIR}/qbert_bertbase_distilbert_results.csv"

WEIGHT_BITS = [8, 16]
MAX_LEN = 128
LATENCY_WARMUP, LATENCY_RUNS, LATENCY_BATCH = 20, 100, 32
GPU_ACTIVE_POWER_W = 210
MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "BERT-Base": {
        "sst2": "textattack/bert-base-uncased-SST-2",
        "qnli": "textattack/bert-base-uncased-QNLI",
        "rte":  "textattack/bert-base-uncased-RTE",
        "mrpc": "textattack/bert-base-uncased-MRPC",
        "wnli": "textattack/bert-base-uncased-WNLI",
        "mnli": "textattack/bert-base-uncased-MNLI",
        "qqp":  "textattack/bert-base-uncased-QQP",
        "stsb": "textattack/bert-base-uncased-STS-B",
        "cola": "textattack/bert-base-uncased-CoLA",
    },
    "DistilBERT": {
        "sst2": "textattack/distilbert-base-uncased-SST-2",
        "qnli": "textattack/distilbert-base-uncased-QNLI",
        "rte":  "textattack/distilbert-base-uncased-RTE",
        "mrpc": "textattack/distilbert-base-uncased-MRPC",
        "wnli": "textattack/distilbert-base-uncased-WNLI",
        "mnli": "textattack/distilbert-base-uncased-MNLI",
        "qqp":  "textattack/distilbert-base-uncased-QQP",
        "stsb": "textattack/distilbert-base-cased-STS-B",  
        "cola": "textattack/distilbert-base-uncased-CoLA",
    },
}
MODELS = list(CHECKPOINTS.keys())

BASE_TOKENIZER_FOR_MODEL = {
    "BERT-Base": "bert-base-uncased",
    "DistilBERT": "distilbert-base-uncased",
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "rte":  ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
TASK_TO_SPLIT = {"mnli": "validation_matched"}  # default is "validation"
IS_REGRESSION = {"stsb"}
TASKS = list(TASK_TO_KEYS.keys())


# ==============================================================================
# Q-BERT: asymmetric per-channel weight quantization (fake-quant)
# ==============================================================================
class AsymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, x_min, x_max, per_channel=True, _=False):
        if k == 32: return x
        n = float(2**k - 1)
        if per_channel and isinstance(x_min, torch.Tensor) and x_min.numel() > 1:
            x_min = x_min.to(x.device); x_max = x_max.to(x.device)
            s  = ((x_max - x_min) / n).clamp(min=1e-8).unsqueeze(0)
            zp = (-x_min.unsqueeze(0) / s).round().clamp(0, n)
            return ((((x / s).round() + zp).clamp(0, n) - zp) * s).to(x.dtype)
        xmin = x_min.min().item() if isinstance(x_min, torch.Tensor) else float(x_min)
        xmax = x_max.max().item() if isinstance(x_max, torch.Tensor) else float(x_max)
        s = (xmax - xmin) / n
        if s < 1e-8: return x
        zp = max(0, min(int(n), int(round(-xmin / s))))
        return ((((x / s).round() + zp).clamp(0, n) - zp) * s).to(x.dtype)
    @staticmethod
    def backward(ctx, g): return g, None, None, None, None, None


class QuantLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, weight_bit=8, per_channel=True):
        super().__init__(in_f, out_f, bias)
        self.weight_bit = weight_bit; self.per_channel = per_channel
        self._qfn = AsymmetricQuantFunction.apply
    @classmethod
    def from_linear(cls, l, weight_bit=8, per_channel=True):
        inst = cls(l.in_features, l.out_features, l.bias is not None, weight_bit, per_channel)
        inst.weight = nn.Parameter(l.weight.data.clone())
        if l.bias is not None: inst.bias = nn.Parameter(l.bias.data.clone())
        return inst
    def forward(self, x):
        w = self.weight
        if self.per_channel:
            wt = w.data.t().contiguous()
            w_min, w_max = wt.min(dim=1)[0], wt.max(dim=1)[0]
        else:
            w_min, w_max = w.data.min().unsqueeze(0), w.data.max().unsqueeze(0)
        return F.linear(x, self._qfn(w, self.weight_bit, w_min, w_max, self.per_channel, False), self.bias)


def apply_qbert(model, weight_bit=8, per_channel=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, QuantLinear): continue
        parts, parent = name.split("."), model
        for p in parts[:-1]: parent = getattr(parent, p)
        setattr(parent, parts[-1], QuantLinear.from_linear(module, weight_bit, per_channel).to(module.weight.device))
        replaced += 1
    print(f"  Replaced {replaced} Linear -> QuantLinear (INT{weight_bit})")
    return model


def quant_size_mb(model, wb=8):
    return sum(p.numel() * (wb/8 if "weight" in n else 4) for n, p in model.named_parameters()) / 1024**2


# ==============================================================================
# Evaluation harness (generalized across all 9 GLUE tasks)
# ==============================================================================
def load_glue_split(task):
    split = TASK_TO_SPLIT.get(task, "validation")
    return load_dataset("glue", task, split=split)


def encode_batch(tokenizer, batch, key1, key2):
    if key2 is None:
        texts = ([str(t) for t in batch[key1]],)
    else:
        texts = ([str(t) for t in batch[key1]], [str(t) for t in batch[key2]])
    return tokenizer(*texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


def evaluate(model, tokenizer, dataset, task, batch_size=64):
    key1, key2 = TASK_TO_KEYS[task]
    is_regression = task in IS_REGRESSION
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"  Evaluating {task}", leave=False):
            b = dataset.select(range(i, min(i + batch_size, len(dataset))))
            enc = encode_batch(tokenizer, b, key1, key2)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits
            if is_regression:
                all_preds.extend(logits.squeeze(-1).float().cpu().tolist())
            else:
                all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend([float(l) if is_regression else int(l) for l in b["label"]])
    if is_regression:
        pearson = round(pearsonr(all_preds, all_labels)[0] * 100, 4)
        accuracy = "N/A"
        p = r = f1 = "N/A"
    else:
        pearson = "N/A"
        if task == "cola":
            accuracy = round(matthews_corrcoef(all_labels, all_preds) * 100, 4)
        else:
            accuracy = round(100.0 * sum(a == b for a, b in zip(all_preds, all_labels)) / len(all_labels), 4)
        p  = round(precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 4)
        r  = round(recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 4)
        f1 = round(f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100, 4)
    return accuracy, p, r, f1, pearson


def measure_latency(model, tokenizer, dataset, task):
    key1, key2 = TASK_TO_KEYS[task]
    model.eval()
    b = dataset.select(range(0, min(LATENCY_BATCH, len(dataset))))
    enc = encode_batch(tokenizer, b, key1, key2)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        for _ in range(LATENCY_WARMUP): model(**enc)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(LATENCY_RUNS): model(**enc)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000 / (LATENCY_RUNS * LATENCY_BATCH)


def compute_energy(lat_ms):
    return round(GPU_ACTIVE_POWER_W * (lat_ms / 1000.0) * 1000.0, 4)


# ==============================================================================
# Checkpoint / resume
# ==============================================================================
completed_runs = set()
if os.path.exists(MASTER_CSV):
    try:
        existing_df = pd.read_csv(MASTER_CSV)
        for _, r in existing_df.iterrows():
            completed_runs.add((str(r["task"]), str(r["Model name"]), int(r["nBits"])))
        print(f"System Log: Found {len(completed_runs)} completed run(s) in {MASTER_CSV}. These will be skipped.")
    except Exception as _e:
        print(f"Warning: could not read existing {MASTER_CSV} for resume check: {_e}")


def run_one(task, model_name, weight_bit, fp32_cache):
    ckpt_path = CHECKPOINTS[model_name][task]

    if ckpt_path.startswith("/") and not os.path.isdir(ckpt_path):
        print(f"System Log: SKIPPING model={model_name}, task={task} -- local checkpoint not found at "
              f"{ckpt_path}. Run finetune_electra_base.py for this task first.")
        return

    cache_key = (task, model_name)
    if cache_key not in fp32_cache:
        print(f"[LOAD] model={model_name} task={task} <- {ckpt_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        except ValueError as e:
            if "sentencepiece" in str(e) or "tiktoken" in str(e):
                print(f"  Fast tokenizer unavailable ({e}); trying slow tokenizer (use_fast=False)")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
                except ValueError as e2:
                    base_tok = BASE_TOKENIZER_FOR_MODEL.get(model_name)
                    print(f"  Slow tokenizer also unavailable ({e2}); "
                          f"falling back to base model tokenizer: {base_tok}")
                    tokenizer = AutoTokenizer.from_pretrained(base_tok)
            else:
                raise
        try:
            model_fp32 = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        except ValueError as e:
            if "model_type" in str(e) and model_name == "ELECTRA-Base":
                print(f"  Config missing model_type (broken/incomplete upload); forcing ElectraConfig for {ckpt_path}")
                from transformers import ElectraConfig, ElectraForSequenceClassification
                econfig = ElectraConfig.from_pretrained(ckpt_path)
                model_fp32 = ElectraForSequenceClassification.from_pretrained(ckpt_path, config=econfig)
            else:
                raise
        model_fp32.eval().to(DEVICE)
        dataset = load_glue_split(task)
        fp32_cache[cache_key] = dict(tokenizer=tokenizer, model_fp32=model_fp32, dataset=dataset)

    c = fp32_cache[cache_key]
    tokenizer, model_fp32, dataset = c["tokenizer"], c["model_fp32"], c["dataset"]

    print(f"[QBERT] model={model_name} task={task} weight_bit={weight_bit} -- quantizing...")
    model_q = apply_qbert(copy.deepcopy(model_fp32), weight_bit)
    model_q.eval().to(DEVICE)

    accuracy, p, r, f1, pearson = evaluate(model_q, tokenizer, dataset, task)
    size_mb = quant_size_mb(model_q, weight_bit)
    lat_ms = measure_latency(model_q, tokenizer, dataset, task)
    throughput = round(1000.0 / lat_ms, 4) if lat_ms > 0 else "N/A"
    energy = compute_energy(lat_ms)

    row = {
        "task": task,
        "Model name": model_name,
        "Memory Size": round(size_mb, 4),
        "Latency": round(lat_ms, 6),
        "Accuracy": accuracy,
        "Pearson Corr": pearson,
        "Throughput": throughput,
        "nBits": weight_bit,
        "Precision": p,
        "Recall": r,
        "F1 Score": f1,
        "Energy Consumption": energy,
        "checkpoint": ckpt_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    if os.path.exists(MASTER_CSV):
        df.to_csv(MASTER_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(MASTER_CSV, index=False)

    del model_q
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    print(f"System Log: DONE model={model_name}, task={task}, weight_bit={weight_bit} -- accuracy={accuracy}, pearson={pearson}")


# ==============================================================================
# Main loop: task x model x weight_bit, with resilience + resume
# ==============================================================================
def main():
    fp32_cache = {}
    for task in TASKS:
        for model_name in MODELS:
            for weight_bit in WEIGHT_BITS:
                if (task, model_name, weight_bit) in completed_runs:
                    print(f"System Log: SKIPPING (already completed) model={model_name}, task={task}, weight_bit={weight_bit}")
                    continue

                attempt = 0
                while attempt < MAX_RETRIES:
                    attempt += 1
                    try:
                        run_one(task, model_name, weight_bit, fp32_cache)
                        break
                    except Exception as e:
                        print(f"WARNING: run failed for model={model_name}, task={task}, weight_bit={weight_bit} "
                              f"(attempt {attempt}/{MAX_RETRIES}): {e}")
                        traceback.print_exc()
                        if attempt < MAX_RETRIES:
                            print(f"Retrying in {RETRY_WAIT_SECONDS}s...")
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            time.sleep(RETRY_WAIT_SECONDS)
                        else:
                            print(f"WARNING: giving up on model={model_name}, task={task}, weight_bit={weight_bit} "
                                  f"after {MAX_RETRIES} attempts. Skipping to next run.")

        # Free FP32 baselines for this task before moving to the next one
        for key in list(fp32_cache.keys()):
            if key[0] == task:
                del fp32_cache[key]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    print("\nSystem Log: Q-BERT (ELECTRA-base + RoBERTa-base) batch execution complete!")


if __name__ == "__main__":
    main()
