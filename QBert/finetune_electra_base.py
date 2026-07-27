TARGET_TASKS = ["mnli", "mrpc", "qnli", "qqp", "sst2", "stsb"]

MODEL_ID = "google/electra-base-discriminator"
MODEL_DISPLAY_NAME = "ELECTRA-Base"

BASE_DIR = "/media/shared/Devshree/edgellm3/glue_pruning"
CHECKPOINT_DIR = f"{BASE_DIR}/electra_finetuned_checkpoints"

SCRIPT_VERSION = "v1-electra-finetune-2026-07-24"

import os, torch, json, shutil, subprocess, time
import pandas as pd
from codecarbon import OfflineEmissionsTracker

print(f"System Log: Running script version: {SCRIPT_VERSION}")

torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(f"{BASE_DIR}/Model-Compression-Research-Package")

original_script = "examples/transformers/text-classification/run_glue.py"
patched_script = "run_glue_patched_electra_finetune.py"
with open(original_script, "r") as f: script_code = f.read()

signature_patch = """
import inspect
_orig_sig_init = inspect.Signature.__init__
def _patched_sig_init(self, parameters=None, *args, **kwargs):
    if parameters is not None:
        try: parameters = sorted(list(parameters), key=lambda p: p.kind)
        except Exception: pass
    return _orig_sig_init(self, parameters, *args, **kwargs)
inspect.Signature.__init__ = _patched_sig_init
"""
script_code = signature_patch + "\n" + script_code

pruning_callback_patch = """
try:
    from model_compression_research.api_utils import HFTrainerPruningCallback
    if not hasattr(HFTrainerPruningCallback, '_keys_to_ignore_on_save'):
        HFTrainerPruningCallback._keys_to_ignore_on_save = set()
except Exception as _e:
    print("Warning: could not patch HFTrainerPruningCallback:", _e)
"""
script_code = pruning_callback_patch + "\n" + script_code

patched_compute = """result = metric.compute(predictions=preds, references=p.label_ids)
        if not is_regression:
            from sklearn.metrics import precision_score, recall_score, f1_score
            result["precision"] = precision_score(p.label_ids, preds, average="macro", zero_division=0)
            result["recall"] = recall_score(p.label_ids, preds, average="macro", zero_division=0)
            if "f1" not in result: result["f1"] = f1_score(p.label_ids, preds, average="macro", zero_division=0)"""
script_code = script_code.replace("result = metric.compute(predictions=preds, references=p.label_ids)", patched_compute)
script_code = script_code.replace("from datasets import load_dataset, load_metric", "from datasets import load_dataset\nfrom evaluate import load as load_metric")
script_code = script_code.replace("use_auth_token=True if model_args.use_auth_token else None", "token=True if model_args.use_auth_token else None")
script_code = script_code.replace("tokenizer=tokenizer,", "processing_class=tokenizer,")
script_code = script_code.replace(
    "if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:",
    "if os.path.isdir(training_args.output_dir) and training_args.do_train and not getattr(training_args, 'overwrite_output_dir', True):"
)

with open(patched_script, "w") as f: f.write(script_code)
print("System Log: Script patched successfully.")

# ==============================================================================
# Task hyperparameters (epochs matched to earlier pipeline's conventions)
# ==============================================================================
all_tasks = {
    "mnli": {"epochs": 3.0},
    "mrpc": {"epochs": 5.0},
    "qnli": {"epochs": 5.0},
    "qqp":  {"epochs": 3.0},
    "sst2": {"epochs": 5.0},
    "stsb": {"epochs": 5.0},
}

MAX_RETRIES = 2
RETRY_WAIT_SECONDS = 30

# ==============================================================================
# Checkpoint / resume: skip tasks already fine-tuned and saved
# ==============================================================================
def already_finetuned(task):
    out_dir = f"{CHECKPOINT_DIR}/{task}"
    has_safetensors = os.path.exists(os.path.join(out_dir, "model.safetensors"))
    has_bin = os.path.exists(os.path.join(out_dir, "pytorch_model.bin"))
    return has_safetensors or has_bin

# ==============================================================================
# Fine-tuning loop
# ==============================================================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

for task in TARGET_TASKS:
    if task not in all_tasks:
        continue

    if already_finetuned(task):
        print(f"System Log: SKIPPING (already fine-tuned) task={task}")
        continue

    cfg = all_tasks[task]
    OUTPUT_DIR = f"{CHECKPOINT_DIR}/{task}"

    print(f"\n{'='*60}\nFINE-TUNING {MODEL_DISPLAY_NAME} | TASK: {task.upper()}\n{'='*60}")

    tracker = OfflineEmissionsTracker(country_iso_code="IND")
    tracker.start()

    cmd = [
        "python", patched_script,
        "--model_name_or_path", MODEL_ID,
        "--task_name", task,
        "--do_train",
        "--do_eval",
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "8",  # smaller batch for ELECTRA-base under GPU contention
        "--learning_rate", "2e-5",
        "--num_train_epochs", str(cfg['epochs']),
        "--output_dir", OUTPUT_DIR,
        "--save_strategy", "no",  # we save explicitly via trainer.save_model() in run_glue.py's own do_train block
    ]

    attempt = 0
    result = None
    while attempt < MAX_RETRIES:
        attempt += 1
        result = subprocess.run(cmd)
        if result.returncode == 0:
            break
        print(f"WARNING: fine-tuning failed for task={task} (exit code {result.returncode}), "
              f"attempt {attempt}/{MAX_RETRIES}.")
        if attempt < MAX_RETRIES:
            print(f"Retrying in {RETRY_WAIT_SECONDS}s...")
            torch.cuda.empty_cache()
            time.sleep(RETRY_WAIT_SECONDS)

    energy_kwh = tracker.stop()

    if result.returncode != 0:
        print(f"WARNING: giving up on task={task} after {MAX_RETRIES} attempts. Skipping to next task.")
        continue

    if already_finetuned(task):
        print(f"System Log: DONE fine-tuning {MODEL_DISPLAY_NAME} on {task}. "
              f"Checkpoint saved at {OUTPUT_DIR} (energy: {energy_kwh:.6f} kWh)")
    else:
        print(f"WARNING: training exited cleanly for task={task} but no checkpoint file found at {OUTPUT_DIR}. "
              f"Check the run's eval_results.json / logs for details.")

print("\nSystem Log: ELECTRA-Base fine-tuning batch complete!")
print(f"System Log: Checkpoints saved under: {CHECKPOINT_DIR}/<task>/")
