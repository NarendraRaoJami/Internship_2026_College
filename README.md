# SparseGPT Pruning Experiment

Comparing three pruning methods on `facebook/opt-125m` using the `wikitext2` dataset.

---

## What This Does

Prunes a language model at 6 sparsity levels (20% to 70%) using three methods and records:

- Perplexity
- Memory Size
- Latency
- Pruning Time
- Energy Consumption
- Throughput

---

## Pruning Methods

| Method | How it works |
|---|---|
| **SparseGPT** | Removes weights using Hessian-based scoring (smartest) |
| **Magnitude** | Removes the smallest weight values (simplest) |
| **Movement** | Removes weights with the lowest gradient × weight score |

---

## Sparsity Levels

20% → 30% → 40% → 50% → 60% → 70%

---

## How to Run

**1. Open in Google Colab**

Upload `SparseGPT_Simple.ipynb` to [colab.research.google.com](https://colab.research.google.com)

**2. Set runtime to GPU**

`Runtime → Change runtime type → T4 GPU`

**3. Run cells in order**

- Cell 1 — Install dependencies
- Cell 2 — Clone SparseGPT repo
- Cell 3 — Imports and setup
- Cell 4 — Inference helper function
- Cell 5 — Run SparseGPT
- Cell 6 — Run Magnitude pruning
- Cell 7 — Run Movement pruning
- Cell 8 — Final summary table

---

## Requirements

- Google Colab (free tier works for opt-125m)
- T4 GPU (~2 GB VRAM minimum)
- Internet connection (to download model and dataset)

No local setup needed — everything runs inside Colab.

---

## Model & Dataset

| | |
|---|---|
| Model | `facebook/opt-125m` (251 MB) |
| Dataset | `wikitext2` |
| Task | Causal language modeling |

---

## Expected Results (50% Sparsity)

| Method | Perplexity (wikitext2) |
|---|---|
| SparseGPT | ~36.9 |
| Magnitude | higher |
| Movement | higher |

SparseGPT gives the best perplexity because it uses second-order information to decide which weights to remove.

---

## Files

```
SparseGPT_Simple.ipynb   — main notebook
README.md                — this file
```

---

## Reference

Frantar & Alistarh (2023). *SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot.* ICML 2023.
https://arxiv.org/abs/2301.00774
