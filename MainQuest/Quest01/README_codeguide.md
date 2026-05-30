# Tokenization Granularity in Low-Resource Korean–English NMT

Code to reproduce the experiments in:

> **When Subwords Hurt: A Controlled Study of Tokenization Granularity and
> Training Stability in Low-Resource Korean–English Attention-Based Seq2Seq
> Translation.** Semi Song, 2026.

A controlled ablation comparing word-level vs. BPE tokenization (vocab sizes
2k–8k) across two data scales (10k vs. full ~71k pairs) for a GRU + Bahdanau-
attention sequence-to-sequence model. The headline result: once confounds are
controlled, BPE outperforms word-level tokenization at every data scale, and
BPE vocabulary size has only a minor effect.

## Contents

| File | Description |
|------|-------------|
| `nmt_tokenization_ablation.py` | Full pipeline: data download, fixed split, tokenizers, model, training, evaluation |
| `results.csv` | Measured results (BLEU, chrF2, stability diagnostics) reported in the paper |

## Requirements

```bash
pip install torch sentencepiece sacrebleu pandas numpy matplotlib tqdm
```

A CUDA GPU is recommended (experiments were run on a single NVIDIA T4).

## Reproducing the experiments

The script downloads the Korean–English parallel news corpus automatically and
builds a fixed train/validation/test split (seed 42), so no manual data setup
is needed.

```python
import nmt_tokenization_ablation as nta

# Reduced grid: word + BPE{2k,4k,8k} x {10k, full}, single seed
df = nta.run_all(data_scales=(10000, None), seeds=(42,))
nta.summarise()
```

Results are checkpointed to `results.csv` after every run, so the process can
be resumed if interrupted. `summarise()` writes `results_summary.csv` with
per-configuration means.

For the full grid (multiple seeds for variance estimation):

```python
df = nta.run_all(data_scales=(10000, 30000, None), seeds=(13, 42, 123))
```

## Key settings (held constant across all conditions)

- Model: single-layer GRU encoder/decoder, Bahdanau attention, emb 256 / hidden 512
- Optimizer: AdamW, lr 5e-4, weight decay 1e-5, gradient clip 1.0
- Loss: cross-entropy over non-pad positions, label smoothing 0.1
- Teacher forcing: 0.9 (lower ratios caused degenerate repetition under greedy decoding)
- 30 epochs, batch size 128, max sequence length 60
- Metrics: sacreBLEU, chrF2 (greedy decoding)

## Notes

- BPE vocabulary must exceed the corpus character floor (~1.3k for Korean), so
  1k BPE is not trainable; the sweep starts at 2k.
- The full-scale BPE-8k cell was not completed in the reported run due to an
  environment interruption; it does not affect the conclusions.
