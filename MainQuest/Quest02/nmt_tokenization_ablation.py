"""
Controlled tokenization ablation for Korean->English attention-based Seq2Seq.

Runs the experiment behind the research question:
  "Under what data-scale and vocabulary-granularity conditions does subword (BPE)
   tokenization fail to outperform word-level tokenization in a low-resource
   attention-based Seq2Seq model, and is the failure intrinsic or an optimization
   artifact?"

Design (everything held fixed except the two independent variables):
  IV1  tokenization granularity:  word-level (capped)  vs  BPE @ {1k,2k,4k,8k,16k,32k}
  IV2  training data scale:       {10k, 30k, ALL(~90k)} sentence pairs
  Controls: model size, optimizer, LR, grad-clip, label smoothing, epochs,
            batch size, decoding, and a FIXED train/val/test split.
  Metrics: sacreBLEU, chrF2  (+ training-stability diagnostics)

Recommended platform: Google Colab (GPU runtime). Runtime grows with the grid;
see RUN PLAN at the bottom for a reduced grid if you are time-constrained.

Dependencies:
    pip install torch sentencepiece sacrebleu pandas numpy matplotlib tqdm
"""

import os, re, gc, json, time, random, urllib.request, tarfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import sentencepiece as spm
import sacrebleu

# --------------------------------------------------------------------------- #
# 0. Reproducibility & device
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --------------------------------------------------------------------------- #
# 1. Data acquisition + a SINGLE fixed split (critical fix vs. the pilot)
#    The pilot evaluated on 4 hand-picked sentences -> not a valid test set.
# --------------------------------------------------------------------------- #
DATA_URL = ("https://raw.githubusercontent.com/jungyeul/korean-parallel-corpora/"
            "master/korean-english-news-v1/korean-english-park.train.tar.gz")
TARBALL = "korean-english-park.train.tar.gz"

def download_and_load():
    if not os.path.exists(TARBALL):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, TARBALL)
    with tarfile.open(TARBALL, "r:gz") as tar:
        tar.extractall(filter="data")
    with open("korean-english-park.train.ko", encoding="utf-8") as f:
        ko = [l.strip() for l in f]
    with open("korean-english-park.train.en", encoding="utf-8") as f:
        en = [l.strip() for l in f]
    df = pd.DataFrame({"ko": ko, "en": en}).drop_duplicates().reset_index(drop=True)
    return df

def light_clean(s):
    # Minimal, tokenizer-agnostic normalisation. We deliberately do NOT strip
    # characters differently per tokenizer, so the only thing that varies across
    # conditions is the tokenizer itself.
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def make_splits(df, seed=42, test_frac=0.05, val_frac=0.05):
    df = df.copy()
    df["ko"] = df["ko"].map(light_clean)
    df["en"] = df["en"].map(light_clean)
    df = df[(df.ko.str.len() > 0) & (df.en.str.len() > 0)].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    n_val  = int(len(df) * val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))

# --------------------------------------------------------------------------- #
# 2. Tokenizers. Both expose the same interface: encode(str)->ids, decode(ids)->str
#    Special ids are identical across tokenizers: pad=0, unk=1, bos=2, eos=3.
# --------------------------------------------------------------------------- #
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]

class WordTokenizer:
    """Whitespace word-level tokenizer with a HARD vocab cap (matched to BPE)."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.w2i = {t: i for i, t in enumerate(SPECIALS)}
        self.i2w = {i: t for t, i in self.w2i.items()}

    def fit(self, sentences):
        counts = Counter(w for s in sentences for w in s.split())
        for w, _ in counts.most_common(self.vocab_size - len(SPECIALS)):
            self.w2i[w] = len(self.w2i)
        self.i2w = {i: w for w, i in self.w2i.items()}
        return self

    def encode(self, s):
        return [self.w2i.get(w, UNK) for w in s.split()]

    def decode(self, ids):
        toks = [self.i2w.get(i, "<unk>") for i in ids
                if i not in (PAD, UNK, BOS, EOS)]
        return " ".join(toks)

    def size(self):
        return len(self.w2i)

class SPMTokenizer:
    """SentencePiece BPE tokenizer. Same special-id layout as WordTokenizer."""
    def __init__(self, vocab_size, prefix):
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.sp = None

    def fit(self, sentences):
        tmp = f"{self.prefix}_corpus.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
        spm.SentencePieceTrainer.Train(
            f"--input={tmp} --model_prefix={self.prefix} "
            f"--vocab_size={self.vocab_size} --model_type=bpe "
            f"--pad_id={PAD} --unk_id={UNK} --bos_id={BOS} --eos_id={EOS} "
            f"--character_coverage=0.9995"
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{self.prefix}.model")
        return self

    def encode(self, s):
        return self.sp.EncodeAsIds(s)

    def decode(self, ids):
        ids = [i for i in ids if i not in (PAD, BOS, EOS)]
        return self.sp.DecodeIds(ids)

    def size(self):
        return self.sp.GetPieceSize()

# --------------------------------------------------------------------------- #
# 3. Tensorisation (source = ...EOS, target = BOS...EOS), with a fixed cap.
# --------------------------------------------------------------------------- #
MAX_LEN = 60   # generous; chosen once and held constant across all conditions

def to_tensor(sentences, tok, add_bos=False):
    rows = []
    for s in sentences:
        ids = tok.encode(s)[: MAX_LEN - 2]
        if add_bos:
            ids = [BOS] + ids + [EOS]
        else:
            ids = ids + [EOS]
        rows.append(ids + [PAD] * (MAX_LEN - len(ids)))
    return torch.LongTensor(rows)

def build_loaders(train_df, val_df, test_df, src_tok, tgt_tok, batch_size=128):
    # Fit tokenizers on TRAIN only (no leakage).
    src_tok.fit(train_df.ko.tolist())
    tgt_tok.fit(train_df.en.tolist())

    def pack(df):
        src = to_tensor(df.ko.tolist(), src_tok, add_bos=False)
        tgt = to_tensor(df.en.tolist(), tgt_tok, add_bos=True)
        return TensorDataset(src, tgt)

    train_loader = DataLoader(pack(train_df), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(pack(val_df),   batch_size=batch_size)
    return train_loader, val_loader, test_df

# --------------------------------------------------------------------------- #
# 4. Model: GRU encoder + Bahdanau attention + GRU decoder (matches the pilot)
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)
    def forward(self, x):
        return self.gru(self.embedding(x))   # outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.W = nn.Linear(hid, hid); self.U = nn.Linear(hid, hid); self.v = nn.Linear(hid, 1)
    def forward(self, query, values, mask=None):
        # query: [B,hid]  values: [B,S,hid]
        score = self.v(torch.tanh(self.W(query.unsqueeze(1)) + self.U(values)))  # [B,S,1]
        if mask is not None:
            score = score.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn = torch.softmax(score, dim=1)
        ctx = torch.sum(attn * values, dim=1)
        return ctx, attn.squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb + hid, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab)
        self.attention = BahdanauAttention(hid)
    def forward(self, x, hidden, enc_out, src_mask=None):
        ctx, attn = self.attention(hidden[-1], enc_out, src_mask)
        emb = self.embedding(x).unsqueeze(1)
        out, hidden = self.gru(torch.cat([ctx.unsqueeze(1), emb], dim=-1), hidden)
        return self.fc(out.squeeze(1)), hidden, attn

def init_weights(m):
    for name, p in m.named_parameters():
        if "weight" in name and p.dim() >= 2:
            init.xavier_uniform_(p)
        elif "bias" in name:
            init.constant_(p, 0.0)

# --------------------------------------------------------------------------- #
# 5. Train / evaluate
# --------------------------------------------------------------------------- #
EMB, HID = 256, 512
EPOCHS = 30
LR = 5e-4
CLIP = 1.0
TF_RATIO = 0.5            # fixed teacher-forcing ratio across conditions
LABEL_SMOOTH = 0.1

def greedy_decode(enc, dec, src, tgt_tok, max_len=MAX_LEN):
    enc.eval(); dec.eval()
    src_mask = (src != PAD)
    with torch.no_grad():
        enc_out, hid = enc(src)
        bsz = src.size(0)
        dec_in = torch.full((bsz,), BOS, dtype=torch.long, device=src.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=src.device)
        outputs = [[] for _ in range(bsz)]
        for _ in range(max_len):
            logits, hid, _ = dec(dec_in, hid, enc_out, src_mask)
            pred = logits.argmax(-1)
            for b in range(bsz):
                if not finished[b]:
                    if pred[b].item() == EOS:
                        finished[b] = True
                    else:
                        outputs[b].append(pred[b].item())
            dec_in = pred
            if finished.all():
                break
    return [tgt_tok.decode(o) for o in outputs]

def evaluate_corpus(enc, dec, df, src_tok, tgt_tok, batch_size=128):
    src = to_tensor(df.ko.tolist(), src_tok, add_bos=False)
    hyps, refs = [], df.en.tolist()
    for i in range(0, len(src), batch_size):
        batch = src[i:i + batch_size].to(DEVICE)
        hyps.extend(greedy_decode(enc, dec, batch, tgt_tok))
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return bleu, chrf, hyps

def train_one_config(cfg, train_df, val_df, test_df, seed):
    set_seed(seed)
    name = cfg["name"]
    # --- tokenizers ---
    if cfg["tok"] == "word":
        src_tok = WordTokenizer(cfg["vocab"])
        tgt_tok = WordTokenizer(cfg["vocab"])
    else:
        src_tok = SPMTokenizer(cfg["vocab"], f"spm_ko_{cfg['vocab']}_{seed}")
        tgt_tok = SPMTokenizer(cfg["vocab"], f"spm_en_{cfg['vocab']}_{seed}")

    train_loader, val_loader, _ = build_loaders(
        train_df, val_df, test_df, src_tok, tgt_tok)

    enc = Encoder(src_tok.size(), EMB, HID).to(DEVICE); enc.apply(init_weights)
    dec = Decoder(tgt_tok.size(), EMB, HID).to(DEVICE); dec.apply(init_weights)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = optim.AdamW(params, lr=LR, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=LABEL_SMOOTH)

    # --- training-stability diagnostics (part of the analysis) ---
    diag = {"nan_batches": 0, "epochs_to_best": -1, "max_grad_norm": 0.0}
    best_val = float("inf"); best_state = None

    for epoch in range(EPOCHS):
        enc.train(); dec.train(); running = 0.0; nb = 0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_mask = (src != PAD)
            opt.zero_grad()
            enc_out, hid = enc(src)
            dec_in = tgt[:, 0]
            step_logits = []
            for t in range(1, tgt.size(1)):
                logits, hid, _ = dec(dec_in, hid, enc_out, src_mask)
                step_logits.append(logits)
                dec_in = tgt[:, t] if random.random() < TF_RATIO else logits.argmax(-1)
            # Compute the loss ONCE over all positions. ignore_index handles the
            # pad tokens; the denominator is the global non-pad count (>0), so the
            # all-pad-column 0/0 NaN that label_smoothing triggers cannot occur.
            out = torch.stack(step_logits, dim=1)          # [B, T-1, V]
            gold = tgt[:, 1:]                              # [B, T-1]
            loss = crit(out.reshape(-1, out.size(-1)), gold.reshape(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                diag["nan_batches"] += 1
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, CLIP).item()
            diag["max_grad_norm"] = max(diag["max_grad_norm"], gnorm)
            opt.step()
            running += loss.item(); nb += 1

        # ---- validation loss (teacher-forced) for model selection ----
        enc.eval(); dec.eval(); vloss = 0.0; vb = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                src_mask = (src != PAD)
                enc_out, hid = enc(src); dec_in = tgt[:, 0]
                step_logits = []
                for t in range(1, tgt.size(1)):
                    logits, hid, _ = dec(dec_in, hid, enc_out, src_mask)
                    step_logits.append(logits); dec_in = tgt[:, t]
                out = torch.stack(step_logits, dim=1)
                gold = tgt[:, 1:]
                vl = crit(out.reshape(-1, out.size(-1)), gold.reshape(-1))
                if torch.isnan(vl) or torch.isinf(vl):
                    continue                              # never poison the running mean
                vloss += vl.item(); vb += 1
        vloss /= max(vb, 1); sched.step(vloss)
        if vloss < best_val:
            best_val = vloss; diag["epochs_to_best"] = epoch + 1
            best_state = ({k: v.cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.cpu().clone() for k, v in dec.state_dict().items()})
        print(f"[{name} seed{seed}] epoch {epoch+1}/{EPOCHS} "
              f"train {running/max(nb,1):.3f} val {vloss:.3f}")

    if best_state is not None:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.to(DEVICE); dec.to(DEVICE)

    bleu, chrf, hyps = evaluate_corpus(enc, dec, test_df, src_tok, tgt_tok)
    result = {**cfg, "seed": seed, "test_bleu": bleu, "test_chrf": chrf,
              "best_val_loss": best_val, "src_vocab": src_tok.size(),
              "tgt_vocab": tgt_tok.size(), **diag}
    del enc, dec, opt; gc.collect(); torch.cuda.empty_cache()
    return result, hyps

# --------------------------------------------------------------------------- #
# 6. The grid + driver
# --------------------------------------------------------------------------- #
def build_grid():
    cfgs = []
    # Word-level reference (capped to keep vocab comparable to mid-range BPE)
    cfgs.append({"name": "word-16k", "tok": "word", "vocab": 16000})
    # BPE granularity sweep
    for v in [1000, 2000, 4000, 8000, 16000, 32000]:
        cfgs.append({"name": f"bpe-{v}", "tok": "bpe", "vocab": v})
    return cfgs

def run_all(data_scales=(10000, 30000, None), seeds=(13, 42, 123),
            out_csv="results.csv"):
    df_all = download_and_load()
    train_full, val_df, test_df = make_splits(df_all)   # split ONCE, reuse
    print(f"Split sizes  train={len(train_full)}  val={len(val_df)}  test={len(test_df)}")

    rows = []
    for scale in data_scales:
        train_df = train_full if scale is None else train_full.iloc[:scale].reset_index(drop=True)
        scale_tag = "ALL" if scale is None else str(scale)
        for cfg in build_grid():
            for seed in seeds:
                tagged = {**cfg, "data_scale": scale_tag,
                          "name": f"{cfg['name']}@{scale_tag}"}
                t0 = time.time()
                res, _ = train_one_config(tagged, train_df, val_df, test_df, seed)
                res["seconds"] = round(time.time() - t0, 1)
                rows.append(res)
                pd.DataFrame(rows).to_csv(out_csv, index=False)  # checkpoint each run
                print(f"DONE {tagged['name']} seed{seed}: "
                      f"BLEU {res['test_bleu']:.2f} chrF {res['test_chrf']:.2f}")
    print("Saved ->", out_csv)
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# 7. Aggregation helper (mean +/- std across seeds) -> ready for the LaTeX table
# --------------------------------------------------------------------------- #
def summarise(csv="results.csv"):
    df = pd.read_csv(csv)
    g = (df.groupby(["data_scale", "tok", "vocab"])
           [["test_bleu", "test_chrf", "best_val_loss",
             "nan_batches", "epochs_to_best", "max_grad_norm"]]
           .agg(["mean", "std"]).round(2))
    print(g)
    g.to_csv("results_summary.csv")
    return g

# =========================================================================== #
# RUN PLAN
#   FULL grid  = 7 configs x 3 scales x 3 seeds = 63 runs (publishable variance).
#   If time-limited, start with 1 seed and the two endpoints of each axis:
#       run_all(data_scales=(10000, None), seeds=(42,))
#   then add seeds/scales as time allows. Each ~90k/30-epoch run is the slow one;
#   on a T4 GPU expect roughly 10-25 min/run depending on vocab size.
# =========================================================================== #
if __name__ == "__main__":
    df = run_all()          # full grid; edit args to reduce
    summarise()
