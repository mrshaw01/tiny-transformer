# Training walkthrough (math) — Qwen3

This note explains:

1. What `python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000` does.
2. How `tiny_transformer/train.py` consumes the resulting files.
3. The forward pass math (how logits are computed), loss computation, and backward pass intuition.
4. An end-to-end view of **one** training sample.

This document is for the **Qwen3** model implementation:

- Model class: `Qwen3ForCausalLM`
- Code: `tiny_transformer/models/qwen3/modeling_qwen3.py`

---

## 0) Notation and shapes

- Vocabulary size: \(V\) (Qwen tokenizer, ~151k)
- Sequence length: \(T\) (here `--seq_len 512`)
- Batch size: \(B\) (here `--micro_batch_size` in training, default 32)
- Hidden size: \(d\) (from `configs/qwen3_demo.json`)
- Layers: \(L\) (from config)
- Attention heads: \(n*h\), head dim: \(d_h\), KV heads: \(n*{kv}\) (GQA)

One training sequence is token ids:
\[
x = (x_1, x_2, \dots, x_T), \quad x_t \in \{0,1,\dots,V-1\}.
\]

For a batch:
\[
X \in \mathbb{Z}^{B \times T}.
\]

---

## 1) What `prepare_dataset` does

Command:

```bash
python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000
```

Implementation: `tiny_transformer/prepare_dataset.py`

### 1.1 Loads tokenizer

It uses the official Qwen tokenizer:

- `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")`

It requires:

- `eos_token_id` exists (used as a document boundary token)
- `pad_token_id` exists
- if `bos_token_id` is missing, it sets `bos_token = pad_token`

### 1.2 Streams Wikipedia and extracts “intro” paragraphs

It streams:

- `load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)`

For each article:

- take the first paragraph (“intro”)
- normalize whitespace
- skip very short intros

### 1.3 Stops after ~100MB of raw UTF-8 text

It tracks:
\[
\text{consumed_bytes} \leftarrow \text{consumed_bytes} + \lvert \text{intro.encode("utf-8")} \rvert
\]
and stops when it would exceed `--max_bytes`.

### 1.4 Deterministic train/val split

It assigns each article to validation if:
\[
\text{stable_float01(title)} < \text{val_ratio}
\]
where `stable_float01` hashes the title (SHA1) into a repeatable float in \([0,1)\).

### 1.5 Batch tokenization + EOS append

It tokenizes intros in batches (256 texts at a time), without adding special tokens:

- `add_special_tokens=False`

Then appends EOS after each intro:
\[
\text{ids} \leftarrow \text{ids} \,\|\, [\text{eos}]
\]

### 1.6 Packs tokens into fixed-length sequences

It maintains two growing token buffers (`train_buffer`, `val_buffer`) and repeatedly writes full chunks of length \(T\).

Packing means:

- \(n = \left\lfloor \frac{|buffer|}{T} \right\rfloor\)
- write out the first \(n \cdot T\) tokens
- keep the remainder in the buffer

So `train.bin` is a flat array of token ids, interpreted as contiguous sequences of length \(T\).

### 1.7 Output files

In `--out_dir data` it produces:

- `data/train.bin` (packed uint32 token ids)
- `data/val.bin` (packed uint32 token ids)
- `data/meta.json` (format + tokenizer info + counts)

---

## 2) How training consumes the packed data

Training entrypoint: `tiny_transformer/train.py`
Dataset code: `tiny_transformer/data/packed_dataset.py`

### 2.1 Loads meta + memmaps the `.bin`

`PackedMemmapDataset` uses:

- `np.memmap(train.bin, dtype=uint32, mode="r")`

No full read into RAM: the OS pages data in as needed.

### 2.2 One dataset item = one packed sequence

For index \(i\):
\[
x^{(i)} = \text{train.bin}[iT : (i+1)T]
\]

The dataset returns:

- `input_ids = x^{(i)}`
- `labels = x^{(i)}`

This is standard for causal LM training: the model computes next-token loss by shifting internally.

---

## 3) Forward pass: tokens → hidden states → logits

### 3.1 Token embedding

Let:
\[
E \in \mathbb{R}^{V \times d}
\]
be the embedding matrix.

Embedding lookup:
\[
h^{(0)}_{b,t} = E[x_{b,t}] \in \mathbb{R}^{d}
\]
so:
\[
H^{(0)} \in \mathbb{R}^{B \times T \times d}.
\]

### 3.2 Transformer layers (repeat \(L\) times)

Each layer is pre-norm with:

- RMSNorm
- causal self-attention (RoPE + GQA)
- RMSNorm
- SwiGLU MLP
- residual connections

#### RMSNorm

For \(u \in \mathbb{R}^{d}\):
\[
\text{rms}(u) = \sqrt{\frac{1}{d}\sum\_{i=1}^{d} u_i^2 + \varepsilon}
\]
\[
\text{RMSNorm}(u) = \gamma \odot \frac{u}{\text{rms}(u)}
\]
with learned scale \(\gamma \in \mathbb{R}^{d}\).

#### Causal self-attention (one layer)

Projection:
\[
Q = HW*Q,\quad K = HW_K,\quad V = HW_V
\]
then reshape to heads:
\[
Q \in \mathbb{R}^{B \times n_h \times T \times d_h},\quad
K,V \in \mathbb{R}^{B \times n*{kv} \times T \times d_h}.
\]

RoPE rotates \(Q,K\) by position-dependent angles:
\[
Q'=\text{RoPE}(Q),\quad K'=\text{RoPE}(K)
\]

Scores:
\[
S*{b,i,t,s} = \frac{\langle Q'*{b,i,t,:}, K'_{b,i,s,:}\rangle}{\sqrt{d_h}}
\]
Causal mask:
\[
S_{b,i,t,s} = -\infty \;\;\text{if } s>t
\]
Softmax:
\[
P*{b,i,t,s} = \text{softmax}(S*{b,i,t,:})_s
\]
Weighted sum:
\[
O_{b,i,t,:} = \sum*{s \le t} P*{b,i,t,s} V*{b,i,s,:}
\]
Concatenate heads and project:
\[
A*{b,t,:} = \text{Concat}_i(O_{b,i,t,:})W_O
\]
Residual:
\[
H \leftarrow H + A
\]

#### SwiGLU MLP

Let:
\[
G = HW_g,\quad U = HW_u
\]
Then:
\[
\text{MLP}(H) = (\text{SiLU}(G)\odot U)W_d
\]
Residual:
\[
H \leftarrow H + \text{MLP}(H)
\]

After \(L\) layers:
\[
H^{(L)} \in \mathbb{R}^{B \times T \times d}.
\]

### 3.3 LM head → logits

LM head:
\[
Z = H^{(L)}W*{\text{lm}} + b,\quad
W*{\text{lm}} \in \mathbb{R}^{d \times V}
\]
so:
\[
Z \in \mathbb{R}^{B \times T \times V}.
\]

Each \(Z\_{b,t,:}\) is the logits vector used to predict the next token.

---

## 4) Loss: next-token cross-entropy

Targets are shifted:
\[
y*{b,t} = x*{b,t+1}\quad\text{for } t=1..T-1
\]

Per-position loss:
\[
\ell*{b,t} = -\log \left(\frac{\exp(Z*{b,t,y*{b,t}})}{\sum*{v=1}^{V}\exp(Z\_{b,t,v})}\right)
\]

Average loss:
\[
\mathcal{L} = \frac{1}{B(T-1)}\sum*{b=1}^{B}\sum*{t=1}^{T-1}\ell\_{b,t}.
\]

---

## 5) Backward pass (key gradient)

Let:
\[
p*{b,t,v} = \text{softmax}(Z*{b,t,:})\_v
\]

Then:
\[
\frac{\partial \ell*{b,t}}{\partial Z*{b,t,v}} = p*{b,t,v} - \mathbf{1}[v=y*{b,t}]
\]

This gradient flows back through the LM head, transformer layers, and embeddings. The optimizer (AdamW) updates parameters each step.

---

## 6) One end-to-end training sample

Pick dataset index \(i\):

1. Load tokens:
   \[
   x = \text{train.bin}[iT : (i+1)T]
   \]
2. Forward → logits \(Z\)
3. Shifted cross-entropy loss \(\mathcal{L}\)
4. Backprop \(\nabla\_\theta \mathcal{L}\)
5. AdamW step updates \(\theta\)
