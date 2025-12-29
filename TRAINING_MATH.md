# End-to-end training walkthrough

This note explains:

1. What `python -m tiny_transformer.prepare_dataset --out_dir data --seq_len 512 --max_bytes 100000000` does.
2. How `tiny_transformer/train.py` consumes the resulting files.
3. The forward pass math (how logits are computed), loss computation, and backward pass intuition.
4. An end-to-end view of **one** training sample.

The repo trains a **decoder-only causal language model**: `Qwen3ForCausalLM` (next-token prediction).

---

## 0) Notation and shapes

- Vocabulary size: \(V\) (Qwen tokenizer, ~151k)
- Sequence length: \(T\) (here `--seq_len 512`)
- Batch size: \(B\) (here `--micro_batch_size` in training, default 32)
- Hidden size (model width): \(d\) (from `configs/qwen3_demo.json`)
- Layers: \(L\) (from config)
- Heads: \(n*h\), head dim \(d_h\), KV heads \(n*{kv}\) (GQA)

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
- if `bos_token_id` is missing, it sets `bos_token = pad_token` (so BOS is defined)

### 1.2 Streams Wikipedia and extracts “intro” paragraphs

It streams:

- `load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)`

For each article:

- takes the first paragraph (“intro”) via `wikipedia_intro()`
- normalizes whitespace
- skips very short intros

### 1.3 Stops after ~100MB of raw UTF-8 text

It tracks:

\[
\text{consumed_bytes} \leftarrow \text{consumed_bytes} + \lvert \text{intro.encode("utf-8")} \rvert
\]

and stops when it would exceed `--max_bytes` (100,000,000 bytes).

### 1.4 Deterministic train/val split

It assigns each article to validation if:

\[
\text{stable_float01(title)} < \text{val_ratio}
\]

where `stable_float01` hashes the title (SHA1) into a repeatable float in \([0,1)\). Default `val_ratio=0.01`.

### 1.5 Batch tokenization + EOS append

It tokenizes intros in batches (256 texts at a time), without adding special tokens:

- `add_special_tokens=False`

Then appends EOS after each intro:
\[
\text{ids} \leftarrow \text{ids} \,\|\, [\text{eos}]
\]

This teaches the model “document ends here”.

### 1.6 Packs tokens into fixed-length sequences (512)

It maintains two growing token buffers:

- `train_buffer`: long list of token ids
- `val_buffer`: long list of token ids

Packing means: cut the buffer into as many full chunks of length \(T\) as possible:

- \(n = \left\lfloor \frac{|buffer|}{T} \right\rfloor\)
- write out the first \(n \cdot T\) tokens
- keep the remainder in the buffer

So `train.bin` is effectively a flat array of token ids, interpreted as contiguous sequences of length \(T\).

### 1.7 Writes outputs

In `--out_dir data` it produces:

- `data/train.bin` (packed uint32 token ids)
- `data/val.bin` (packed uint32 token ids)
- `data/meta.json` (format + tokenizer info + counts)

`meta.json` includes:

- `seq_len`, `dtype` (`uint32`)
- `vocab_size` (`len(tokenizer)`)
- `tokenizer_name`
- `eos_token_id`, `bos_token_id`
- counts: sequences/tokens, `consumed_bytes`, etc.

---

## 2) How training consumes this data

Training entrypoint: `tiny_transformer/train.py`
Dataset code: `tiny_transformer/data/packed_dataset.py`

### 2.1 Loads meta + memmaps the `.bin`

`PackedDatasetMeta.load(data/meta.json)` provides `seq_len`, `dtype`, etc.

`PackedMemmapDataset(data/train.bin, meta)` does:

- `np.memmap(train.bin, dtype=uint32, mode="r")`

No full read into RAM: the OS pages data in as needed.

### 2.2 One dataset item = one packed sequence

For index \(i\), it slices:
\[
x^{(i)} = \text{train.bin}[iT : (i+1)T]
\]

Then returns:

- `input_ids = x^{(i)}` (as torch int64 tensor)
- `labels = x^{(i)}` (same tensor)

That “labels=input_ids” is standard in causal LM training: loss is computed by shifting internally (next-token prediction).

### 2.3 Batching

The collator stacks tensors into:
\[
X \in \mathbb{Z}^{B \times T}, \quad \text{labels} \in \mathbb{Z}^{B \times T}.
\]

---

## 3) Forward pass: tokens → hidden states → logits

Model class: `Qwen3ForCausalLM` in `tiny_transformer/models/qwen3/modeling_qwen3.py`.

### 3.1 Token embedding

Let the embedding matrix be:
\[
E \in \mathbb{R}^{V \times d}.
\]

Embeddings:
\[
h^{(0)}_{b,t} = E[x_{b,t}] \in \mathbb{R}^{d}.
\]

Collect into:
\[
H^{(0)} \in \mathbb{R}^{B \times T \times d}.
\]

### 3.2 Transformer layer (repeated \(L\) times)

Each layer is pre-norm with:

- RMSNorm
- causal self-attention (with RoPE)
- RMSNorm
- SwiGLU MLP
- residual connections

#### RMSNorm

For a vector \(u \in \mathbb{R}^{d}\):
\[
\text{rms}(u) = \sqrt{\frac{1}{d}\sum\_{i=1}^{d} u_i^2 + \varepsilon}
\]
\[
\text{RMSNorm}(u) = \gamma \odot \frac{u}{\text{rms}(u)}
\]
with learned scale \(\gamma \in \mathbb{R}^{d}\).

#### Self-attention (causal, with RoPE)

For each position \(t\), compute projections:
\[
Q = HW*Q,\quad K = HW_K,\quad V = HW_V
\]
then reshape into heads:
\[
Q \in \mathbb{R}^{B \times n_h \times T \times d_h},\;
K,V \in \mathbb{R}^{B \times n*{kv} \times T \times d_h}.
\]

RoPE applies position-dependent rotations to \(Q\) and \(K\) (in 2D subspaces):
\[
Q' = \text{RoPE}(Q),\quad K' = \text{RoPE}(K)
\]

Attention scores:
\[
S*{b,i,t,s} = \frac{\langle Q'*{b,i,t,:}, K'\_{b,i,s,:}\rangle}{\sqrt{d_h}}
\]

Causal masking forces \(s \le t\) only:
\[
S\_{b,i,t,s} = -\infty \quad \text{if } s>t
\]

Softmax over \(s\):
\[
P*{b,i,t,s} = \text{softmax}(S*{b,i,t,:})\_s
\]

Weighted sum:
\[
O*{b,i,t,:} = \sum*{s \le t} P*{b,i,t,s} \, V*{b,i,s,:}
\]

Concatenate heads and project back to \(d\):
\[
A*{b,t,:} = \text{Concat}\_i(O*{b,i,t,:})W_O
\]

Add residual:
\[
H\_{\text{mid}} = H + A
\]

#### SwiGLU MLP

Let:
\[
G = H*{\text{mid}}W_g,\quad U = H*{\text{mid}}W*u
\]
SwiGLU nonlinearity:
\[
\text{MLP}(H*{\text{mid}}) = (\text{SiLU}(G) \odot U)W*d
\]
Residual:
\[
H \leftarrow H*{\text{mid}} + \text{MLP}(H\_{\text{mid}})
\]

After \(L\) layers, output hidden states:
\[
H^{(L)} \in \mathbb{R}^{B \times T \times d}.
\]

### 3.3 LM head: hidden states → logits

The LM head is a linear map to vocab size:
\[
Z = H^{(L)}W*{\text{lm}} + b,\quad
W*{\text{lm}} \in \mathbb{R}^{d \times V},\;
Z \in \mathbb{R}^{B \times T \times V}.
\]

Each \(Z\_{b,t,:}\) is the **logits** vector for the distribution of the _next_ token at position \(t\).

---

## 4) Loss: next-token cross-entropy

The dataset gives `labels = input_ids`, but causal LM loss is computed using a shift:

Targets:
\[
y*{b,t} = x*{b,t+1}\quad\text{for } t=1..T-1
\]
Predictions use logits at \(t\):
\[
Z*{b,t,:}\;\text{predicts}\; y*{b,t}
\]

Per-token cross-entropy:
\[
\ell*{b,t} = -\log \left(\frac{\exp(Z*{b,t,y*{b,t}})}{\sum*{v=1}^{V}\exp(Z\_{b,t,v})}\right)
\]

Average loss (no padding mask needed since sequences are fixed-length packed):
\[
\mathcal{L} = \frac{1}{B(T-1)}\sum*{b=1}^{B}\sum*{t=1}^{T-1}\ell\_{b,t}
\]

This \(\mathcal{L}\) is what `Trainer` backpropagates.

---

## 5) Backward pass (intuition + key gradient)

Let:
\[
p*{b,t,v} = \text{softmax}(Z*{b,t,:})\_v
\]

For one position \((b,t)\), the gradient w.r.t. logits is:
\[
\frac{\partial \ell*{b,t}}{\partial Z*{b,t,v}} = p*{b,t,v} - \mathbf{1}[v=y*{b,t}]
\]

So:

- the correct next-token logit gets pushed up (because \(p-1\) is negative when \(p<1\))
- incorrect logits get pushed down/up according to their probability mass

Then gradients flow backward through:

- LM head \(W\_{\text{lm}}\) (and possibly tied embeddings)
- transformer blocks (MLP + attention + norms)
- embeddings for the input tokens

The optimizer (AdamW) updates parameters \(\theta\) using first/second-moment estimates \(m, v\) and weight decay:
\[
\theta \leftarrow \theta - \eta \left(\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} + \lambda\theta\right)
\]

---

## 6) One end-to-end training sample (single sequence)

Assume we pick dataset index \(i\).

### 6.1 Load from disk

From `train.bin`:

1. Slice 512 token ids:
   \[
   x = \text{train.bin}[iT : (i+1)T]
   \]
2. Construct tensors:
   - `input_ids = x`
   - `labels = x`

### 6.2 Forward

1. Embedding lookup:
   \[
   H^{(0)}\_{t} = E[x_t]
   \]
2. Run \(L\) transformer layers (masked self-attention + MLP) to get \(H^{(L)}\)
3. Compute logits:
   \[
   Z*t = H^{(L)}\_t W*{\text{lm}} + b
   \]

### 6.3 Loss

Targets are shifted:
\[
y*t = x*{t+1}
\]
Compute cross-entropy over vocab for \(t=1..T-1\), then average:
\[
\mathcal{L} = \frac{1}{T-1}\sum*{t=1}^{T-1} -\log \text{softmax}(Z_t)*{y_t}
\]

### 6.4 Backward + update

1. Autograd computes \(\nabla\_\theta \mathcal{L}\)
2. Optimizer applies parameter update
3. Next step loads the next sequence index (or next batch) and repeats

That’s the full “LLM training loop”: **repeat next-token prediction** over many sequences of real text.

---

## 7) Where this lives in the repo

- Dataset preparation: `tiny_transformer/prepare_dataset.py`
- Dataset reading: `tiny_transformer/data/packed_dataset.py`
- Training orchestration (Trainer, LR schedule, callbacks, saving): `tiny_transformer/train.py`
- Model architecture + LM head + loss computation: `tiny_transformer/models/qwen3/modeling_qwen3.py`
