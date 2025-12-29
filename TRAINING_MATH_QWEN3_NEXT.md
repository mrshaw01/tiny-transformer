# Training walkthrough (math) — Qwen3-Next

This note explains the same end-to-end training pipeline as `TRAINING_MATH_QWEN3.md`, but for the **Qwen3-Next**
implementation selected when your config has `"model_type": "qwen3_next"`.

- Model class: `Qwen3NextForCausalLM`
- Code: `tiny_transformer/models/qwen3_next/modeling_qwen3_next.py`

What’s special about Qwen3-Next (vs a “plain” decoder-only Transformer) is the **token-mixer** inside each decoder
layer: each layer is either:

- `full_attention`: causal self-attention with RoPE **plus an output gate**, or
- `linear_attention`: a **Gated DeltaNet** token mixer (recurrent / linear-time style) with a depthwise causal conv and
  a gated RMSNorm using an extra signal \(z\).

Your demo config controls this per-layer via `layer_types` in `configs/qwen3_next_demo.json`.

---

## 0) Notation and shapes

- Vocab size: \(V\)
- Sequence length: \(T\)
- Batch size: \(B\)
- Hidden size: \(d\)
- Layers: \(L\)

For full attention:

- Attention heads: \(n_h\)
- KV heads (GQA): \(n\_{kv}\)
- Head dim: \(d_h\)

For linear attention (Gated DeltaNet):

- Linear key heads: \(n_k\), key head dim: \(d_k\)
- Linear value heads: \(n_v\), value head dim: \(d_v\)

Input token ids:
\[
X \in \mathbb{Z}^{B \times T}
\]

Hidden states at layer \(\ell\):
\[
H^{(\ell)} \in \mathbb{R}^{B \times T \times d}
\]

---

## 1) Dataset and batching

The dataset pipeline is identical to Qwen3:

- `tiny_transformer/prepare_dataset.py` creates `data/train.bin`, `data/val.bin`, `data/meta.json`
- `PackedMemmapDataset` slices packed windows of length \(T\)
- the dataset returns `input_ids` and `labels` as the same tensor

So each batch is:
\[
X \in \mathbb{Z}^{B \times T},\quad \text{labels} \in \mathbb{Z}^{B \times T}
\]

---

## 2) Forward pass (detailed)

### 2.1 Embedding

With token embedding matrix \(E \in \mathbb{R}^{V \times d}\):
\[
H^{(0)}_{b,t,:} = E[x_{b,t}]
\]

### 2.2 Decoder layer structure (pre-norm)

In code (`Qwen3NextDecoderLayer`), each layer does:

1. Input RMSNorm:
   \[
   \tilde{H} = \text{RMSNorm}(H)
   \]
2. Token mixer: either full attention or Gated DeltaNet:
   \[
   M = \text{TokenMixer}(\tilde{H})
   \]
3. Residual add:
   \[
   H \leftarrow H + M
   \]
4. Post-attention RMSNorm + MLP + residual add (standard decoder block):
   \[
   H \leftarrow H + \text{MLP}(\text{RMSNorm}(H))
   \]

The interesting part is the token mixer.

---

## 2A) `full_attention`: gated causal self-attention

Implementation: `Qwen3NextAttention`

### 2A.1 Projections with a gate

Let the layer input be \(\tilde{H} \in \mathbb{R}^{B \times T \times d}\).

Qwen3-Next computes queries and a **gate** from one projection:
\[
\begin{aligned}
[\tilde{Q}; G] &= \tilde{H} W_Q \\
\tilde{Q} &\in \mathbb{R}^{B \times T \times (n_h d_h)} \\
G &\in \mathbb{R}^{B \times T \times (n_h d_h)}
\end{aligned}
\]

and keys/values:
\[
\tilde{K} = \tilde{H}W*K,\quad \tilde{V} = \tilde{H}W_V
\]
with shapes:
\[
\tilde{K},\tilde{V} \in \mathbb{R}^{B \times T \times (n*{kv} d_h)}
\]

Then reshape into heads:
\[
Q \in \mathbb{R}^{B \times n*h \times T \times d_h},\quad
K,V \in \mathbb{R}^{B \times n*{kv} \times T \times d_h}
\]

### 2A.2 Per-head RMSNorm + RoPE

The implementation RMS-normalizes \(Q\) and \(K\) over the head dimension:
\[
Q \leftarrow \text{RMSNorm}(Q),\quad K \leftarrow \text{RMSNorm}(K)
\]

Then applies RoPE to \(Q\) and \(K\). Qwen3-Next supports **partial RoPE**: only the first `rotary_dim` features are
rotated; the remaining features are passed through unchanged.

### 2A.3 Causal attention

For each head \(i\), and positions \(t,s\):
\[
S\_{t,s}^{(i)} = \frac{\langle Q^{(i)}\_t, K^{(i)}\_s\rangle}{\sqrt{d_h}} + \text{mask}(t,s)
\]
where \(\text{mask}(t,s)=-\infty\) if \(s>t\).

Softmax weights:
\[
P*{t,s}^{(i)} = \text{softmax}(S*{t,:}^{(i)})\_s
\]

Attention output:
\[
A*t^{(i)} = \sum*{s \le t} P\_{t,s}^{(i)} V_s^{(i)}
\]

Concatenate heads:
\[
A_t = \text{Concat}\_i(A_t^{(i)}) \in \mathbb{R}^{n_h d_h}
\]

### 2A.4 Output gating (the key difference vs Qwen3)

Qwen3-Next gates the attention output elementwise:
\[
\hat{A}\_t = A_t \odot \sigma(G_t)
\]
where \(G_t \in \mathbb{R}^{n_h d_h}\) comes from the \(W_Q\) projection above and \(\sigma\) is the sigmoid.

Finally apply output projection:
\[
M_t = \hat{A}\_t W_O \in \mathbb{R}^{d}
\]

So the token-mixer output is \(M \in \mathbb{R}^{B \times T \times d}\).

---

## 2B) `linear_attention`: Gated DeltaNet (Qwen3NextGatedDeltaNet)

Implementation: `Qwen3NextGatedDeltaNet`

This is the most important “new” structure. It mixes tokens using:

1. Linear projections to produce \(q,k,v\) **plus** an extra signal \(z\) and scalars \(a,b\)
2. A depthwise causal 1D convolution on the concatenated \(q,k,v\)
3. A recurrent “delta-rule” state update that produces outputs in \(O(T)\) (conceptually)
4. A gated RMSNorm using \(z\)
5. A final output projection back to \(d\)

### 2B.1 Dimensions

From config:
\[
\begin{aligned}
n_k &= \texttt{linear_num_key_heads}, \quad d_k = \texttt{linear_key_head_dim} \\
n_v &= \texttt{linear_num_value_heads}, \quad d_v = \texttt{linear_value_head_dim}
\end{aligned}
\]

Define:
\[
\text{key_dim} = n_k d_k,\quad \text{value_dim} = n_v d_v
\]

### 2B.2 Projections: produce \(q,k,v,z\) and \(a,b\)

Given input \(\tilde{H} \in \mathbb{R}^{B \times T \times d}\), the module computes:

- `in_proj_qkvz`: produces \(q,k,v,z\)
- `in_proj_ba`: produces \(b,a\)

Conceptually (ignoring internal ordering details), we obtain:
\[
\begin{aligned}
Q &\in \mathbb{R}^{B \times T \times n_k \times d_k} \\
K &\in \mathbb{R}^{B \times T \times n_k \times d_k} \\
V &\in \mathbb{R}^{B \times T \times n_v \times d_v} \\
Z &\in \mathbb{R}^{B \times T \times n_v \times d_v} \\
b &\in \mathbb{R}^{B \times T \times n_v} \\
a &\in \mathbb{R}^{B \times T \times n_v}
\end{aligned}
\]

The model uses:
\[
\beta = \sigma(b) \in (0,1)
\]
and a learned negative decay:
\[
g = -\exp(A\_{\log}) \cdot \text{softplus}(a + \text{dt_bias})
\]
so \(g \le 0\) and \(\exp(g)\in (0,1]\) acts like a per-step decay.

### 2B.3 Depthwise causal convolution on concatenated \(q,k,v\)

The code concatenates flattened \(q,k,v\) and applies a depthwise 1D convolution along time (with SiLU):
\[
\begin{aligned}
U_t &= [Q_t; K_t; V_t] \in \mathbb{R}^{(2\cdot \text{key_dim}+\text{value_dim})} \\
\tilde{U} &= \text{SiLU}(\text{DWConv1D}(U))
\end{aligned}
\]

Then it splits \(\tilde{U}\) back into \(\tilde{Q},\tilde{K},\tilde{V}\) and reshapes to the head shapes above.

### 2B.4 Delta-rule recurrence (exact update used in code)

The fallback implementation (`torch_recurrent_gated_delta_rule`) maintains a per-head state:
\[
S_t \in \mathbb{R}^{d_k \times d_v}
\]

For clarity, consider a single batch element and head; drop batch/head indices.

First scale and optionally \(\ell_2\)-normalize \(q_t,k_t\):
\[
q_t \leftarrow \frac{q_t}{\sqrt{d_k}},\quad k_t \leftarrow k_t\;\;\;(\text{plus optional } \ell_2\text{-norm})
\]

Then for each timestep \(t=1..T\):

1. Apply decay:
   \[
   S \leftarrow S \cdot \exp(g_t)
   \]
2. Compute “memory” value implied by current key:
   \[
   m_t = S^\top k_t \in \mathbb{R}^{d_v}
   \]
3. Delta towards the current value:
   \[
   \Delta_t = (v_t - m_t)\cdot \beta_t
   \]
4. Rank-1 update:
   \[
   S \leftarrow S + k_t \otimes \Delta_t
   \]
5. Output:
   \[
   o_t = S^\top q_t \in \mathbb{R}^{d_v}
   \]

Stacking over heads/tokens gives:
\[
O \in \mathbb{R}^{B \times T \times n_v \times d_v}
\]

### 2B.5 Gated RMSNorm with \(z\)

After the delta-rule output, Qwen3-Next applies a gated RMSNorm using \(z\) (`Qwen3NextRMSNormGated`):
\[
\hat{O} = \text{RMSNorm}(O)\odot \text{SiLU}(Z)
\]

This is not the same as the attention gate; it’s a post-mixer gating driven by \(z\).

### 2B.6 Final projection back to model dim

Flatten heads and project:
\[
M*t = \text{reshape}(\hat{O}\_t) W*{\text{out}} \in \mathbb{R}^{d}
\]

This \(M\) is returned as the token-mixer output for the layer.

---

## 2C) LM head → logits

Same as Qwen3: a linear head to vocab size:
\[
\text{logits } Z = H^{(L)} W\_{\text{lm}} + b,\quad Z \in \mathbb{R}^{B \times T \times V}
\]

---

## 3) Loss: next-token cross-entropy

Same as Qwen3: shifted next-token prediction.

Targets:
\[
y*{b,t} = x*{b,t+1}\quad\text{for } t=1..T-1
\]

Loss:
\[
\mathcal{L} = \frac{1}{B(T-1)}\sum*{b=1}^{B}\sum*{t=1}^{T-1}
\left[-\log \text{softmax}(Z_{b,t,:})_{y_{b,t}}\right]
\]

---

## 4) Backward pass (key gradient)

Same key result:
\[
\frac{\partial \ell*{b,t}}{\partial Z*{b,t,v}} = p*{b,t,v} - \mathbf{1}[v=y*{b,t}]
\]

Everything else is standard backprop through the model’s computation graph (token mixer + MLP + embeddings).

---

## 5) One end-to-end training sample

Exactly the same step sequence as Qwen3:

1. Load one packed sequence \(x\) from `train.bin`
2. Forward through `Qwen3NextForCausalLM` → logits \(Z\)
3. Compute shifted cross-entropy loss \(\mathcal{L}\)
4. Backprop \(\nabla\_\theta \mathcal{L}\)
5. AdamW update
