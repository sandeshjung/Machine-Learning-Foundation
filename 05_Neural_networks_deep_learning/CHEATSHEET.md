# 05 · Cheat Sheet: Neural Networks & Deep Learning

> **Notebooks:** [mlp_backprop](mlp_backprop.ipynb) · [cnn](cnn.ipynb) · [rnn](rnn.ipynb) · [transformer](transformer.ipynb) · [optimizers_regularization](optimizers_regularization.ipynb)
>
> **Full explanations:** [README](README.md)

## MLP & backprop

**Forward**
- $Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$, $A^{(l)} = g(Z^{(l)})$
- Softmax + cross-entropy together give $\frac{\partial L}{\partial Z^{(L)}} = \frac{1}{m}(\hat{P} - Y_{\text{onehot}})$

**Backward**
- $\frac{\partial L}{\partial W^{(l)}} = A^{(l-1)\top} \delta^{(l)}$
- $\frac{\partial L}{\partial b^{(l)}} = \sum_{\text{batch}} \delta^{(l)}$
- $\delta^{(l-1)} = \delta^{(l)} W^{(l)\top} \odot g'(Z^{(l-1)})$

| Activation | $g(z)$ | $g'(z)$ | Notes |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $g(1-g)$ | Saturates, not zero-centred. Use for output probabilities |
| Tanh | $\tanh z$ | $1 - g^2$ | Zero-centred, still saturates |
| ReLU | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | Default choice. Units can "die" |
| GELU / SiLU | smooth ReLU variants | | Common in Transformers |

**Initialisation**
- **Xavier / Glorot:** $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$, for tanh and sigmoid
- **He:** $\text{Var}(W) = \frac{2}{n_{in}}$, for ReLU

## CNN

- Output size: $\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$.
- "Same" padding with $s = 1$ means $p = \frac{k-1}{2}$.
- Conv parameters: $(k \cdot k \cdot C_{in} + 1) \cdot C_{out}$. Much less than a dense layer on the same input.
- A typical block is `Conv → BatchNorm → ReLU → Pool`. Spatial size shrinks while channels grow.

## RNN / LSTM / GRU

- RNN: $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$. Suffers vanishing or exploding gradients through time (BPTT).
- LSTM: forget, input and output gates plus a cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The additive path keeps gradients alive.
- GRU: update and reset gates, no separate cell state. Fewer parameters, often on par with LSTM.
- Use gradient clipping (`clip_grad_norm_`) against exploding gradients.

## Transformer

- $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$, where the mask $M$ is $-\infty$ at blocked positions.
- Multi-head: $h$ heads of size $d_k = d_{model}/h$, concatenated and then projected.
- Each block: `x + Attention(Norm(x))`, then `x + FFN(Norm(x))` (pre-norm) or the post-norm variant.
- Masks: **padding** (ignore `<pad>`) and **causal** (the decoder cannot see the future). Positional encodings are needed because attention is order-agnostic.

## Optimisers

| | Update | Typical lr |
|---|---|---|
| SGD + momentum | $v \leftarrow \mu v + g$, $\theta \leftarrow \theta - \eta v$ (PyTorch convention) | 0.01–0.1, $\mu = 0.9$ |
| RMSProp | $s \leftarrow \beta s + (1-\beta) g^2$, $\theta \leftarrow \theta - \eta \frac{g}{\sqrt{s} + \epsilon}$ | 1e-3 |
| Adam | $m, v$ moments, bias-corrected $\hat{m} = \frac{m}{1-\beta_1^t}$, $\hat{v} = \frac{v}{1-\beta_2^t}$, $\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | 1e-3 (3e-4 for Transformers) |
| AdamW | Adam with **decoupled** weight decay | Preferred over Adam + L2 |

**Schedules:** StepLR, ReduceLROnPlateau, cosine annealing, and warm-up (essential for Transformers).

## Regularisation

- L2 / weight decay (prefer `AdamW`)
- Dropout, $p$ = 0.1–0.5, only active in `model.train()`
- BatchNorm / LayerNorm
- Early stopping
- Data augmentation

## Pitfalls

- Call `model.train()` and `model.eval()` at the right times. Dropout and BatchNorm behave differently in each.
- Wrap inference in `torch.no_grad()`, and call `optimizer.zero_grad()` every step.
- `CrossEntropyLoss` expects **raw logits** and class indices. Don't apply softmax first.
- Loss is NaN? Lower the learning rate, check the input scale, clip gradients, and use the log-sum-exp trick for manual softmax.
- Debug by overfitting one small batch first. If you can't, there's a bug.
