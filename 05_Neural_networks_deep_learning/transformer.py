# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: develop_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/transformer.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# # Attention Is All You Need: A Transformer from Scratch
#
# A from-scratch implementation of the encoder-decoder Transformer (Vaswani et al., 2017), trained on a toy **sequence-reversal** task (`[5, 9, 3] → [3, 9, 5]`). The task is easy to verify by eye, but still requires the decoder to attend to the right source positions.
#
# Theory: [Attention Is All You Need](README.md#4-transformers-attention-is-all-you-need)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# %% [markdown]
# ## Building blocks
# ### Scaled dot-product attention
# $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
# Scaling by $\sqrt{d_k}$ keeps the dot products from growing with dimension and saturating the softmax. Masked positions are set to a large negative number ($-10^9$) before the softmax, so they get zero weight.

# %%
def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (batch_size, seq_len, d_k)
        K: Key matrix (batch_size, seq_len, d_k)
        V: Value matrix (batch_size, seq_len, d_v)
        mask: Optional mask (batch_size, seq_len, seq_len)
        dropout: Optional dropout layer
    
    Returns:
        output: Attention output (batch_size, seq_len, d_v)
        attention_weights: Attention weights (batch_size, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


# %% [markdown]
# ### Verifying against PyTorch
# `F.scaled_dot_product_attention` (PyTorch ≥ 2.0) is the fused library version. Note the mask convention: ours masks where `mask == 0`, and PyTorch's boolean `attn_mask` uses `True` = *may attend*, so the same boolean mask works for both.

# %%
from mlf_utils import check_close

gen = torch.Generator().manual_seed(0)              # local generator: doesn't disturb the global RNG
q, k, v = torch.randn(3, 2, 6, 16, generator=gen).unbind(0)
out_ours, _ = scaled_dot_product_attention(q, k, v)
check_close("Attention vs F.scaled_dot_product_attention", out_ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5)

causal = torch.tril(torch.ones(6, 6)).bool()
out_ours_masked, _ = scaled_dot_product_attention(q, k, v, mask=causal)
check_close("Causal attention vs F.scaled_dot_product_attention", out_ours_masked,
            F.scaled_dot_product_attention(q, k, v, attn_mask=causal), atol=1e-5)


# %% [markdown]
# ### Multi-head attention
# $Q$, $K$ and $V$ are projected into `num_heads` subspaces of size $d_k = d_{model} / h$. Attention is computed in each subspace in parallel, and the results are concatenated and projected back, which lets heads specialise on different relationships.

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear layer
        output = self.W_o(attention_output)
        
        return output, attention_weights


# %% [markdown]
# ### Position-wise feed-forward network
# $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$, applied independently at each position.

# %%
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# %% [markdown]
# ### Positional encoding
# Attention is permutation-invariant, so position information is added with fixed sinusoids:
# $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$ and $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$.

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# %% [markdown]
# ### Encoder layer
# Self-attention → Add & Norm → Feed-forward → Add & Norm.

# %%
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# %% [markdown]
# ### Decoder layer
# Masked self-attention → Add & Norm → Cross-attention over the encoder output → Add & Norm → Feed-forward → Add & Norm.

# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, attention_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, attention_weights


# %% [markdown]
# ### Full model
# Token embeddings (scaled by $\sqrt{d_{model}}$) + positional encoding → $N$ encoder layers → $N$ decoder layers → linear projection to the vocabulary.

# %%
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final linear layer
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, src, src_mask=None):
        # Embedding + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # Pass through encoder layers
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
            
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through decoder layers
        decoder_output = tgt_emb
        attention_weights = []
        for layer in self.decoder_layers:
            decoder_output, attn_weights = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)
            
        return decoder_output, attention_weights
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output, attention_weights = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        return output, attention_weights


# %% [markdown]
# ## Masks
# - **Padding mask:** stops attention to `<pad>` tokens.
# - **Look-ahead (causal) mask:** stops decoder position $t$ from seeing positions $> t$, so the model cannot cheat during training.

# %%
def create_padding_mask(seq, pad_idx=0):
    """Create mask to hide padding tokens"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


# %%
def create_look_ahead_mask(size):
    """Create mask to prevent looking at future tokens"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


# %%
def create_masks(src, tgt, pad_idx=0):
    """Create all masks needed for training"""
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_mask = create_padding_mask(tgt, pad_idx)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(src.device)
    tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
    
    return src_mask, tgt_mask


# %% [markdown]
# ## Toy task: sequence reversal
# Random token sequences of variable length. Targets are the reversed sequence wrapped in `<sos>` / `<eos>`, and index 0 is reserved for padding.

# %%
# Dataset creation
class SimpleTranslationDataset(Dataset):
    def __init__(self, num_samples=1000, max_len=20):
        """
        Create a simple synthetic dataset for demonstration.
        Task: Reverse the input sequence (e.g., [1,2,3,4] -> [4,3,2,1])
        """
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = 100
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        
        # Generate synthetic data
        self.data = []
        for _ in range(num_samples):
            # Random sequence length
            seq_len = np.random.randint(3, max_len - 2)
            
            # Random sequence (avoid special tokens)
            src_seq = np.random.randint(3, self.vocab_size, seq_len)
            
            # Target is reversed sequence with SOS and EOS
            tgt_seq = np.concatenate([[self.sos_idx], src_seq[::-1], [self.eos_idx]])
            
            # Pad sequences
            src_padded = np.pad(src_seq, (0, max_len - len(src_seq)), constant_values=self.pad_idx)
            tgt_padded = np.pad(tgt_seq, (0, max_len - len(tgt_seq)), constant_values=self.pad_idx)
            
            self.data.append((src_padded, tgt_padded))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)



# %%
train_dataset = SimpleTranslationDataset(num_samples=2000, max_len=15)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# %%
len(train_dataset), train_dataset.vocab_size

# %% [markdown]
# ## Training
# A smaller configuration than the paper's base model ($d_{model}=512$, 6 layers) so it trains quickly. The decoder uses **teacher forcing**: it receives the target shifted right and predicts the next token, and the loss ignores padding.

# %%
# Model hyperparameters
d_model = 256
num_heads = 8
num_layers = 4
d_ff = 512
dropout = 0.1
vocab_size = train_dataset.vocab_size

# %%
# Initialize model
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    dropout=dropout
).to(device)

# %%
# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# %%
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Training")):
        src, tgt = src.to(device), tgt.to(device)
        
        # Prepare input and target
        tgt_input = tgt[:, :-1]  # Remove last token for input
        tgt_output = tgt[:, 1:]  # Remove first token for target
        
        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt_input, train_dataset.pad_idx)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)


# %%
# Train the model
num_epochs = 50
train_losses = []

# %%
print("Starting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(avg_loss)
    print(f"Average Loss: {avg_loss:.4f}")

# %%
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# %% [markdown]
# ## Inference: greedy decoding
# Encode the source once, then generate one token at a time, feeding each prediction back into the decoder until `<eos>` or the maximum length.

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """Simple greedy decoding for inference"""
    model.eval()
    
    encoder_output = model.encode(src, src_mask)
    
    # Initialize target sequence with start symbol
    tgt = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = create_look_ahead_mask(tgt.size(1)).to(device)
        
        decoder_output, _ = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = model.linear(decoder_output)
        
        # Get the next token
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if end token is generated
        if next_token.item() == end_symbol:
            break
    
    return tgt


# %%
model.eval()
test_samples = 5

# %%
with torch.no_grad():
    for i in range(test_samples):
        src, tgt = train_dataset[i]
        src = src.unsqueeze(0).to(device)
        
        # Remove padding for display
        src_tokens = src.squeeze().cpu().numpy()
        src_tokens = src_tokens[src_tokens != train_dataset.pad_idx]
        
        tgt_tokens = tgt.cpu().numpy()
        tgt_tokens = tgt_tokens[tgt_tokens != train_dataset.pad_idx]
        
        # Create source mask
        src_mask = create_padding_mask(src, train_dataset.pad_idx).to(device)
        
        # Generate prediction
        pred = greedy_decode(
            model, src, src_mask, max_len=15, 
            start_symbol=train_dataset.sos_idx, 
            end_symbol=train_dataset.eos_idx, 
            device=device
        )
        
        pred_tokens = pred.squeeze().cpu().numpy()
        
        print(f"Sample {i+1}:")
        print(f"Source:     {src_tokens}")
        print(f"Target:     {tgt_tokens}")
        print(f"Predicted:  {pred_tokens}")
        print(f"Correct:    {np.array_equal(tgt_tokens, pred_tokens)}")
        print("-" * 30)


# %%
