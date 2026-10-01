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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/rnn.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# %% [markdown]
# # Recurrent Neural Networks (RNN, LSTM, GRU)
# <p>RNNs are a class of neural networks designed to process sequential data, where the order of elements matters (e.g., text, speech, time series)</p>
# <p>Key Idea: RNNs have a "memory" or "hidden state" that captures information about previous elements in the sequence, influencing the processing of current elements. This is achieved by having recurrent connections (loops) in the network.</p>

# %% [markdown]
# This notebook compares a **vanilla RNN**, an **LSTM** and a **GRU** on one-step-ahead forecasting of a synthetic time series, then uses them for multi-step forecasting.
#
# Theory: [Recurrent Neural Networks (RNN)](README.md#3-recurrent-neural-networks-rnn-lstm-gru) · [RNN vs LSTM vs GRU](README.md#37-rnn-vs-lstm-vs-gru)
#
# ## Data: synthetic time series
# A sine wave plus a linear trend and noise. It is simple enough to visualise, but still needs memory of past values to predict well.

# %%
def generate_simple_data(n_samples=500):
    """
    Generate simple synthetic data - a combination of sine wave and linear trend
    This is easier to understand and visualize
    """
    x = torch.linspace(0, 4*torch.pi, n_samples)
    
    # Simple pattern: sine wave + small linear trend + minimal noise
    data = torch.sin(x) + 0.1*x + 0.05*torch.randn(n_samples)
    
    return data

raw_data = generate_simple_data(n_samples=400)

# %%
raw_data[:5]

# %%
plt.figure(figsize=(12, 4))
plt.plot(raw_data)
plt.title('Simple Synthetic Time Series Data')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# %%
print(f"Generated {len(raw_data)} data points")
print(f"Data range: {raw_data.min():.3f} to {raw_data.max():.3f}")


# %% [markdown]
# ## Sliding-window sequences
# Each input is a window of `SEQUENCE_LENGTH` consecutive values, and the target is the **next** value. The data is scaled to $[0, 1]$ with `MinMaxScaler` first. The train/test split is chronological: the last 20% of windows form the test set.

# %%
def create_sequences(data, seq_length):
    """
    Create sequences for RNN training
    
    Returns:
        X: Input sequences
        y: Target values (next value after each sequence)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input sequence
        seq_x = data[i:i + seq_length]
        # Target (next single value)
        seq_y = data[i + seq_length]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

# Parameters
SEQUENCE_LENGTH = 10 

# %%
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()

# %%
X, y = create_sequences(normalized_data, SEQUENCE_LENGTH)

# %%
X[:5], y[:5], X.shape, y.shape

# %%
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")


# %% [markdown]
# ## Models
# All three share the same interface: a recurrent layer followed by a linear head on the **last** hidden state.
# - **RNN:** $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$
# - **LSTM:** adds a cell state and input/forget/output gates to fight vanishing gradients
# - **GRU:** a lighter gated variant with update/reset gates and no separate cell state

# %%
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Forward pass through RNN
        rnn_out, _ = self.rnn(x, h0)
        
        # Take the output from the last time step
        last_output = rnn_out[:, -1, :]
        
        # Pass through final linear layer
        output = self.fc(last_output)
        
        return output


# %%
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        
        return output


# %%
class SimpleGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        gru_out, _ = self.gru(x, h0)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        
        return output


# %% [markdown]
# ## DataLoaders
# The inputs get a trailing feature dimension, giving shape `(batch, seq_len, 1)`, which matches `batch_first=True`.

# %%
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1) 
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_tensor = torch.FloatTensor(y_test)


# %%
BATCH_SIZE = 16

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %% [markdown]
# ## Training
# MSE loss with the Adam optimizer. Train and test loss are recorded every epoch.

# %%
def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.01):
    """
    Train the RNN model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    
    print(f"Training {model.__class__.__name__}...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                test_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}')
    
    return train_losses, test_losses


# %%
rnn_model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1, output_size=1)
lstm_model = SimpleLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
gru_model = SimpleGRU(input_size=1, hidden_size=32, num_layers=1, output_size=1)

# %%
# Train RNN model
rnn_train_losses, rnn_test_losses = train_model(rnn_model, train_loader, test_loader, num_epochs=30)


# %%
# Train LSTM and GRU
lstm_train_losses, lstm_test_losses = train_model(lstm_model, train_loader, test_loader, num_epochs=30)

gru_train_losses, gru_test_losses = train_model(gru_model, train_loader, test_loader, num_epochs=30)


# %% [markdown]
# ## Unrolling the RNN by hand
# `nn.RNN` implements $h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$. Applying that recurrence manually with the trained weights, followed by the same linear head, should reproduce the model's predictions exactly.

# %%
from mlf_utils import check_close

rnn_layer = rnn_model.rnn
x_seq = X_test_tensor[:8]                           # (batch, seq_len, 1)
h = torch.zeros(x_seq.size(0), rnn_layer.hidden_size)
with torch.no_grad():
    for t in range(x_seq.size(1)):
        h = torch.tanh(x_seq[:, t] @ rnn_layer.weight_ih_l0.T + rnn_layer.bias_ih_l0
                       + h @ rnn_layer.weight_hh_l0.T + rnn_layer.bias_hh_l0)
    manual_pred = rnn_model.fc(h)
    library_pred = rnn_model(x_seq)
check_close("Manual RNN unroll vs nn.RNN", manual_pred, library_pred, atol=1e-5)

# %% [markdown]
# ## Loss curves

# %%
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(rnn_train_losses, label='RNN Train', alpha=0.8)
plt.plot(rnn_test_losses, label='RNN Test', alpha=0.8)
# plt.plot(lstm_train_losses, label='LSTM Train', alpha=0.8)
# plt.plot(lstm_test_losses, label='LSTM Test', alpha=0.8)
# plt.plot(gru_train_losses, label='GRU Train', alpha=0.8)
# plt.plot(gru_test_losses, label='GRU Test', alpha=0.8)
plt.title('Training Progress - Linear Scale')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(rnn_train_losses, label='RNN Train', alpha=0.8)
plt.plot(rnn_test_losses, label='RNN Test', alpha=0.8)
# plt.plot(lstm_train_losses, label='LSTM Train', alpha=0.8)
# plt.plot(lstm_test_losses, label='LSTM Test', alpha=0.8)
# plt.plot(gru_train_losses, label='GRU Train', alpha=0.8)
# plt.plot(gru_test_losses, label='GRU Test', alpha=0.8)
plt.title('Training Progress - Log Scale')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Evaluation
# Predictions are transformed back to the original scale before computing error metrics.

# %%
def evaluate_model(model, X_test, y_test, scaler, model_name="Model"):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()
    
    # Denormalize predictions and actual values
    y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    mse = np.mean((y_test_actual - predictions_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - predictions_actual))
    
    print(f"{model_name} Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    return predictions_actual, y_test_actual


# %%
rnn_pred, y_actual = evaluate_model(rnn_model, X_test_tensor, y_test_tensor, scaler, "Simple RNN")
print()
lstm_pred, _ = evaluate_model(lstm_model, X_test_tensor, y_test_tensor, scaler, "LSTM")
print()
gru_pred, _ = evaluate_model(gru_model, X_test_tensor, y_test_tensor, scaler, "GRU")

# %%
plt.figure(figsize=(15, 5))

# Show first 50 test points for better visibility
n_points = 50

plt.subplot(1, 3, 1)
plt.plot(y_actual[:n_points], 'b-', label='Actual', linewidth=2)
plt.plot(rnn_pred[:n_points], 'r--', label='RNN Prediction', alpha=0.8)
plt.title('Simple RNN Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(y_actual[:n_points], 'b-', label='Actual', linewidth=2)
plt.plot(lstm_pred[:n_points], 'g--', label='LSTM Prediction', alpha=0.8)
plt.title('LSTM Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(y_actual[:n_points], 'b-', label='Actual', linewidth=2)
plt.plot(gru_pred[:n_points], 'm--', label='GRU Prediction', alpha=0.8)
plt.title('GRU Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 4))
plt.plot(y_actual[:n_points], 'b-', label='Actual', linewidth=2)
plt.plot(rnn_pred[:n_points], 'r--', label='RNN', alpha=0.8)
plt.plot(lstm_pred[:n_points], 'g--', label='LSTM', alpha=0.8)
plt.plot(gru_pred[:n_points], 'm--', label='GRU', alpha=0.8)
plt.title('All Models Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Multi-step (autoregressive) forecasting
# Each prediction is appended to the input window and fed back in to predict the next step. Errors compound over the horizon, which makes this a harder test of what each model has learned.

# %%
def multi_step_prediction(model, initial_sequence, n_steps, scaler):
    model.eval()
    predictions = []
    
    if len(initial_sequence.shape) > 1:
        current_seq = initial_sequence.squeeze()
    else:
        current_seq = initial_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Reshape for model input: (1, seq_length, 1)
            model_input = current_seq.unsqueeze(0).unsqueeze(-1)
            
            pred = model(model_input)
            pred_value = pred.squeeze().item()
            predictions.append(pred_value)
            
            current_seq = torch.cat([current_seq[1:], torch.tensor([pred_value])])
    
    return np.array(predictions)


# %%
test_sequence = X_test_tensor[0].squeeze() 
n_future_steps = 15


# %%
rnn_future = multi_step_prediction(rnn_model, test_sequence, n_future_steps, scaler)
lstm_future = multi_step_prediction(lstm_model, test_sequence, n_future_steps, scaler)
gru_future = multi_step_prediction(gru_model, test_sequence, n_future_steps, scaler)


# %%
historical_data = scaler.inverse_transform(test_sequence.numpy().reshape(-1, 1)).flatten()
rnn_future_actual = scaler.inverse_transform(rnn_future.reshape(-1, 1)).flatten()
lstm_future_actual = scaler.inverse_transform(lstm_future.reshape(-1, 1)).flatten()
gru_future_actual = scaler.inverse_transform(gru_future.reshape(-1, 1)).flatten()


# %%
plt.figure(figsize=(12, 6))
historical_steps = range(len(historical_data))
future_steps = range(len(historical_data), len(historical_data) + n_future_steps)

plt.plot(historical_steps, historical_data, 'b-', label='Historical Data', linewidth=2)
plt.plot(future_steps, rnn_future_actual, 'r--', label='RNN Future', alpha=0.8)
plt.plot(future_steps, lstm_future_actual, 'g--', label='LSTM Future', alpha=0.8)
plt.plot(future_steps, gru_future_actual, 'm--', label='GRU Future', alpha=0.8)

plt.axvline(x=len(historical_data)-1, color='black', linestyle=':', alpha=0.5, label='Prediction Start')
plt.title('Multi-step Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# %%

# %%
