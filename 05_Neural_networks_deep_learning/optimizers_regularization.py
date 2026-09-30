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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/optimizers_regularization.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # For learning rate scheduling
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# <p>Effective training of deep neural networks involves more than just basic Gradient Descent.</p>
#
# 1. Advanced Optimization Algorithms
# 2. Learning Rate Scheduling
# 3. Regularization Techniques

# %%
transform_reg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))      # mean and std for FashionMNIST (approx)
])

# %%
transform_reg

# %%
batch_size_reg = 128

train_dataset_or = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                             download=True, transform=transform_reg)
test_or = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                            download=True, transform=transform_reg)

# %%
train_size_or = int(0.8 * len(train_dataset_or))
val_size_or = len(train_dataset_or) - train_size_or
train_or, val_or = random_split(train_dataset_or, [train_size_or, val_size_or],
                                generator=torch.Generator().manual_seed(42))

# %%
train_loader_or = DataLoader(train_or, batch_size=batch_size_reg, shuffle=True, num_workers=2)
val_loader_or = DataLoader(val_or, batch_size=batch_size_reg, shuffle=False, num_workers=2)
test_loader_or = DataLoader(test_or, batch_size=batch_size_reg, shuffle=False, num_workers=2)


# %%
classes_or = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
N_CLASSES_OR = len(classes_or)
FLATTENED_DIM_OR = 28 * 28

# %%
print(f"Train samples: {len(train_or)}, Val samples: {len(val_or)}, Test samples: {len(test_or)}")


# %%
# simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.0):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) 
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, output_dim) # Output logits

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x) 
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# %%
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 128


# %% [markdown]
# ### Optimization Algorithms
#
# #### Stochastic Gradient Descent (SGD)
# <p>We'll use this as a baseline to compare with Adam. SGD with momentum is often better than vanilla SGD.</p>
#
# #### Adam Optimizer
# <p>Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that computes individual learning rates for different parameters. It combines ideas from RMSProp (adaptive learning rates based on squared gradients) and Momentum (using a moving average of gradients). Ofter a good default choice for many deep learning tasks. Common parameters: lr (learning rate), betas (coefficients for moving averages), eps (for numerical stability).</p>

# %%
def train_and_validate_model(model_instance, model_name,
                             train_dl, val_dl,
                             optimizer_instance, criterion_instance,
                             scheduler_instance=None, num_epochs=10):
    print(f"\n--- Training {model_name} with Optimizer: {type(optimizer_instance).__name__} ---")
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model_instance.train() # Set model to training mode
        running_train_loss = 0.0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer_instance.zero_grad()
            outputs_logits = model_instance(inputs)
            loss = criterion_instance(outputs_logits, labels)
            loss.backward()
            optimizer_instance.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_dl.dataset)
        history['train_loss'].append(epoch_train_loss)

        model_instance.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs_v, labels_v in val_dl:
                inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)
                outputs_logits_v = model_instance(inputs_v)
                loss_v = criterion_instance(outputs_logits_v, labels_v)
                running_val_loss += loss_v.item() * inputs_v.size(0)
                _, preds_v = torch.max(outputs_logits_v, 1)
                correct_val += (preds_v == labels_v).sum().item()
                total_val += labels_v.size(0)
        
        epoch_val_loss = running_val_loss / len(val_dl.dataset)
        history['val_loss'].append(epoch_val_loss)
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0
        history['val_accuracy'].append(epoch_val_acc)

        current_lr_str = ""
        if scheduler_instance:
            current_lr_str = f", Current LR: {optimizer_instance.param_groups[0]['lr']:.1e}"
            scheduler_instance.step(epoch_val_loss if isinstance(scheduler_instance, lr_scheduler.ReduceLROnPlateau) else None)


        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}{current_lr_str}")
              
    return history

# %%
# Experiment: SGD vs Adam
epochs = 10
criterion = nn.CrossEntropyLoss()

# %%
# model for SGD
mlp_sgd = SimpleMLP(FLATTENED_DIM_OR, HIDDEN_DIM1, HIDDEN_DIM2, N_CLASSES_OR, dropout_rate=0.0).to(device)
optimizer_sgd = optim.SGD(mlp_sgd.parameters(), lr=0.01, momentum=0.9)
history_sgd = train_and_validate_model(mlp_sgd, "MLP with SGD + Momentum",
                            train_loader_or, val_loader_or,
                            optimizer_sgd, criterion,
                            num_epochs=epochs)

# %%
# Model for Adam
mlp_adam = SimpleMLP(FLATTENED_DIM_OR, HIDDEN_DIM1, HIDDEN_DIM2, N_CLASSES_OR, dropout_rate=0.0).to(device)
optimizer_adam = optim.Adam(mlp_adam.parameters(), lr=0.001) # Adam often uses smaller LR
history_adam = train_and_validate_model(mlp_adam, "MLP with Adam",
                                      train_loader_or, val_loader_or,
                                      optimizer_adam, criterion,
                                      num_epochs=epochs)

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_sgd['val_loss'], label='SGD+Momentum Val Loss')
plt.plot(history_adam['val_loss'], label='Adam Val Loss')
plt.title('Validation Loss: SGD vs Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_sgd['val_accuracy'], label='SGD+Momentum Val Acc')
plt.plot(history_adam['val_accuracy'], label='Adam Val Acc')
plt.title('Validation Accuracy: SGD vs Adam')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
print("Adam often converges faster or to a better minimum than basic SGD.")

# %% [markdown]
# #### Adam from scratch vs `torch.optim.Adam`
# The [Adam update](README.md#adam-optimizer) keeps running averages of the gradient ($m_t$) and squared gradient ($v_t$), corrects their bias towards zero, and takes a step $\theta \leftarrow \theta - \eta \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$. Running our version and PyTorch's side by side on the same loss should give identical parameters.

# %%
from mlf_utils import check_close

gen = torch.Generator().manual_seed(0)       # local generator: doesn't disturb the global RNG
target = torch.randn(5, generator=gen)
w_torch = torch.zeros(5, requires_grad=True)
adam = optim.Adam([w_torch], lr=0.1, betas=(0.9, 0.999), eps=1e-8)

w_manual, m, v = torch.zeros(5), torch.zeros(5), torch.zeros(5)
lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
for t in range(1, 21):
    adam.zero_grad()
    ((w_torch - target) ** 2).sum().backward()
    adam.step()

    g = 2 * (w_manual - target)                 # gradient of sum((w - target)^2)
    m = beta1 * m + (1 - beta1) * g             # first moment
    v = beta2 * v + (1 - beta2) * g ** 2        # second moment
    m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)   # bias correction
    w_manual = w_manual - lr * m_hat / (v_hat.sqrt() + eps)

check_close("Manual Adam vs torch.optim.Adam after 20 steps", w_manual, w_torch, atol=1e-6)

# %% [markdown]
# ### Learning Rate Scheduling
# <p>Adjusting the learning rate during training can improve performance and convergence.</p>
#
# - Start with a larger LR for faster initial progress.
# - Gradually decrease LR as training progresses to allow for finer adjustments and avoid overshooting the minimum.

# %% [markdown]
# 1. StepLR : Decays the learning rate by a factor (gamma) every `step_size` epochs.
# 2. ReduceLROnPlateau : Reduces learning rate when a metric (e.g. validation loss) has stopped improving.

# %%
mlp_lr_scheduler = SimpleMLP(FLATTENED_DIM_OR, HIDDEN_DIM1, HIDDEN_DIM2, N_CLASSES_OR).to(device)
optimizer_lr_scheduler = optim.Adam(mlp_lr_scheduler.parameters(), lr=0.001) # Initial LR

# %%
# decay lr by factor of 0.1 every 7 epochs
scheduler_step = lr_scheduler.StepLR(optimizer_lr_scheduler, step_size=7, gamma=0.1)

# %%
print("\nTraining MLP with Adam and StepLR Scheduler...")
history_lr_scheduler = train_and_validate_model(
    mlp_lr_scheduler, "MLP with Adam + StepLR",
    train_loader_or, val_loader_or,
    optimizer_lr_scheduler, criterion,
    scheduler_instance=scheduler_step,
    num_epochs=20 
)

# %% [markdown]
# ### Regularization Techniques
# <p>Regularization helps prevent overfitting by adding constraints or penalties to the learning algorithm.</p>

# %%
# L2 Regularization (Weight decay)
mlp_l2_reg = SimpleMLP(FLATTENED_DIM_OR, HIDDEN_DIM1, HIDDEN_DIM2, N_CLASSES_OR, dropout_rate=0.0).to(device) # No dropout yet
WEIGHT_DECAY_VALUE = 1e-4 # This is the lambda for L2 penalty
optimizer_l2_reg = optim.Adam(mlp_l2_reg.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY_VALUE)

# %%
print(f"\nTraining MLP with Adam and L2 Regularization (Weight Decay={WEIGHT_DECAY_VALUE})...")
history_l2_reg = train_and_validate_model(
    mlp_l2_reg, f"MLP with L2 (wd={WEIGHT_DECAY_VALUE})",
    train_loader_or, val_loader_or,
    optimizer_l2_reg, criterion,
    num_epochs=epochs 
)

# %%
# Dropout
DROPOUT_RATE = 0.3 # e.g., 30% of neurons dropped
mlp_dropout = SimpleMLP(FLATTENED_DIM_OR, HIDDEN_DIM1, HIDDEN_DIM2, N_CLASSES_OR, dropout_rate=DROPOUT_RATE).to(device)
optimizer_dropout = optim.Adam(mlp_dropout.parameters(), lr=0.001)

# %%
print(f"\nTraining MLP with Adam and Dropout (p={DROPOUT_RATE})...")
history_dropout = train_and_validate_model(
    mlp_dropout, f"MLP with Dropout (p={DROPOUT_RATE})",
    train_loader_or, val_loader_or,
    optimizer_dropout, criterion,
    num_epochs=epochs 
)

# %%
# --- Compare Regularization Effects ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_adam['val_loss'], label='Adam (No Reg) Val Loss') # From optimizer demo section
plt.plot(history_l2_reg['val_loss'], label=f'Adam + L2 (wd={WEIGHT_DECAY_VALUE}) Val Loss')
plt.plot(history_dropout['val_loss'], label=f'Adam + Dropout (p={DROPOUT_RATE}) Val Loss')
plt.title('Validation Loss with Regularization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_adam['val_accuracy'], label='Adam (No Reg) Val Acc')
plt.plot(history_l2_reg['val_accuracy'], label=f'Adam + L2 (wd={WEIGHT_DECAY_VALUE}) Val Acc')
plt.plot(history_dropout['val_accuracy'], label=f'Adam + Dropout (p={DROPOUT_RATE}) Val Acc')
plt.title('Validation Accuracy with Regularization')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
