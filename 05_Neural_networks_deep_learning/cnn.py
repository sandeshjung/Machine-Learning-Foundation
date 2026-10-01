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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/cnn.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, random_split 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Convolutional Neural Networks (CNNs)
# <p>CNNs are a class of deep neural networks most commonly applied to analyzing visual imagery.</p>

# %%
transform_cifar = transforms.Compose([
    transforms.ToTensor(),      # Convert to tensor and scale [0,1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))     # Normalize to [-1,1] for 3 channels
])

# %%
batch = 64
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar)


# %%
classes = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')
nclasses = len(classes)

# %%
ninput = 3
img_size = 32

# %%
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train, val = random_split(train_dataset, [train_size, val_size])

# %%
len(train), len(val)

# %%
# dataloaders
train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
val_loader = DataLoader(val, batch_size=batch, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=2)

# %%
len(train_loader), len(val_loader), len(test_loader)

# %%
from mlf_utils import check_close, show_images

# %%
dataiter = iter(train_loader)
images, labels = next(dataiter)

# %%
show_images(images[:8], title="Sample Training Images", unnormalize=True)


# %% [markdown]
# ## CNN Architecture
# <p><b>We will define a simple CNN</b></p>
# Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU -> MaxPool2 -> Flatten -> FC1 -> ReLU -> FC2 (output)

# %%
class sCNN(nn.Module):
    def __init__(self, ninput, nclasses, img_size):
        super(sCNN, self).__init__()
        # conv layer 1
        self.conv1 = nn.Conv2d(in_channels=ninput, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv layer 2
        # input size after pool1: img_size / 2 -> 16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layers
        # input size after pool2: img_size / 4 -> 8x8
        self.fc_input_size = self._get_conv_output_size(ninput, img_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, nclasses)     # output layer (logits)

    def _get_conv_output_size(self, ninput, img_size):
        dummy_input = torch.zeros(1, ninput, img_size, img_size)
        with torch.no_grad():
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# %%
model = sCNN(ninput=ninput, nclasses=nclasses, img_size=img_size).to(device)

# %%
print(model)

# %%
# test with a dummy input batch to check shapes
dummy = torch.randn(batch, ninput, img_size, img_size).to(device)

# %%
with torch.no_grad():
    dummy_output = model(dummy)
dummy_output.shape      # [batch, nclasses]


# %% [markdown]
# ### Checking the output-size formula
# For every convolution and pooling layer, $\text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1$ (see [Output Size Calculation](README.md#23-output-size-and-parameter-count)). Forward hooks record each layer's actual input and output width, so the formula is checked layer by layer.

# %%
def expected_size(size, layer):
    k, s, p = [v if isinstance(v, int) else v[0] for v in (layer.kernel_size, layer.stride, layer.padding)]
    return (size + 2 * p - k) // s + 1

spatial_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d))]
recorded = {}
hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: recorded.__setitem__(n, (inp[0].shape[-1], out.shape[-1])))
         for n, m in spatial_layers]
with torch.no_grad():
    model(dummy[:1])
for h in hooks:
    h.remove()

for n, m in spatial_layers:
    in_size, out_size = recorded[n]
    check_close(f"{n} ({type(m).__name__}): {in_size} -> {out_size}", expected_size(in_size, m), out_size)

# %%
# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
criterion, optimizer

# %%
epochs = 15
train_loss = []
val_loss = []
val_accuracy = []
for epoch in range(epochs):
    model.train()
    train_losses = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs_logits = model(inputs)
        loss = criterion(outputs_logits, labels)    # expect logits
        # backward pass and optimize
        loss.backward()
        optimizer.step()

        train_losses += loss.item() * inputs.size(0)  # accumulate loss 

    epoch_train_loss = train_losses / len(train_dataset)
    train_loss.append(epoch_train_loss)

    # validation phase
    model.eval()
    val_losses = 0.0
    correct_val_preds = 0
    total_val_samples = 0
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_logits_val = model(inputs_val)
            loss_val = criterion(outputs_logits_val, labels_val)
            val_losses += loss_val.item() * inputs_val.size(0)

            # get predictions
            _, predicted_classes_val = torch.max(outputs_logits_val, 1)
            correct_val_preds += (predicted_classes_val == labels_val).sum().item()
            total_val_samples += labels_val.size(0)

    epoch_val_loss = val_losses / len(val)
    val_loss.append(epoch_val_loss)
    epoch_val_accuracy = correct_val_preds / total_val_samples
    val_accuracy.append(epoch_val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {epoch_train_loss:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, "
          f"Val Acc: {epoch_val_accuracy:.4f}")

# %%
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_loss, color=color, linestyle='-', label='Train Loss')
ax1.plot(val_loss, color=color, linestyle='--', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(val_accuracy, color=color, linestyle=':', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('CNN: Training/Validation Loss & Accuracy')
plt.show()

# %%
# Evaluate CNN on Test set
model.eval()
test_correct = 0
test_total = 0
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for images_test, labels_test in test_loader:
        images_test, labels_test = images_test.to(device), labels_test.to(device)
        
        outputs_logits_test = model(images_test)
        _, predicted_test = torch.max(outputs_logits_test.data, 1)
        
        test_total += labels_test.size(0)
        test_correct += (predicted_test == labels_test).sum().item()
        
        all_test_preds.extend(predicted_test.cpu().numpy())
        all_test_labels.extend(labels_test.cpu().numpy())


# %%
test_accuracy = test_correct / test_total 
test_accuracy * 100, test_total

# %%
# confusion matrix and classification report
cm = confusion_matrix(all_test_labels, all_test_preds)
report_cnn = classification_report(all_test_labels, all_test_preds, target_names=classes, zero_division=0) 

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CNN')
plt.show()

print("\nClassification Report - CNN:\n", report_cnn)

# %%

# %%
