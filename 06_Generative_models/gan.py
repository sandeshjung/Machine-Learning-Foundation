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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/gan.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Generative Adversarial Networks (GANs)
# <p>GANs are a class of generative models that learn to generate new data with the same statistics as the training set.</p>
# <p>Core Idea: Two neural networks, a Generator (G) and a Discriminator (D), are trained simulataneously in a "game-like" (adversarial) setting</p>

# %% [markdown]
# Theory: [Generative Adversarial Networks (GANs)](README.md#3-generative-adversarial-networks-gans)
#
# ### Hyperparameters
# `lr = 2e-4` and Adam with `beta1 = 0.5` follow the DCGAN paper's recommendations for stable adversarial training.

# %%
# Hyperparameters
img_channels = 1
img_size = 28
flattened_dim = img_size * img_size
latent_dim = 100
batch_size = 128
lr = 0.0002 
beta1 = 0.5
num_epochs = 50

# %% [markdown]
# ### Data: Fashion-MNIST
# Images are normalised to $[-1, 1]$ to match the generator's `tanh` output range.

# %%
transform_gan = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))   # Normalizes [0,1] to [-1,1]
])

# %%
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_gan)

# %%
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# %%
len(train_dataset), train_dataset[0][0].shape

# %% [markdown]
# ### Visualisation helper
# `show_images` from the shared [`mlf_utils`](../mlf_utils/plotting.py) package arranges a batch into a grid. `unnormalize=True` maps the $[-1, 1]$ images back to $[0, 1]$.

# %%
from mlf_utils import show_images

# %%
dataiter = iter(train_loader)
images_batch, _ = next(dataiter)
show_images(images_batch, title="Sample Training Images", num_images=16, nrow=4, unnormalize=True)


# %% [markdown]
# ## GAN Architecture
# A DCGAN (Deep Convolution GAN) would use convolutional layers and typically perform better for images.

# %%
# Generator Network (G)
# Takes random noise z (from latent_dim) and outputs a fake image (flattened_dim).
class GeneratorMLP(nn.Module):
    
    def __init__(self, latent_dim, img_flat_dim, hidden_dim1=256, hidden_dim2=512):
        super(GeneratorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU often used in GANs
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim2, img_flat_dim),
            nn.Tanh() # Output in [-1, 1] to match normalized image data
        )
        self.img_shape = (img_channels, img_size, img_size)

    def forward(self, z_noise):
        # z_noise: [batch_size, latent_dim]
        img_flat = self.model(z_noise)
        # Reshape flat output to image dimensions [batch_size, C, H, W]
        img_reshaped = img_flat.view(img_flat.size(0), *self.img_shape)
        return img_reshaped


# %%
# Discriminator Network (D)
# Takes an image (flattened) and outputs a scalar probability (0 for fake, 1 for real).
class DiscriminatorMLP(nn.Module):
    def __init__(self, img_flat_dim, hidden_dim1=512, hidden_dim2=256):
        super(DiscriminatorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_flat_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), # Dropout can help Discriminator generalize
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, 1) 
        )

    def forward(self, img_flat):
        # img_flat: [batch_size, img_flat_dim]
        validity_logit = self.model(img_flat)
        return validity_logit 


# %%
generator = GeneratorMLP(latent_dim, flattened_dim).to(device)
discriminator = DiscriminatorMLP(flattened_dim).to(device)

# %%
generator

# %%
discriminator

# %%
# test with dummy noise for generator
dummy_noise = torch.randn(batch_size, latent_dim).to(device)
with torch.no_grad():
    dummy_fake_imgs = generator(dummy_noise)
dummy_fake_imgs.shape

# %%
# test discriminator
with torch.no_grad():
    dummy_validity = discriminator(dummy_fake_imgs.view(batch_size, -1))
dummy_validity.shape

# %% [markdown]
# ### Loss and optimizers
# `BCEWithLogitsLoss` combines the sigmoid and binary cross-entropy in one numerically stable operation, so the discriminator outputs raw logits. $G$ and $D$ each get their own optimizer.

# %%
# Loss function and optimizer
adversarial_loss = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# %% [markdown]
# ### Training loop
# For each batch:
# 1. **Discriminator step:** maximise $\log D(x) + \log(1 - D(G(z)))$. Fake images are `detach()`ed so this step doesn't update $G$.
# 2. **Generator step:** use the **non-saturating** loss, maximising $\log D(G(z))$ (i.e. labelling fakes as real). This gives stronger gradients early in training than minimising $\log(1 - D(G(z)))$.
#
# A fixed noise vector is reused to track how samples evolve across epochs.

# %%
G_losses = []
D_losses = []
fixed_noise_for_sampling = torch.randn(16, latent_dim).to(device)

for epoch in range(num_epochs):
    epoch_D_loss = 0.0
    epoch_G_loss = 0.0
    num_batches_gan = 0

    for i, (real_imgs_batch, _) in enumerate(train_loader): # Labels not needed for GAN training
        current_batch_size = real_imgs_batch.size(0)
        real_imgs_batch = real_imgs_batch.to(device)
        
        # Create labels for real (1) and fake (0) images
        real_labels = torch.ones(current_batch_size, 1, device=device, dtype=torch.float32)
        fake_labels = torch.zeros(current_batch_size, 1, device=device, dtype=torch.float32)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        real_imgs_flat = real_imgs_batch.view(current_batch_size, -1)
        d_output_real_logits = discriminator(real_imgs_flat)
        d_loss_real = adversarial_loss(d_output_real_logits, real_labels)
        
        # Loss for fake images
        z_noise = torch.randn(current_batch_size, latent_dim, device=device)
        fake_imgs_batch = generator(z_noise) # Output shape [batch, C, H, W]
        
        # Detach fake_imgs_batch so G's gradients are not computed during D's update
        d_output_fake_logits = discriminator(fake_imgs_batch.detach().view(current_batch_size, -1))
        d_loss_fake = adversarial_loss(d_output_fake_logits, fake_labels)
        
        # Total discriminator loss
        d_loss_total = d_loss_real + d_loss_fake
        d_loss_total.backward()
        optimizer_D.step()
        
        epoch_D_loss += d_loss_total.item()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        d_output_on_fake_for_G_logits = discriminator(fake_imgs_batch.view(current_batch_size, -1)) 
        g_loss = adversarial_loss(d_output_on_fake_for_G_logits, real_labels) # G tries to fool D
        
        g_loss.backward()
        optimizer_G.step()

        epoch_G_loss += g_loss.item()
        num_batches_gan +=1

    avg_epoch_D_loss = epoch_D_loss / num_batches_gan
    avg_epoch_G_loss = epoch_G_loss / num_batches_gan
    D_losses.append(avg_epoch_D_loss)
    G_losses.append(avg_epoch_G_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], D_Loss: {avg_epoch_D_loss:.4f}, G_Loss: {avg_epoch_G_loss:.4f}")

    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
        generator.eval() # Set generator to eval mode for sampling
        with torch.no_grad():
            sampled_imgs = generator(fixed_noise_for_sampling)
        show_images(sampled_imgs, title=f"Generated Images - Epoch {epoch+1}", num_images=16, nrow=4, unnormalize=True)
        generator.train() # Set back to train mode

# %% [markdown]
# ### Loss curves
# GAN losses don't decrease monotonically like a supervised loss. Roughly stable, oscillating curves usually mean the two players are balanced.

# %%
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.title("GAN Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ### Samples from the trained generator

# %%
generator.eval()
with torch.no_grad():
    final_noise = torch.randn(64, latent_dim).to(device) 
    final_generated_images = generator(final_noise)
show_images(final_generated_images, title="Final Generated Samples", num_images=32, nrow=8, unnormalize=True)

# %% [markdown]
# ### Common GAN Challenges
# - Mode Collapse: Generator produces very limited variety of samples.
# - Non-Convergence/Oscillations: D and G losses may not smoothly converge; they might oscillate.
# - Vanishing Gradients for G: If D gets too good too quickly, G struggles to learn.
# - Hyperparameter Sensitivity: GANs can be very sensitive to learning rates, batch sizes, network architecture.
# - Evaluation: Quantitatively evaluating GANs is challenging (FID, IS scores are common).
#
# **Tips**:
# - Use LeakyReLU instead of ReLU in Discriminator.
# - Use Tanh for Generator's output if data is normalized to [-1,1], Sigmoid if [0,1].
# - Normalize input data (e.g., to [-1,1]).
# - Use Adam optimizer with specific betas (e.g., beta1=0.5).
# - Label smoothing for Discriminator targets (e.g., real labels 0.9 instead of 1.0).
# - Careful architectural choices (e.g., DCGAN for images).

# %%
