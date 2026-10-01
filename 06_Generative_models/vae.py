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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/vae.ipynb)

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
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Variational Autoencoders (VAEs)
# <p>VAEs are a type of generative model. Unlike standard autoencoders that learn a deterministic mapping to a latent space, VAEs learn a probability distribution for that latent space.</p>
# <p>Goal:</p>
#
# - Learn a good latent representation.
# - Be able to generate new data samples similar to the training data by sampling from the latent space and passing it through the decoder.
#
# <p>VAEs are trained by maximizing the Evidence Lower Bound (ELBO) on the log-likelihood of the data.</p>

# %% [markdown]
# Theory: [Variational Autoencoders (VAEs)](README.md#2-variational-autoencoders-vaes) · [ELBO](README.md#24-the-objective-the-elbo)
#
# ### Data: Fashion-MNIST
# Pixels stay in $[0, 1]$ (no normalisation) because the reconstruction loss is binary cross-entropy against a sigmoid output.

# %%
transform_vae = transforms.Compose([
    transforms.ToTensor()
])

# %%
batch_size = 128

# %%
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_vae)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_vae)

# %%
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# %%
img_channels, img_height, img_width = train_dataset[0][0].shape
img_channels, img_height, img_width 

# %%
flatten_dim = img_channels * img_height * img_width
flatten_dim

# %% [markdown]
# ### Visualisation helper
# `show_images` from the shared [`mlf_utils`](../mlf_utils/plotting.py) package arranges a batch into a grid.

# %%
from mlf_utils import show_images

# %%
dataiter_vae = iter(train_loader)
images_sample, _ = next(dataiter_vae)
show_images(images_sample, num_images=8, title="Sample Training Images")

# %% [markdown]
# ## VAE model definition (Fully Connected)
# <p>Architecture:
#
# - Encoder: Input -> FC Layers -> mu (mean), log_var (log variance) of latent distribution
# - Reparameterization Trick: z = mu + sigma * epsilon
# - Decoder: Latent z -> FC Layers -> Reconstructed Output</p>

# %% [markdown]
# ### Latent dimension
# A 2-D latent space gives blurrier reconstructions, but it can be plotted directly. Switch to `latent_dim = 20` for sharper results.

# %%
# latent_dim = 20
latent_dim = 2      # train with 2 to visualize 2D Latent space


# %% [markdown]
# The encoder outputs $\mu$ and $\log \sigma^2$ (log-variance is unconstrained and numerically safer than $\sigma$). The **reparameterisation trick** $z = \mu + \sigma \odot \epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$, moves the randomness into $\epsilon$ so gradients can flow through $\mu$ and $\sigma$.

# %%
class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim1)
        self.encoder_fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        # two output layers from encoder: one for mean (mu), one for log_variance (log_var)
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim2, latent_dim)

        # decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_fc2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decoder_fc_out = nn.Linear(hidden_dim1, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()     # output layer to match input [0,1] range

    def encode(self, x_flat):
        # x_flat -> [batch_size, input_dim]
        h1 = self.relu(self.encoder_fc1(x_flat))
        h2 = self.relu(self.encoder_fc2(h1))
        mu = self.fc_mu(h2)
        log_var = self.fc_log_var(h2)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        # z -> [batch_size, latent_dim]
        h3 = self.relu(self.decoder_fc1(z))
        h4 = self.relu(self.decoder_fc2(h3))
        x_reconstructed_logits = self.decoder_fc_out(h4)
        x_reconstructed = self.sigmoid(x_reconstructed_logits)
        return x_reconstructed

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)     # flatten input image
        mu, log_var = self.encode(x_flat)
        z_sampled = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z_sampled)
        return x_reconstructed, mu, log_var, z_sampled


# %%
hidden_dim1 = 400
hidden_dim2 = 100

# %%
vae_model = VAE(flatten_dim, hidden_dim1, hidden_dim2, latent_dim).to(device)

# %%
print(vae_model)


# %% [markdown]
# ### VAE Loss Function (Negative ELBO)
#
# <span>Loss = Reconstruction Loss + KL Divergence</span>

# %% [markdown]
# For a Gaussian encoder and a standard normal prior, the KL term has a closed form:
# $$D_{KL}\big(q(z|x) \,\|\, \mathcal{N}(0, I)\big) = -\frac{1}{2} \sum_{j} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
# The reconstruction term pulls outputs towards the input, and the KL term keeps the latent space smooth and close to the prior, which is what makes sampling possible.

# %%
def vae_loss_function(x_reconstructed, x_original_flat, mu, log_var):
    # Reconstruction Loss (BCE for [0,1] pixel values)
    reconstruction_loss = F.binary_cross_entropy(x_reconstructed, x_original_flat, reduction="sum") / x_original_flat.size(0)
    # D_KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) across latent dimensions
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_divergence /= x_original_flat.size(0)    # average over the batch, like the reconstruction term
    # Total VAE Loss = Reconstruction Loss + KL Divergence
    total_loss = reconstruction_loss + kl_divergence
    return total_loss, reconstruction_loss, kl_divergence


# %% [markdown]
# ### Verifying the closed-form KL
# `torch.distributions.kl_divergence` computes the same quantity for two `Normal` distributions. Summing over latent dimensions and averaging over the batch should match our loss term.

# %%
from torch.distributions import Normal, kl_divergence
from mlf_utils import check_close

gen = torch.Generator().manual_seed(0)
mu_test, log_var_test = torch.randn(2, 4, latent_dim, generator=gen).unbind(0)
x_dummy = torch.rand(4, flatten_dim, generator=gen)
_, _, kl_ours = vae_loss_function(x_dummy, x_dummy, mu_test, log_var_test)

q_z = Normal(mu_test, torch.exp(0.5 * log_var_test))
p_z = Normal(torch.zeros_like(mu_test), torch.ones_like(mu_test))
check_close("KL: closed form vs torch.distributions", kl_ours, kl_divergence(q_z, p_z).sum(dim=1).mean(), atol=1e-5)

# %% [markdown]
# ### Training

# %%
optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

# %%
optimizer

# %%
num_epochs = 15
train_total_losses = []
train_recon_losses = []
train_kl_divs = []

for epoch in range(num_epochs):
    vae_model.train()
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_kl_div = 0.0

    for images, _ in train_loader:      # labels not used for VAE training
        images_flat = images.view(images.size(0), -1).to(device)

        optimizer.zero_grad()
        reconstructions, mu, log_var, _ = vae_model(images_flat) 

        total_loss, recon_loss, kl_div = vae_loss_function(reconstructions, images_flat, mu, log_var)

        total_loss.backward()
        optimizer.step()

        running_total_loss += total_loss.item() * images.size(0)
        running_recon_loss += recon_loss.item() * images.size(0)
        running_kl_div += kl_div.item() * images.size(0)

    epoch_total_loss = running_total_loss / len(train_loader.dataset)
    epoch_recon_loss = running_recon_loss / len(train_loader.dataset)
    epoch_kl_div = running_kl_div / len(train_loader.dataset)

    train_total_losses.append(epoch_total_loss)
    train_recon_losses.append(epoch_recon_loss)
    train_kl_divs.append(epoch_kl_div)

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_total_loss:.4f}, "
          f"Recon Loss: {epoch_recon_loss:.4f}, KL Div: {epoch_kl_div:.4f}")

# %% [markdown]
# ### Loss curves
# The total loss is the negative ELBO, shown alongside its reconstruction and KL components.

# %%
plt.figure(figsize=(10, 5))
plt.plot(train_total_losses, label='Total Loss (-ELBO)')
plt.plot(train_recon_losses, label='Reconstruction Loss (BCE)')
plt.plot(train_kl_divs, label='KL Divergence')
plt.title("VAE Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ### Reconstructions
# Test images (top) and their reconstructions (bottom).

# %%
# Visualizing VAE Results
vae_model.eval()
test_dataiter_vae_viz = iter(test_loader)
test_images_vae_viz, _ = next(test_dataiter_vae_viz)
test_images_flat_viz = test_images_vae_viz.view(test_images_vae_viz.size(0), -1).to(device)

with torch.no_grad():
    reconstructed_images_vae_viz, _, _, _ = vae_model(test_images_flat_viz)

# %%
reconstructed_images_vae_viz = reconstructed_images_vae_viz.view(-1, img_channels, img_height, img_width).cpu()
test_images_to_show_vae = test_images_vae_viz[:10].cpu() # Show first 10

# %%
fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
for i in range(10):
    show_images(test_images_to_show_vae[i], ax=axes[0, i], show=False)
    show_images(reconstructed_images_vae_viz[i], ax=axes[1, i], show=False)
axes[0, 0].set_title("Original", loc="left")
axes[1, 0].set_title("Reconstructed", loc="left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Generating new images
# Sample $z \sim \mathcal{N}(0, I)$ from the prior and decode it. No encoder is involved.

# %%
print("\n--- Generating New Images from Latent Space Samples ---")
vae_model.eval()
num_generated_samples = 8
with torch.no_grad():
    # Sample z from N(0, I)
    latent_samples_z = torch.randn(num_generated_samples, latent_dim).to(device)
    generated_images = vae_model.decode(latent_samples_z) # Pass through decoder
    generated_images = generated_images.view(-1, img_channels, img_height, img_width).cpu()

show_images(generated_images, title="Generated Images from VAE Latent Space", num_images=num_generated_samples)

# %% [markdown]
# ### The 2-D latent space
# Each test image is plotted at its encoded mean $\mu$, coloured by class. Similar garments cluster together even though the VAE never saw the labels.

# %%
# Visualizing 2D Latent Space 
if latent_dim == 2:
    print(f"\n--- Visualizing 2D Latent Space (VAE) ---")
    vae_model.eval()
    all_latent_mu = []
    all_labels_viz = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i * batch_size > 1000: break # Limit to ~1000 samples for viz
            images_flat = images.view(images.size(0), -1).to(device)
            mu, log_var = vae_model.encode(images_flat) 
            all_latent_mu.append(mu.cpu())
            all_labels_viz.append(labels.cpu())
                
    all_latent_mu_tensor = torch.cat(all_latent_mu, dim=0)
    all_labels_tensor_viz = torch.cat(all_labels_viz, dim=0)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_latent_mu_tensor[:, 0].numpy(), all_latent_mu_tensor[:, 1].numpy(),
                          c=all_labels_tensor_viz.numpy(), cmap='tab10', alpha=0.6, s=10)
    plt.xlabel("Latent Dimension 1 (mu_1)")
    plt.ylabel("Latent Dimension 2 (mu_2)")
    class_names_fm = test_dataset.classes 
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names_fm, title="Classes")
    plt.title("2D Latent Space (Mean Vectors μ) of VAE for FashionMNIST"); plt.show()
else:
    print(f"\nLatent space dimension is {latent_dim}. Visualization of latent space is for 2D.")

# %% [markdown]
# ### Try it: walk the latent space
# Pick a point $z = (z_1, z_2)$ and decode it. Because the KL term keeps the latent space close to $\mathcal{N}(0, I)$, nearby points decode to similar garments and moving across the plane morphs smoothly between classes. Compare with the 2-D scatter above to see which region is which.
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
from ipywidgets import FloatSlider, interact

if latent_dim == 2:
    @interact(z1=FloatSlider(value=0.0, min=-3, max=3, step=0.1, continuous_update=False),
              z2=FloatSlider(value=0.0, min=-3, max=3, step=0.1, continuous_update=False))
    def explore_latent(z1, z2):
        with torch.no_grad():
            img = vae_model.decode(torch.tensor([[z1, z2]], device=device)).view(1, img_channels, img_height, img_width)
        fig, ax = plt.subplots(figsize=(3, 3))
        show_images(img, ax=ax, show=False, title=f"z = ({z1:.1f}, {z2:.1f})")
        plt.show()
else:
    print("Set latent_dim = 2 above to explore the latent space with sliders.")

# %%
