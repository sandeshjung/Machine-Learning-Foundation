# Cheat Sheet: Generative Models

Notebooks: [vae](vae.ipynb) · [gan](gan.ipynb) · Theory: [README](README.md)

## Variational Autoencoder (VAE)
- **Encoder** $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}\,\sigma^2_\phi(x))$ · **Decoder** $p_\theta(x \mid z)$ · **Prior** $p(z) = \mathcal{N}(0, I)$
- **ELBO** (maximise), or equivalently minimise its negative:
$$\log p(x) \ge \underbrace{\mathbb{E}_{q}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{KL}\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{regulariser}}$$
- **Closed-form KL** (Gaussian encoder vs standard normal): $D_{KL} = -\frac{1}{2} \sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$
- **Reparameterisation:** $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, so gradients flow through $\mu$ and $\sigma$.
- **Reconstruction loss:** BCE for $[0,1]$ pixels (sigmoid output), MSE for real-valued data (Gaussian decoder).
- **β-VAE:** loss $= \text{recon} + \beta \cdot \text{KL}$. $\beta > 1$ gives more disentangled but blurrier results.

## Generative Adversarial Network (GAN)
- **Minimax game:** $\min_G \max_D \; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$
- The optimal discriminator is $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$. At the global optimum $p_g = p_{data}$ and $D^* = \frac{1}{2}$.
- **Non-saturating generator loss:** maximise $\log D(G(z))$ (train $G$ with "real" labels). Its gradients are much stronger early on.
- **Training loop:** (1) a $D$ step on real and **detached** fake images, then (2) a $G$ step through $D$.
- **DCGAN recipe:** strided convolutions (no pooling), BatchNorm, ReLU in $G$ / LeakyReLU in $D$, `tanh` output with data in $[-1, 1]$, Adam with `lr=2e-4`, `betas=(0.5, 0.999)`.

## VAE vs GAN
| | VAE | GAN |
|---|---|---|
| Training | Stable, single loss | Adversarial, can be unstable |
| Samples | Blurrier | Sharp |
| Likelihood / encoder | Lower bound on $\log p(x)$, has an encoder | No likelihood, no encoder |
| Latent space | Smooth, easy to interpolate | Less structured |
| Failure mode | Posterior collapse | Mode collapse, oscillation |

## Pitfalls
- **Scale the KL term consistently with the reconstruction term** (both summed per sample, then averaged over the batch). Shrinking KL turns a VAE into a plain autoencoder, and then sampling from the prior produces garbage.
- Predict $\log\sigma^2$, not $\sigma$, because it's unconstrained and numerically safe.
- GAN losses don't tell you sample quality. Look at samples from a **fixed noise vector** over time, or use FID.
- Mode collapse: try label smoothing, minibatch discrimination, a WGAN-GP loss, or spectral normalisation.
- Match the data range to the generator output: images normalised to $[-1, 1]$ for a `tanh` output.
- Put `generator.eval()` or `torch.no_grad()` around sampling, and `.detach()` fakes in the discriminator step.
