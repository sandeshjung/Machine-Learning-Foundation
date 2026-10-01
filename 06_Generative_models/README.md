# 06 · Generative Models

Most models in this repo answer "what is this?". **Generative models** answer "what else could exist?" They learn the distribution of the data, $p(\mathbf{x})$, well enough to **create new samples** that look like the training data. This module covers the two classic deep generative models: **variational autoencoders (VAEs)** and **generative adversarial networks (GANs)**.

> **Notebooks:** [vae](vae.ipynb) · [gan](gan.ipynb). Both train on Fashion-MNIST.
>
> **Quick revision:** [CHEATSHEET.md](CHEATSHEET.md)

## Contents

1. [What a generative model does](#1-what-a-generative-model-does)
2. [Variational autoencoders (VAEs)](#2-variational-autoencoders-vaes)
3. [Generative adversarial networks (GANs)](#3-generative-adversarial-networks-gans)
4. [VAE vs GAN](#4-vae-vs-gan)

---

## 1. What a generative model does

Given training data $\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)}$ drawn from an unknown distribution $p_{\text{data}}$, a generative model learns $p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$. With it you can:

- **generate** new samples $\mathbf{x}_{\text{new}} \sim p_\theta$
- **evaluate** how likely a data point is (density estimation)
- **fill in** missing parts of the data
- **learn representations**: compact codes that capture the structure of the data

<p align="center">
  <img src="assets/taxonomy.png" alt="Taxonomy of generative models" width="680">
  <br>
  <em>Families of generative models. VAEs approximate the density; GANs learn to sample without ever writing it down.</em>
</p>

The classic training objective is **maximum likelihood**: make the training data as probable as possible.

```math
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta\big(\mathbf{x}^{(i)}\big)
```

---

## 2. Variational autoencoders (VAEs)

### 2.1 From autoencoders to VAEs

A plain **autoencoder** compresses the input to a small code $\mathbf{h}$ and reconstructs it:

```math
\mathbf{x} \xrightarrow{\text{encoder}} \mathbf{h} \xrightarrow{\text{decoder}} \hat{\mathbf{x}}, \qquad \mathcal{L} = \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert^2
```

It's good at compression but **bad at generating**. The codes are scattered with gaps between them, so decoding a random point usually gives garbage.

A **VAE** makes everything probabilistic:

- The **encoder** outputs a *distribution* $q_\phi(\mathbf{z} \mid \mathbf{x})$ instead of a single point.
- The **decoder** models $p_\theta(\mathbf{x} \mid \mathbf{z})$.
- A **prior** $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$ pulls all the codes into one smooth, gap-free region.

So you can generate by sampling $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ and decoding it.

<p align="center">
  <img src="assets/vae.png" alt="VAE architecture: encoder, latent distribution, decoder" width="680">
  <br>
  <em>The encoder predicts a mean and variance; a latent z is sampled from them and decoded.</em>
</p>

### 2.2 The latent variable model

A VAE assumes each data point is generated in two steps: first pick a latent code $\mathbf{z}$, then generate $\mathbf{x}$ from it:

```math
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}
```

The problem is that this integral, and with it the true posterior $p(\mathbf{z} \mid \mathbf{x})$, is **intractable** for a neural-network decoder.

The VAE solution is **variational inference**: train the encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ to *approximate* the true posterior.

### 2.3 The three components

**Encoder** $q_\phi(\mathbf{z} \mid \mathbf{x})$: a diagonal Gaussian whose mean and variance are predicted by a neural network.

```math
q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\big(\mathbf{z};\; \boldsymbol{\mu}_\phi(\mathbf{x}),\; \text{diag}\,\boldsymbol{\sigma}^2_\phi(\mathbf{x})\big)
```

The network outputs $\boldsymbol{\mu}$ and $\log \boldsymbol{\sigma}^2$. Using the log-variance keeps the output unconstrained and numerically safe.

**Decoder** $p_\theta(\mathbf{x} \mid \mathbf{z})$: its form depends on the data.

| Data | Decoder output | Reconstruction loss |
|---|---|---|
| Pixels in [0, 1] (as in the notebook) | Bernoulli probabilities (sigmoid) | Binary cross-entropy |
| Real-valued data | Gaussian mean | Mean squared error |

**Prior** $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$: simple to sample from, and it gives the KL term a closed form. Richer priors, such as Gaussian mixtures, are possible.

### 2.4 The objective: the ELBO

We can't maximise $\log p_\theta(\mathbf{x})$ directly, but we can maximise a **lower bound** on it. Inserting $q_\phi$ and rearranging gives:

```math
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]}_{\text{ELBO}} + \underbrace{D_{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x})\big)}_{\ge 0}
```

Since the KL term is never negative, the **ELBO** (Evidence Lower BOund) is always $\le \log p_\theta(\mathbf{x})$. Pushing the ELBO up improves the model *and* makes $q_\phi$ a better approximation of the posterior.

<p align="center">
  <img src="assets/elbo.png" alt="The ELBO as a lower bound on the log-likelihood" width="560">
  <br>
  <em>The gap between log p(x) and the ELBO is exactly the KL between q and the true posterior.</em>
</p>

The ELBO splits into two terms with clear meanings:

```math
\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big]}_{\text{reconstruct the input well}} - \underbrace{D_{KL}\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\big)}_{\text{keep codes close to the prior}}
```

- The **reconstruction term** wants each code to describe its input precisely.
- The **KL term** wants all the codes to look like $\mathcal{N}(0, I)$, which keeps the latent space smooth so that random samples decode into sensible images.

### 2.5 The reparameterisation trick

To train with gradient descent we need gradients **through the sampling step**, and "draw a random $\mathbf{z}$" has no gradient.

The fix is to move the randomness into a separate noise variable:

```math
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I), \qquad \boldsymbol{\sigma} = \exp\left(\tfrac{1}{2} \log \boldsymbol{\sigma}^2\right)
```

Now $\mathbf{z}$ is a **deterministic, differentiable** function of $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, plus noise that doesn't depend on any parameters. So the gradients flow straight through.

<p align="center">
  <img src="assets/reparam.png" alt="VAE computation graph with the reparameterisation trick" width="680">
  <br>
  <em>The full VAE computation graph. The noise ε enters from outside, so z = μ + σ·ε is differentiable with respect to the encoder; the KL term is computed analytically from μ and σ.</em>
</p>

### 2.6 The loss in code

Training **minimises the negative ELBO**:

```math
\mathcal{L}_{\text{VAE}} = \underbrace{\text{BCE}(\hat{\mathbf{x}}, \mathbf{x})}_{\text{reconstruction}} + \underbrace{\left(-\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)\right)}_{\text{closed-form } D_{KL}(q \,\|\, \mathcal{N}(0, I))}
```

> [!IMPORTANT]
> **Scale the two terms the same way.** The notebook **sums** both over the pixels or latent dimensions and then **averages** over the batch. If the KL is made much smaller than the reconstruction term (for example by averaging it over pixels), the VAE quietly turns into a plain autoencoder and sampling from the prior stops working.

The notebook checks the closed-form KL against `torch.distributions.kl_divergence`.

### 2.7 The notebook's architecture

A fully connected VAE on 28 × 28 Fashion-MNIST images (784 pixels):

```text
Encoder:  784 → 400 → 100 → (μ, log σ²)  each of size d
Decoder:  d   → 100 → 400 → 784 (sigmoid)
```

With $d = 2$ the latent space can be **plotted directly**, and sliders let you walk through it. Use $d = 20$ for sharper reconstructions.

### 2.8 β-VAE

Weight the KL term with a factor $\beta$:

```math
\mathcal{L}_{\beta\text{-VAE}} = \text{reconstruction} + \beta \cdot D_{KL}
```

| β | Effect |
|---|---|
| β < 1 | Sharper reconstructions, but a less regular latent space (sampling gets worse) |
| β = 1 | The standard VAE |
| β > 1 | A smoother, more **disentangled** latent space (each dimension tends to capture one factor), with blurrier reconstructions |

> [!NOTE]
> **Posterior collapse** is a failure mode where a powerful decoder ignores $\mathbf{z}$ entirely and the KL drops to 0. KL annealing (slowly raising β from 0 to 1) helps.

---

## 3. Generative adversarial networks (GANs)

### 3.1 The idea: a game between two networks

GANs (Goodfellow et al., 2014) never write down $p(\mathbf{x})$ at all. Instead, two networks compete:

- The **generator** $G$ turns random noise $\mathbf{z} \sim \mathcal{N}(0, I)$ into fake data $G(\mathbf{z})$. Its job is to fool the discriminator.
- The **discriminator** $D$ looks at a sample and outputs the probability that it is **real**. Its job is to catch fakes.

As $D$ gets better at spotting fakes, $G$ is forced to produce more realistic ones.

<p align="center">
  <img src="assets/gan.png" alt="GAN: generator, discriminator, real and fake samples" width="680">
  <br>
  <em>The generator maps noise to images; the discriminator judges real vs fake.</em>
</p>

This is called **implicit** density modelling. $G$ learns a mapping whose outputs follow $p_{\text{data}}$, without ever computing a probability. That sidesteps the intractable normalising constants that explicit models struggle with.

### 3.2 The minimax objective

```math
\min_G \max_D \; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big] + \mathbb{E}_{\mathbf{z} \sim p_z}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big]
```

- $D$ wants $D(\mathbf{x}) \to 1$ for real data and $D(G(\mathbf{z})) \to 0$ for fakes, which **maximises** $V$.
- $G$ wants $D(G(\mathbf{z})) \to 1$, which **minimises** $V$.

### 3.3 What the game converges to

**The optimal discriminator.** For a fixed $G$ that produces samples with density $p_g$, set the derivative of $V$ with respect to $D(\mathbf{x})$ to zero:

```math
D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}
```

**The generator's real objective.** Plugging $D^*$ back in gives:

```math
V(D^*, G) = -\log 4 + 2 \cdot D_{JS}\big(p_{\text{data}} \,\|\, p_g\big)
```

So training $G$ minimises the **Jensen–Shannon divergence** between the real and generated distributions. Its minimum is at $p_g = p_{\text{data}}$, where $D^*(\mathbf{x}) = \frac{1}{2}$ everywhere: the discriminator can do no better than guessing.

In game-theory terms, this is the **Nash equilibrium**: neither player can improve by changing strategy alone.

### 3.4 The losses in practice

**Discriminator:** ordinary binary cross-entropy, with label 1 for real and 0 for fake.

```math
L_D = -\mathbb{E}_{\mathbf{x}}\big[\log D(\mathbf{x})\big] - \mathbb{E}_{\mathbf{z}}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big]
```

**Generator:** the *non-saturating* loss, which trains $G$ as if its fakes were labelled "real":

```math
L_G = -\mathbb{E}_{\mathbf{z}}\big[\log D(G(\mathbf{z}))\big]
```

> [!NOTE]
> **Why not minimise $\log(1 - D(G(\mathbf{z})))$ directly?** Early in training $D$ rejects fakes easily, so $D(G(\mathbf{z})) \approx 0$. That curve is flat there, so $G$ gets almost no gradient. The non-saturating version gives strong gradients exactly when $G$ is doing badly.

**One training step:**

1. **Update $D$** on a batch of real images and a batch of fakes. **`.detach()`** the fakes so this step doesn't update $G$.
2. **Update $G$** by generating fakes, passing them through $D$, and back-propagating $L_G$ into $G$.

### 3.5 The notebook's architecture

Both networks are small **MLPs** on flattened 28 × 28 Fashion-MNIST images, scaled to [−1, 1]:

```text
Generator:      z (100) → 256 → 512 → 784, then tanh     (LeakyReLU 0.2 between layers)
Discriminator:  784 → 512 → 256 → 1 (logit)               (LeakyReLU 0.2 + dropout 0.3)
```

Training uses Adam with `lr = 2e-4` and `betas = (0.5, 0.999)`, following the DCGAN paper.

### 3.6 DCGAN: convolutional GANs for images

For larger images, the **DCGAN** guidelines (Radford et al., 2015) became the standard recipe:

<p align="center">
  <img src="assets/dcgan.png" alt="DCGAN generator upsampling noise to an image" width="680">
  <br>
  <em>The DCGAN generator: transposed convolutions upsample the noise vector into an image.</em>
</p>

| | Generator | Discriminator |
|---|---|---|
| Resizing | **Transposed** convolutions with stride 2 (upsample) | **Strided** convolutions (downsample), no pooling |
| Normalisation | BatchNorm (not on the output) | BatchNorm (not on the input layer) |
| Activations | ReLU, with **tanh** at the output | **LeakyReLU** (slope 0.2) |
| Output | An image in [−1, 1] | One logit |

### 3.7 Why GANs are hard to train, and the fixes

| Problem | Symptom | Common fixes |
|---|---|---|
| **Mode collapse** | $G$ produces only a few kinds of output | Minibatch discrimination, WGAN losses, unrolled GANs |
| **Vanishing gradients** | $D$ wins too easily, so $G$ stops improving | Non-saturating loss, label smoothing, a weaker $D$ |
| **Oscillation** | The losses swing back and forth and never settle | Lower learning rates, TTUR (different learning rates for $G$ and $D$), spectral normalisation |
| **No quality signal** | The loss values say little about image quality | Look at samples from a **fixed** noise batch, or measure FID |

**Wasserstein GAN (WGAN).** This variant replaces the JS divergence with the **Earth-Mover (Wasserstein-1) distance**, which gives useful gradients even when the two distributions don't overlap:

```math
W_1(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \le 1} \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]
```

The "critic" $f$ must be **1-Lipschitz**. There are three ways to enforce this:

- **weight clipping**, as in the original WGAN, which is crude
- a **gradient penalty**, as in WGAN-GP: add $\lambda\, \mathbb{E}_{\hat{\mathbf{x}}}\big[(\lVert \nabla D(\hat{\mathbf{x}}) \rVert_2 - 1)^2\big]$, where $\hat{\mathbf{x}}$ is a random mix of a real and a fake sample
- **spectral normalisation**: divide each weight matrix by its largest singular value, $\bar{W} = W / \sigma(W)$

Other alternative losses include the **least-squares GAN**, $\min_G \mathbb{E}\big[(D(G(\mathbf{z})) - 1)^2\big]$, which gives smoother gradients.

### 3.8 Beyond DCGAN

- **Progressive GANs** start by generating 4 × 4 images and add layers until they reach 1024 × 1024. This makes high-resolution training stable.
- **StyleGAN** first maps $\mathbf{z}$ to an intermediate style vector $\mathbf{w}$, then injects $\mathbf{w}$ at every layer through adaptive instance normalisation:

```math
\text{AdaIN}(\mathbf{x}_i, \mathbf{y}) = \mathbf{y}_{s,i}\,\frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + \mathbf{y}_{b,i}
```

This gives fine control over coarse features (pose) and fine ones (texture).

---

## 4. VAE vs GAN

| | VAE | GAN |
|---|---|---|
| Training | One stable loss | An adversarial game that can be unstable |
| Sample quality | Often **blurry** | **Sharp** |
| Likelihood | A lower bound on $\log p(\mathbf{x})$ | None |
| Encoder (x → z) | ✅ Built in | ❌ Not by default |
| Latent space | Smooth, easy to interpolate | Less structured |
| Typical failure | Posterior collapse, blurriness | Mode collapse, oscillation |

> [!TIP]
> Modern systems often combine the ideas. **Diffusion models** and **VQ-GAN** use autoencoder-style latents together with adversarial or denoising objectives.
