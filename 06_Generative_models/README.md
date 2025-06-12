# Generative Models

Generative models constitute a fundamental class of machine learning algorithms that aim to learn and model the underlying probability distribution $\large p(\mathbf{x})$ of observed data $\large \mathbf{x}$. The primary objective is to capture the statistical structure of the data such that we can:

-   **Generate new samples**: Draw new data points $\large \mathbf{x}_{new} \sim p(\mathbf{x})$ that are statistically similar to the training data
-   **Density estimation**: Evaluate the likelihood $\large p(\mathbf{x})$ for any given data point
-   **Data completion**: Fill in missing parts of partially observed data
-   **Representation learning**: Learn meaningful low-dimensional representations of high-dimensional data

<div align="center">
<img src="assets/taxonomy.png" width="850" height="620">
<p>Fig. Taxonomy of Generative models</p>
</div>

### Mathematical Formulation

Given a dataset $\large \mathcal{D} = {\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}}$ where each $\large \mathbf{x}^{(i)} \in \mathbb{R}^D$, we seek to learn a model $\large p_{\theta}(\mathbf{x})$ parameterized by $\large \theta$ that approximates the true data distribution $\large p_{data}(\mathbf{x})$.

The maximum likelihood estimation (MLE) objective is: 

$$\large 
\theta^* = \arg\max_{\theta} \sum_{i=1}^{N} \log p_{\theta}(\mathbf{x}^{(i)})
$$

## Variational Autoencoders (VAEs)

### Variational Autoencoders vs. Standard Autoencoders

#### Standard Autoencoders

A standard autoencoder consists of:

-   **Encoder**: $\large f_{\phi}: \mathbb{R}^D \rightarrow \mathbb{R}^d$ mapping input $\large \mathbf{x}$ to latent code $\large \mathbf{h} = f_{\phi}(\mathbf{x})$
-   **Decoder**: $\large g_{\theta}: \mathbb{R}^d \rightarrow \mathbb{R}^D$ reconstructing $\large \hat{\mathbf{x}} = g_{\theta}(\mathbf{h})$

**Objective**: Minimize reconstruction error $\large \mathcal{L}_{AE} = |\mathbf{x} - g_{\theta}(f_{\phi}(\mathbf{x}))|^2$

**Limitations**:

-   Deterministic encoding/decoding
-   No probabilistic interpretation
-   Latent space may have "holes" or discontinuities
-   Cannot generate new samples by sampling from latent space

#### Variational Autoencoders

VAEs introduce a probabilistic framework:

**Encoder (Recognition Model)**: 

$$\large 
q_{\phi}(\mathbf{z}|\mathbf{x}): \mathbb{R}^D \rightarrow \mathcal{P}(\mathbb{R}^d)
$$ 

Maps input to a probability distribution over latent variables

**Decoder (Generative Model)**: 

$$\large 
p_{\theta}(\mathbf{x}|\mathbf{z}): \mathbb{R}^d \rightarrow \mathcal{P}(\mathbb{R}^D)
$$ 

Maps latent variables to a probability distribution over data space

**Key Advantages**:

-   Probabilistic interpretation enables uncertainty quantification
-   Structured latent space suitable for generation
-   Principled regularization through prior matching
-   Enables sampling of new data points

<div align="center">
<img src="assets/vae.png" width="1200" height="550">
<p>Fig. Variational Autoencoders (VAEs)</p>
</div>

### Mathematical Foundation of VAEs

#### Latent Variable Model

VAEs are based on the latent variable model assumption: 

$$\large 
p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z}),d\mathbf{z}
$$

Where:

-   $\large \mathbf{z} \in \mathbb{R}^d$ are latent variables (typically $\large d \ll D$)
-   $\large p(\mathbf{z})$ is the prior distribution over latent variables
-   $\large p(\mathbf{x}|\mathbf{z})$ is the likelihood of data given latent variables
-   $\large p(\mathbf{x})$ is the marginal likelihood (evidence)

#### The Intractability Problem

The posterior distribution $\large p(\mathbf{z}|\mathbf{x}) = \frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p(\mathbf{x})}$ is intractable because:

1.  **Marginal likelihood**: $\large p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z}),d\mathbf{z}$ requires integration over all possible latent configurations
2.  **High-dimensional integration**: For neural networks, this integral has no closed form

#### Variational Inference Solution

VAEs employ **variational inference** to approximate the intractable posterior $\large p(\mathbf{z}|\mathbf{x})$ with a tractable variational distribution $\large q_{\phi}(\mathbf{z}|\mathbf{x})$.

The quality of approximation is measured by KL divergence: 

$$\large 
D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z}|\mathbf{x})) = \int q_{\phi}(\mathbf{z}|\mathbf{x}) \log \frac{q_{\phi}(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}
$$

### Core Components of VAEs

#### Probabilistic Encoder: $\large q_{\phi}(\mathbf{z}|\mathbf{x})$

The encoder network parameterizes the variational posterior. For computational tractability, we typically assume a factorized Gaussian form:

<div align="center">

![encoder](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20q_{\phi}(\mathbf{z}|\mathbf{x})%20=%20\mathcal{N}(\mathbf{z};%20\boldsymbol{\mu}_{\phi}(\mathbf{x}),%20\text{diag}(\boldsymbol{\sigma}^2_{\phi}(\mathbf{x}))))

</div>>

Where:

-   $\large \boldsymbol{\mu}_{\phi}(\mathbf{x}) \in \mathbb{R}^d$ is the mean vector
-   $\large \boldsymbol{\sigma}^2_{\phi}(\mathbf{x}) \in \mathbb{R}^d$ is the variance vector (diagonal covariance)
-   $\large \phi$ represents the neural network parameters

**Neural Network Implementation**:

```
Input: x ∈ ℝᴰ
↓
Hidden Layers (ReLU/Tanh activations)
↓
Split into two heads:
μ_φ(x) ∈ ℝᵈ (linear output)
log σ²_φ(x) ∈ ℝᵈ (linear output, for numerical stability)

```

#### Probabilistic Decoder: $\large p_{\theta}(\mathbf{x}|\mathbf{z})$

The decoder network parameterizes the conditional likelihood of data given latent variables.

**For Continuous Data** (e.g., real-valued images): 

<div align="center">

![continuous](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20p_{\theta}(\mathbf{x}|\mathbf{z})%20=%20\mathcal{N}(\mathbf{x};%20\boldsymbol{\mu}_{\theta}(\mathbf{z}),%20\sigma^2_{dec}\mathbf{I}))

</div>

Where $\large \sigma^2_{dec}$ can be fixed or learned.

**For Binary Data** (e.g., binary images, text): 

$$\large 
p_{\theta}(\mathbf{x}|\mathbf{z}) = \prod_{i=1}^{D} \text{Bernoulli}(x_i; p_i)
$$

Where $\large p_i = \sigma(\mu_{\theta,i}(\mathbf{z}))$ and $\large \sigma(\cdot)$ is the sigmoid function.

#### Prior Distribution: $\large p(\mathbf{z})$

The prior distribution is typically chosen to be a standard multivariate Gaussian: 

$$\large 
p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
$$

This choice provides several advantages:

-   **Simplicity**: Easy to sample from and compute KL divergence
-   **Regularity**: Encourages smooth latent space structure
-   **Interpretability**: Centered at origin with unit variance

**Alternative Priors**:

-   **Mixture of Gaussians**: $\large p(\mathbf{z}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
-   **Von Mises-Fisher**: For directional data
-   **Beta**: For bounded latent variables

#### The Reparameterization Trick

**Problem**: Direct sampling $\large \mathbf{z} \sim q_{\phi}(\mathbf{z}|\mathbf{x})$ is not differentiable with respect to $\large \phi$.

**Solution**: Reparameterize the sampling process to separate the stochastic and deterministic components.

**For Gaussian Distributions**: 

<div align="center">

![gaussian](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathbf{z}%20=%20\boldsymbol{\mu}_{\phi}(\mathbf{x})%20+%20\boldsymbol{\sigma}_{\phi}(\mathbf{x})%20\odot%20\boldsymbol{\epsilon})

</div>

Where:

-   $\large \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is auxiliary noise
-   $\large \odot$ denotes element-wise multiplication
-   ![phi](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\boldsymbol{\sigma}_{\phi}(\mathbf{x})%20=%20\exp(0.5%20\cdot%20\log%20\boldsymbol{\sigma}^2_{\phi}(\mathbf{x})))

**Gradient Flow**: 

<div align="center">

![gradient](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\frac{\partial}{\partial%20\phi}%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})]%20=%20\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\frac{\partial%20f(\mathbf{z})}{\partial%20\mathbf{z}}%20\frac{\partial%20\mathbf{z}}{\partial%20\phi}\right])

</div>

<div align="center">
<img src="assets/reparam.png" width="1200" height="625" style="background-color: white;">
<p>Fig. The Reparameterization Trick</p>
</div>

### The VAE Objective: Evidence Lower Bound (ELBO) 

<div align="center">
<img src="assets/elbo.png">
<p>Fig. Evidence Lower Bound (ELBO)</p>
</div>

#### Problem Formulation

**Goal**: Maximize the marginal log-likelihood of observed data: 

<div align="center">

![goal](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{ML}%20=%20\sum_{i=1}^{N}%20\log%20p_{\theta}(\mathbf{x}^{(i)}))

</div>

**Challenge**: $\large \log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}|\mathbf{z})p(\mathbf{z}),d\mathbf{z}$ is intractable.

#### Variational Lower Bound

For any variational distribution $\large q_{\phi}(\mathbf{z}|\mathbf{x})$, we can write:

<div align="center">

![vlb1](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\log%20p_{\theta}(\mathbf{x})%20=%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log%20p_{\theta}(\mathbf{x})]%20=%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log%20\frac{p_{\theta}(\mathbf{x},\mathbf{z})}{p_{\theta}(\mathbf{z}|\mathbf{x})}\right])

</div>>

<div align="center">

![vlb2](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20=%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log%20\frac{p_{\theta}(\mathbf{x},\mathbf{z})q_{\phi}(\mathbf{z}|\mathbf{x})}{p_{\theta}(\mathbf{z}|\mathbf{x})q_{\phi}(\mathbf{z}|\mathbf{x})}\right])

</div>

<div align="center">

![vlb3](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20=%20\underbrace{\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log%20\frac{p_{\theta}(\mathbf{x},\mathbf{z})}{q_{\phi}(\mathbf{z}|\mathbf{x})}\right]}_{\text{ELBO}}%20+%20\underbrace{D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})%20|%20p_{\theta}(\mathbf{z}|\mathbf{x}))}_{\geq%200})

</div>

#### ELBO Decomposition

The Evidence Lower Bound (ELBO) can be expressed as:

<div align="center">

![elbo](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{ELBO}(\theta,%20\phi;%20\mathbf{x})%20=%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log%20p_{\theta}(\mathbf{x}|\mathbf{z})]%20-%20D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})%20|%20p(\mathbf{z})))

</div>

**Two Interpretations**:

1.  **Reconstruction + Regularization**:
    
    -   **Reconstruction Term**: ![reconstruction](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log%20p_{\theta}(\mathbf{x}|\mathbf{z})])
    -   **Regularization Term**: $\large -D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z}))$
2.  **Rate-Distortion**:
    
    -   **Rate**: $\large D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z}))$ (information cost)
    -   **Distortion**: ![distortion](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20-\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log%20p_{\theta}(\mathbf{x}|\mathbf{z})]) (reconstruction error)

### Implementation Details

#### Network Architecture

**Encoder Architecture**:

```
Input: x ∈ ℝᴰ (e.g., 784 for MNIST)
↓
Linear(D, 512) + ReLU
↓
Linear(512, 256) + ReLU
↓
Split:
├── μ: Linear(256, d)
└── log σ²: Linear(256, d)

```

**Decoder Architecture**:

```
Input: z ∈ ℝᵈ
↓
Linear(d, 256) + ReLU
↓
Linear(256, 512) + ReLU
↓
Linear(512, D) + Sigmoid (for binary data)

```

#### Loss Function Implementation

**Total Loss (Negative ELBO)**: 

![negelbo](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{VAE}%20=%20\mathcal{L}_{recon}%20+%20\beta%20\cdot%20\mathcal{L}_{KL})

**Reconstruction Loss**:

-   **Binary Cross-Entropy** (for binary/normalized data): 

<div align="center">

![bce](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{recon}%20=%20-\sum_{i=1}^{D}%20x_i%20\log%20\hat{x}_i%20+%20(1-x_i)\log(1-\hat{x}_i))

</div>>
    
-   **Mean Squared Error** (for continuous data): 

<div align="center">

![mse](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{recon}%20=%20\frac{1}{D}\sum_{i=1}^{D}%20(x_i%20-%20\hat{x}_i)^2)

</div>>
    

**KL Divergence Loss**: 

<div align="center">

![kl](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{KL}%20=%20\frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2%20+%20\sigma_j^2%20-%201%20-%20\log%20\sigma_j^2\right))

</div>>

#### β-VAE Extension

The β-VAE introduces a hyperparameter β to control the trade-off: 

<div align="center">

![bvae](https://math.vercel.app/?color=white&bgcolor=auto&from=\large%20\mathcal{L}_{\beta-VAE}%20=%20\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log%20p_{\theta}(\mathbf{x}|\mathbf{z})]%20-%20\beta%20\cdot%20D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x})%20|%20p(\mathbf{z})))

</div>>

**Effects of β**:

-   **β < 1**: Prioritizes reconstruction, may lead to posterior collapse
-   **β = 1**: Standard VAE
-   **β > 1**: Emphasizes disentanglement, may sacrifice reconstruction quality
