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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/probability_statistics.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git")

# %% [markdown]
# # Probability & Statistics
#
# ## Probability Distributions

# %% [markdown]
# A **probability distribution** is a mathematical function that describes how likely each possible value of a random variable is.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# %%
sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"SciPy Version: {scipy.__version__}")
print(f"Seaborn Version: {sns.__version__}")

# %% [markdown]
# ### Discrete Distributions

# %% [markdown]
# #### Bernoulli Distribution
# - Represents a single trial with two outcomes (e.g., success/failure, 0/1).
# - Parameter: p (probability of success, i.e., outcome 1).

# %%
probs_bernoulli = torch.tensor([0.7])
bernoulli_dist = torch.distributions.Bernoulli(probs=probs_bernoulli)

# %%
bernoulli_dist

# %% [markdown]
# - Probability Mass Function (PMF) - P(X=k) 
# - For Bernoulli, k can be 0 or 1.
# - P(X=1) = p; P(X=0) = 1 - p

# %%
# We can use log_prob and then exp to get the probability.
k_values_bernoulli = torch.tensor([0., 1.])
pmf_bernoulli_log = bernoulli_dist.log_prob(k_values_bernoulli)
pmf_bernoulli_log

# %%
pmf_bernoulli = torch.exp(pmf_bernoulli_log)
print(f"PMF for k=0: {pmf_bernoulli[0]:.2f}, k=1: {pmf_bernoulli[1]:.2f}")

# %% [markdown]
# - Cumulative Distribution Function (CDF) - P (X <= k)
# - PyTorch distributions generally don't have a direct CDF method for discrete distributions (We can calculate it manually for simple cases or SciPy)

# %%
cdf_bernoulli_0 = pmf_bernoulli[0]
cdf_bernoulli_1 = pmf_bernoulli[0] + pmf_bernoulli[1]
print(f"CDF for k=0: {cdf_bernoulli_0:.2f}, k=1: {cdf_bernoulli_1:.2f}")

# %%
n_samples = 10
samples_bernoulli = bernoulli_dist.sample(sample_shape=torch.Size([n_samples]))
samples_bernoulli

# %%
samples_bernoulli.squeeze()

# %%
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.bar(k_values_bernoulli.numpy(), pmf_bernoulli.numpy(), color='skyblue', label=f'p={probs_bernoulli.item():.1f}')
plt.title('Bernoulli PMF')
plt.xlabel('Outcome (k)')
plt.ylabel('Probability P(X=k)')
plt.xticks([0, 1])
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(samples_bernoulli.squeeze().numpy(), discrete=True, stat="probability", shrink=0.8)
plt.title(f'Histogram of {n_samples} Bernoulli Samples')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.xticks([0, 1])
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Binomial Distribution
# - Represents the number of successes in a fixed number 'n' (total_count) of independent Bernoulli trials.
# - Parameters: n (total_count), p (probability of success in each trial).

# %%
total_count_binomial = 10
probs_binomial = torch.tensor([0.5])
binomial_dist = torch.distributions.Binomial(total_count=total_count_binomial, probs=probs_binomial)
binomial_dist

# %%
# PMF - P(X=k) = C(n, k) * p^k * (1-p)^(n-k)
k_values_binomial = torch.arange(0, total_count_binomial + 1, dtype=torch.float32)
pmf_binomial_log = binomial_dist.log_prob(k_values_binomial)
pmf_binomial = torch.exp(pmf_binomial_log)
pmf_binomial

# %%
n_samples_binom = 1000
samples_binomial = binomial_dist.sample(sample_shape=torch.Size([n_samples_binom]))

# %%
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(k_values_binomial.numpy(), pmf_binomial.numpy(), 
       color='lightcoral', 
       alpha=0.8,
       label=f'n={total_count_binomial}, p={probs_binomial.item():.1f}')
       
plt.title('Binomial PMF')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability P(X=k)')
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(samples_binomial.numpy(), 
            discrete=True, 
            stat="probability", 
            bins=total_count_binomial+1,
            color='lightcoral',
            alpha=0.8)
            
plt.title(f'Histogram of {n_samples_binom} Binomial Samples')
plt.xlabel('Number of Successes')
plt.ylabel('Frequency')
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Categorical Distribution (Generalized Bernoulli)
# - Represents a single trial with K possible outcomes (categories).
# - Parameter: probs (a vector of K probabilities, must sum to 1).

# %%
# Probabilities for K=3 categories (e.g., rolling a loaded die)
probs_categorical = torch.tensor([0.2, 0.5, 0.3])
categorical_dist = torch.distributions.Categorical(probs=probs_categorical)
categorical_dist

# %%
# PMf - P(X=k_i) = p_i
k_values_categorical = torch.arange(len(probs_categorical))
k_values_categorical

# %%
pmf_categorical_log = categorical_dist.log_prob(k_values_categorical.float())
pmf_categorical_log

# %%
pmf_categorical = torch.exp(pmf_categorical_log)
k_values_categorical.numpy(), pmf_categorical.numpy()

# %%
n_samples_cat = 1000
samples_categorical = categorical_dist.sample(sample_shape=torch.Size([n_samples_cat]))

# %%
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(k_values_categorical.numpy(), pmf_categorical.numpy(), 
       color='lightcoral', 
       alpha=0.8,
       label=f'Probs={probs_categorical.numpy()}')
       
plt.title('Categorical PMF')
plt.xlabel('Category (k)')
plt.ylabel('Probability P(X=k)')
plt.xticks(k_values_categorical.numpy())
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(samples_categorical.numpy(), 
            discrete=True, 
            stat="probability", 
            bins=len(probs_categorical),
            color='lightcoral',
            alpha=0.8)
            
plt.title(f'Histogram of {n_samples_cat} Categorical Samples')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(k_values_categorical.numpy())
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Continuous Distributions

# %% [markdown]
# #### Uniform Distribution
# - All values within a given range [a,b] are equally likely.
# - Parameters: a (low), b (high)

# %%
low_uniform = torch.tensor(0.0)
high_uniform = torch.tensor(5.0)
uniform_dist_pt = torch.distributions.Uniform(low=low_uniform, high=high_uniform)
uniform_dist_pt

# %%
# Probability Density Function (PDF) - f(x)
# f(x) = 1 / (b - a) for a <= x <= b, and 0 otherwise.
x_values_uniform = torch.linspace(low_uniform - 1, high_uniform + 1, 500)
x_values_uniform

# %%
# Calculate PDF values manually to avoid errors
pdf_uniform_pt = torch.zeros_like(x_values_uniform)
valid_indices = (x_values_uniform >= low_uniform) & (x_values_uniform <= high_uniform)
valid_x = x_values_uniform[valid_indices]
valid_x

# %%
# For valid x values within range, compute log_prob and then exp
if valid_x.numel() > 0:  # Check if there are any valid values
    pdf_uniform_pt[valid_indices] = torch.exp(uniform_dist_pt.log_prob(valid_x))

# %%
cdf_uniform_pt = torch.zeros_like(x_values_uniform)
cdf_uniform_pt[x_values_uniform < low_uniform] = 0.0
cdf_uniform_pt[x_values_uniform > high_uniform] = 1.0
mid_indices = (x_values_uniform >= low_uniform) & (x_values_uniform <= high_uniform)
cdf_uniform_pt[mid_indices] = (x_values_uniform[mid_indices] - low_uniform) / (high_uniform - low_uniform)

# %%
# Draw samples to compare the empirical histogram with the PDF
n_samples_uniform = 1000
samples_uniform_pt = uniform_dist_pt.sample(sample_shape=torch.Size([n_samples_uniform]))

# %%
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(15, 5))

# Plot PDF
plt.subplot(1, 3, 1)
plt.plot(x_values_uniform.numpy(), pdf_uniform_pt.numpy(), 
         color='mediumorchid', 
         label=f'U({low_uniform.item()}, {high_uniform.item()})')
plt.title('Uniform PDF (PyTorch)')
plt.xlabel('x')
plt.ylabel('Density f(x)')
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

# Plot CDF
plt.subplot(1, 3, 2)
plt.plot(x_values_uniform.numpy(), cdf_uniform_pt.numpy(), 
         color='plum', 
         label=f'U({low_uniform.item()}, {high_uniform.item()})')
plt.title('Uniform CDF (PyTorch)')
plt.xlabel('x')
plt.ylabel('Cumulative Probability P(X ≤ x)')
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

# Histogram of samples
plt.subplot(1, 3, 3)
sns.histplot(samples_uniform_pt.squeeze().numpy(), 
             stat="density", 
             bins=30, 
             kde=True,
             color='orchid', 
             alpha=0.8)
plt.title(f'Histogram of {n_samples_uniform} Uniform Samples')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()


# %% [markdown]
# #### Normal (Gaussian) Distribution

# %%
mean_normal = torch.tensor(0.0)
std_normal = torch.tensor(1.0)
normal_dist_pt = torch.distributions.Normal(loc=mean_normal, scale=std_normal)
normal_dist_pt

# %%
# For better CDF plotting, let's also use scipy.stats
norm_scipy = scipy.stats.norm(loc=mean_normal, scale=std_normal.item())

# %%
# PDF - f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
x_values_normal = torch.linspace(mean_normal - 4*std_normal, mean_normal + 4*std_normal, 500)
pdf_normal_log_pt = normal_dist_pt.log_prob(x_values_normal)
pdf_normal_pt = torch.exp(pdf_normal_log_pt)
pdf_normal_scipy = norm_scipy.pdf(x_values_normal.numpy())

# %%
# CDF - P(X <= x)
cdf_normal_pt = normal_dist_pt.cdf(x_values_normal)
cdf_normal_scipy = norm_scipy.cdf(x_values_normal.numpy())

# %% [markdown]
# ### Verifying against SciPy
# The manual uniform PDF/CDF and the `torch.distributions` results should match `scipy.stats` to float32 precision.

# %%
from mlf_utils import check_close

check_close("Normal PDF: torch vs scipy", pdf_normal_pt, pdf_normal_scipy, atol=1e-6)
check_close("Normal CDF: torch vs scipy", cdf_normal_pt, cdf_normal_scipy, atol=1e-6)

uniform_scipy = scipy.stats.uniform(loc=low_uniform.item(), scale=(high_uniform - low_uniform).item())
check_close("Uniform PDF: manual vs scipy", pdf_uniform_pt, uniform_scipy.pdf(x_values_uniform.numpy()), atol=1e-6)
check_close("Uniform CDF: manual vs scipy", cdf_uniform_pt, uniform_scipy.cdf(x_values_uniform.numpy()), atol=1e-6)

binom_scipy = scipy.stats.binom(total_count_binomial, probs_binomial.item())
check_close("Binomial PMF: torch vs scipy", pmf_binomial, binom_scipy.pmf(k_values_binomial.numpy()), atol=1e-5)

# %%
# Draw samples to compare the empirical histogram with the PDF
n_samples_normal = 1000
samples_normal_pt = normal_dist_pt.sample(sample_shape=torch.Size([n_samples_normal]))

# %%
plt.style.use('seaborn-v0_8')
plt.figure(figsize=(18, 5))

# Plot PDF
plt.subplot(1, 3, 1)
plt.plot(x_values_normal.numpy(), pdf_normal_pt.numpy(), 
         color='dodgerblue', 
         label=f'PyTorch N({mean_normal.item()}, {std_normal.item()}²)')
plt.plot(x_values_normal.numpy(), pdf_normal_scipy, 
         color='tomato', linestyle='--', 
         label=f'SciPy N({mean_normal.item()}, {std_normal.item()}²)')
plt.title('Normal PDF')
plt.xlabel('x')
plt.ylabel('Density f(x)')
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

# Plot CDF
plt.subplot(1, 3, 2)
plt.plot(x_values_normal.numpy(), cdf_normal_pt.numpy(), 
         color='deepskyblue', 
         label=f'PyTorch N({mean_normal.item()}, {std_normal.item()}²)')
plt.plot(x_values_normal.numpy(), cdf_normal_scipy, 
         color='orangered', linestyle='--', 
         label=f'SciPy N({mean_normal.item()}, {std_normal.item()}²)')
plt.title('Normal CDF')
plt.xlabel('x')
plt.ylabel('Cumulative Probability P(X ≤ x)')
plt.grid(alpha=0.3, linestyle='--')
plt.legend()

# Histogram of samples
plt.subplot(1, 3, 3)
sns.histplot(samples_normal_pt.squeeze().numpy(), 
             stat="density", 
             bins=30, 
             kde=True, 
             color='skyblue', 
             alpha=0.8)
plt.title(f'Histogram of {n_samples_normal} Normal Samples')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Try it: the normal distribution
# Change $\mu$, $\sigma$ and the number of samples. With few samples the histogram is noisy, and it approaches the PDF as $n$ grows (the law of large numbers). About 68% of samples fall within $\mu \pm \sigma$.
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
from ipywidgets import FloatSlider, IntSlider, interact

@interact(mu=FloatSlider(value=0.0, min=-3, max=3, step=0.1, continuous_update=False),
          sigma=FloatSlider(value=1.0, min=0.2, max=3, step=0.1, continuous_update=False),
          n=IntSlider(value=200, min=10, max=5000, step=10, continuous_update=False))
def explore_normal(mu, sigma, n):
    dist = torch.distributions.Normal(mu, sigma)
    samples = dist.sample((n,))
    x = torch.linspace(-10, 10, 500)
    within = ((samples - mu).abs() <= sigma).float().mean().item()
    plt.figure(figsize=(8, 4))
    plt.hist(samples.numpy(), bins=40, density=True, alpha=0.5, color="dodgerblue", label=f"{n} samples")
    plt.plot(x.numpy(), dist.log_prob(x).exp().numpy(), color="tomato", label="PDF")
    plt.axvspan(mu - sigma, mu + sigma, color="gray", alpha=0.1, label=f"μ ± σ ({within:.0%} of samples)")
    plt.xlim(-10, 10); plt.legend(); plt.title(f"N({mu:.1f}, {sigma:.1f}²)")
    plt.show()


# %% [markdown]
# ### Bayes' Theorem

# %% [markdown]
# Bayes' Theorem describes how to update the probability of a hypothesis (H) given new evidence (E).</br>
# </br>
# Formula: P(H|E) = [P(E|H) * P(H)] / P(E) </br>
# </br>
# where, 
# - P(H|E): Posterior probability - probability of H after observing E.
# - P(E|H): Likelihood - probability of observing E if H is true.
# - P(H): Prior probability - initial probability of H before observing E.
# - P(E): Evidence (or Marginal Likelihood) - total probability of observing E.
# </br>
# P(E) = P(E|H) * P(H) + P(E|~H) * P(~H)  (for binary H) </br>
#      = Σ P(E|H_i) * P(H_i) (for multiple hypotheses H_i)

# %% [markdown]
# #### Example: Medical Test
# - H: Patient has the disease.
# - E: Patient tests positive
# </br></br>
# Prior probability of having the disease </br>
# P(H) = 0.01 (1% of the population has the disease)

# %%
P_H = torch.tensor(0.01)
P_not_H = 1 - P_H
P_not_H

# %% [markdown]
# Likelihoods: </br></br>
# P(E|H) = Probability of testing positive if patient does NOT have the disease (FP) </br>
# P(E|~H) = 0.05 (5% of healthy people test positive) 

# %%
P_E_given_H = torch.tensor(0.99)

# %% [markdown]
# P(E|~H): Probability of testing positive if patient does NOT have the disease (False Positive Rate) </br> </br>
# P(E|~H) = 0.05 (5% of healthy people test positive)

# %%
P_E_given_not_H = torch.tensor(0.05)

# %%
# Calculate P(E) - the evidence (overall probability of testing positive)
# P(E) = P(E|H) * P(H) + P(E|~H) * P(~H)
P_E = (P_E_given_H * P_H) + (P_E_given_not_H * P_not_H)
print(f"P(H) - Prior (Patient has disease): {P_H.item():.4f}")
print(f"P(E|H) - Likelihood (Test positive | Has disease): {P_E_given_H.item():.4f}")
print(f"P(E|~H) - Likelihood (Test positive | No disease): {P_E_given_not_H.item():.4f}")
print(f"P(E) - Evidence (Overall prob. of testing positive): {P_E.item():.4f}")

# %%
# Calculate P(H|E) - the posterior probability (Patient has disease | Tests positive)
# P(H|E) = [P(E|H) * P(H)] / P(E)
P_H_given_E = (P_E_given_H * P_H) / P_E
print(f"\nP(H|E) - Posterior (Patient has disease | Tests positive): {P_H_given_E.item():.4f}")

# %% [markdown]
# **Interpretation**: Even with a positive test, the probability of actually having the disease if ~16.7%, due to the low prior probability (rarity of the disease) and the false positive rate. This is common illustration of how Bayes' theorem helps in reasoning under uncertainty.

# %% [markdown]
# ### Sampling

# %% [markdown]
# Sampling is the process of selecting a subset of individuals or items from within a statistical population to estimate characteristics of the whole population. In ML, we sample from distributions or from datasets.

# %% [markdown]
# #### Sampling Techniques

# %% [markdown]
# #### Sampling from Distributions

# %%
samples_normal_pt = normal_dist_pt.sample(sample_shape=torch.Size([n_samples_normal]))

# %% [markdown]
# #### Simple Random Sampling from a Dataset
# - Each element has an equal chance of being selected.

# %%
# Create a dummy dataset (PyTorch tensor)
population_data = torch.arange(1, 101, dtype=torch.float32)
population_data.shape

# %%
# Sample 10 items without replacement
sample_size = 10

# %%
# Method 1: Using torch.randperm to get random indices
indices = torch.randperm(population_data.shape[0])[:sample_size]
indices

# %%
simple_random_sample_1 = population_data[indices]
simple_random_sample_1

# %%
# Method 2: using np.random.choice 
indices_np = np.random.choice(population_data.shape[0], size=sample_size, replace=False)
indices_np

# %%
simple_random_sample_2 = population_data[torch.from_numpy(indices_np)]
simple_random_sample_2

# %% [markdown]
# #### Stratified Sampling (Conceptual)

# %% [markdown]
# The population is divided into subgroups (strata), and random samples are taken from each stratum, often proportionally. This ensures representation from all subgroups. Example: Sampling users, ensuring you get representation from different age groups. PyTorch doesn't have a direct stratified sampling function for general tensors. </br></br>
# For datasets in ML, libraries like scikit-learn (`sklearn.model_selection.StratifiedShuffleSplit` or `StratifiedKFold`) are commonly used for this, especially when creating train/test splits for classification tasks to maintain class proporitions.

# %%
from sklearn.model_selection import StratifiedShuffleSplit

# %%
X_strat = np.arange(20).reshape(10,2)
y_strat = np.array([0,0,0,0,1,1,1,1,1,1])

# %%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss

# %%
for train_index, test_index in sss.split(X_strat, y_strat):
    X_train, X_test = X_strat[train_index], X_strat[test_index]
    y_train, y_test = y_strat[train_index], y_strat[test_index]
    print(f"\nStratified Sampling (sklearn example):")
    print(f"  Train indices: {train_index}, y_train: {y_train}")
    print(f"  Test indices: {test_index}, y_test: {y_test}")
    print(f"  Proportion of class 1 in y_train: {np.mean(y_train):.2f}")
    print(f"  Proportion of class 1 in y_test: {np.mean(y_test):.2f}")

# %%
