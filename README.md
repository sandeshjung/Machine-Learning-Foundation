# Machine Learning Foundation

**🖱️ Try the interactive explorables in your browser: [sandeshjung.github.io/Machine-Learning-Foundation](https://sandeshjung.github.io/Machine-Learning-Foundation/)**

A hands-on tour of machine learning, from the underlying maths to Transformers, generative models and reinforcement learning.

Every algorithm is **built from scratch** in NumPy/PyTorch, **checked against a library** (scikit-learn, PyTorch, SciPy, fairlearn), and explained with the theory behind it.

I put this together to document what I've studied and to make revision easier. It isn't a complete course, but it's a good starting point if you want to get your hands dirty with ML.

## What's inside

| | |
|---|---|
| 📖 **Theory** | Each module's README explains the ideas step by step, with derivations |
| 🛠️ **From-scratch code** | Notebooks implement each algorithm by hand, then compare it with the library version |
| ✅ **Verified** | Built-in checks fail loudly if a from-scratch result doesn't match the reference |
| 🎚️ **Interactive** | Sliders for learning rates, polynomial degree, regularisation, SVM `C`/`gamma`, K, perplexity, thresholds and a VAE's latent space |
| 🖱️ **Explorables** | In-browser visualisations: drag points, move lines and watch fits, errors and optimisers update live |
| 📝 **Cheat sheets** | One page per module with the key equations, defaults and common pitfalls |
| ☁️ **Runs anywhere** | Open any notebook in Google Colab with one click, or run it locally |

## Contents

- [Getting started](#getting-started)
- [How to use this repo](#how-to-use-this-repo)
- [Modules and notebooks](#modules-and-notebooks)
- [Interactive explorables](#interactive-explorables)
- [How the notebooks are built](#how-the-notebooks-are-built)
- [Repository structure](#repository-structure)

---

## Getting started

### Option 1: Google Colab (no setup)

1. Click a **Colab** link in the tables below, or the <a href="https://colab.research.google.com"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="16"></a> badge at the top of any notebook.
2. Run the first cell. It installs everything the notebook needs.
3. For modules 05–07, switch to a GPU: *Runtime → Change runtime type*.

### Option 2: Run locally

The notebooks were tested with Python 3.11.

```bash
git clone https://github.com/sandeshjung/Machine-Learning-Foundation.git
cd Machine-Learning-Foundation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # also installs the shared helpers in mlf_utils/

jupyter notebook
```

> [!NOTE]
> - **Graphviz:** the autograd notebooks in module 00 draw computation graphs, which needs the Graphviz program as well: `brew install graphviz`, `sudo apt install graphviz`, or see [graphviz.org](https://graphviz.org/download/).
> - **Datasets** (CIFAR-10, Fashion-MNIST, EMNIST, UCI Adult) download automatically into a `data/` folder on the first run.
> - **GPU:** modules 05–07 train much faster on a GPU, but everything also runs on a CPU.
> - **Widgets** need a running kernel, locally or in Colab. GitHub only shows a static preview.

---

## How to use this repo

For each module:

1. **Read** its README for the theory.
2. **Run** its notebooks to see the ideas in code, and play with the sliders.
   Where a module has an [explorable](#interactive-explorables), open it to build intuition first.
3. **Revise** later with its one-page cheat sheet.

The modules build on each other, so going in order (00 → 08) works best. Modules 00–01 are enough to start any other module, though.

---

## Modules and notebooks

### 00 · Mathematical foundation

The maths every later module relies on. [Theory](00_Mathematical_foundation/README.md) · [Cheat sheet](00_Mathematical_foundation/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [linear_algebra](00_Mathematical_foundation/linear_algebra.ipynb) | Vectors, matrices, eigen-decomposition and SVD in NumPy and PyTorch | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/linear_algebra.ipynb) |
| [probability_statistics](00_Mathematical_foundation/probability_statistics.ipynb) | Distributions, Bayes' theorem and sampling, checked against SciPy | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/probability_statistics.ipynb) |
| [calculus_optimization](00_Mathematical_foundation/calculus_optimization.ipynb) | Derivatives, gradients and finite-difference checks | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/calculus_optimization.ipynb) |
| [optimization_algorithms](00_Mathematical_foundation/optimization_algorithms.ipynb) | Gradient descent, SGD and `torch.optim`, with a learning-rate slider | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/optimization_algorithms.ipynb) |
| [autograd(scalar)](00_Mathematical_foundation/autograd%28scalar%29.ipynb) | A micrograd-style autograd engine for scalars, verified against PyTorch | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/autograd%28scalar%29.ipynb) |
| [autograd(tensor)](00_Mathematical_foundation/autograd%28tensor%29.ipynb) | The same engine for tensors, including broadcasting in the backward pass | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/autograd%28tensor%29.ipynb) |

### 01 · Supervised regression

Predicting numbers, and the problem of over- and underfitting. [Theory](01_Supervised_Regression/README.md) · [Cheat sheet](01_Supervised_Regression/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [linear_regression](01_Supervised_Regression/linear_regression.ipynb) | Gradient descent by hand vs `nn.Linear`, checked against scikit-learn | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/linear_regression.ipynb) |
| [polynomial_overfitting](01_Supervised_Regression/polynomial_overfitting.ipynb) | Polynomial features and overfitting, with a degree slider | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/polynomial_overfitting.ipynb) |
| [regularization](01_Supervised_Regression/regularization.ipynb) | Ridge and Lasso from scratch, with a regularisation-strength slider | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/regularization.ipynb) |

### 02 · Supervised classification

Six classic classifiers, from linear models to boosted trees. [Theory](02_Supervised_classification/README.md) · [Cheat sheet](02_Supervised_classification/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [logistic_regression](02_Supervised_classification/logistic_regression.ipynb) | Sigmoid, cross-entropy and a decision-threshold slider | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/logistic_regression.ipynb) |
| [svm_kernels](02_Supervised_classification/svm_kernels.ipynb) | Hinge-loss SVM in PyTorch, then kernel SVMs with `C`/`gamma` sliders | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/svm_kernels.ipynb) |
| [naive_bayes](02_Supervised_classification/naive_bayes.ipynb) | Gaussian, Multinomial and Bernoulli Naive Bayes from scratch | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/naive_bayes.ipynb) |
| [knn](02_Supervised_classification/knn.ipynb) | k-nearest neighbours, distance metrics and the choice of k | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/knn.ipynb) |
| [decision_trees](02_Supervised_classification/decision_trees.ipynb) | A CART tree from scratch: Gini, entropy and pruning | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/decision_trees.ipynb) |
| [ensembles](02_Supervised_classification/ensembles.ipynb) | Bagging, random forests, AdaBoost and gradient boosting | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/ensembles.ipynb) |

### 03 · Model evaluation & selection

Measuring honestly how well a model will do on new data. [Theory](03_Model_evaluation_selection/README.md) · [Cheat sheet](03_Model_evaluation_selection/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [bias_variance](03_Model_evaluation_selection/bias_variance.ipynb) | Under- and overfitting made visible | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/bias_variance.ipynb) |
| [cross_validation](03_Model_evaluation_selection/cross_validation.ipynb) | K-fold and stratified K-fold from scratch | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/cross_validation.ipynb) |
| [hyperparameter_tuning](03_Model_evaluation_selection/hyperparameter_tuning.ipynb) | Grid search vs random search | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/hyperparameter_tuning.ipynb) |
| [classification_metrics](03_Model_evaluation_selection/classification_metrics.ipynb) | Precision/recall, ROC and PR curves, thresholds, imbalance and calibration | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/classification_metrics.ipynb) |
| [preprocessing_pipelines](03_Model_evaluation_selection/preprocessing_pipelines.ipynb) | Missing values, encoding, scaling, pipelines and data leakage | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/preprocessing_pipelines.ipynb) |

### 04 · Unsupervised learning

Finding structure without labels. [Theory](04_Unsupervised_learning/README.md) · [Cheat sheet](04_Unsupervised_learning/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [kmeans_hierarchical](04_Unsupervised_learning/kmeans_hierarchical.ipynb) | K-means from scratch, choosing K, and hierarchical clustering | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/04_Unsupervised_learning/kmeans_hierarchical.ipynb) |
| [pca_tsne](04_Unsupervised_learning/pca_tsne.ipynb) | PCA via SVD (checked against scikit-learn) and t-SNE with a perplexity slider | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/04_Unsupervised_learning/pca_tsne.ipynb) |

### 05 · Neural networks & deep learning

From a hand-written MLP to Transformers. [Theory](05_Neural_networks_deep_learning/README.md) · [Cheat sheet](05_Neural_networks_deep_learning/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [mlp_backprop](05_Neural_networks_deep_learning/mlp_backprop.ipynb) | An MLP with backpropagation written by hand, on EMNIST letters | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/mlp_backprop.ipynb) |
| [cnn](05_Neural_networks_deep_learning/cnn.ipynb) | A small CNN on CIFAR-10, with the output-size formula checked layer by layer | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/cnn.ipynb) |
| [rnn](05_Neural_networks_deep_learning/rnn.ipynb) | RNN, LSTM and GRU for time-series forecasting | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/rnn.ipynb) |
| [transformer](05_Neural_networks_deep_learning/transformer.ipynb) | A Transformer built from scratch, trained to reverse sequences | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/transformer.ipynb) |
| [optimizers_regularization](05_Neural_networks_deep_learning/optimizers_regularization.ipynb) | SGD vs Adam (Adam from scratch), LR schedules, weight decay and dropout | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/optimizers_regularization.ipynb) |

### 06 · Generative models

Models that create new data. [Theory](06_Generative_models/README.md) · [Cheat sheet](06_Generative_models/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [vae](06_Generative_models/vae.ipynb) | A variational autoencoder with an explorable 2-D latent space | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/vae.ipynb) |
| [gan](06_Generative_models/gan.ipynb) | A generative adversarial network on Fashion-MNIST | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/gan.ipynb) |

### 07 · Reinforcement learning

Learning by trial and error. [Theory](07_Reinforcement_learning/README.md) · [Cheat sheet](07_Reinforcement_learning/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [rl_basics](07_Reinforcement_learning/rl_basics.ipynb) | MDPs, value iteration on a grid world, and REINFORCE | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/07_Reinforcement_learning/rl_basics.ipynb) |
| [dqn_and_actor_critic](07_Reinforcement_learning/dqn_and_actor_critic.ipynb) | DQN and A2C on CartPole | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/07_Reinforcement_learning/dqn_and_actor_critic.ipynb) |

### 08 · Fairness & interpretability

Responsible ML: is the model fair, and why does it decide what it does? [Theory](08_Other_topics/README.md) · [Cheat sheet](08_Other_topics/CHEATSHEET.md)

| Notebook | What it covers | |
|---|---|---|
| [fairness_interpretability](08_Other_topics/fairness_interpretability.ipynb) | Group fairness metrics (checked against fairlearn), threshold mitigation, LIME and SHAP | [Colab](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/08_Other_topics/fairness_interpretability.ipynb) |

---

## Interactive explorables

**Live site: [sandeshjung.github.io/Machine-Learning-Foundation](https://sandeshjung.github.io/Machine-Learning-Foundation/)**

Small visualisations that run in your browser, with nothing to install. Every chart updates as you drag or slide, so you can *see* an idea instead of just reading about it.

| Explorable | What you can do | Module |
|---|---|---|
| [Fitting a line](https://sandeshjung.github.io/Machine-Learning-Foundation/linear-regression.html) | Drag points, move the line, see the squared errors, the loss landscape and gradient descent's path | 01 |
| [Bias and variance](https://sandeshjung.github.io/Machine-Learning-Foundation/bias-variance.html) | Fit polynomials to many resampled datasets and watch the fits spread as the degree grows | 01, 03 |
| [Ridge vs Lasso](https://sandeshjung.github.io/Machine-Learning-Foundation/regularization.html) | Drag the least-squares solution and see why L1 gives exact zeros while L2 only shrinks | 01 |
| [Optimiser race](https://sandeshjung.github.io/Machine-Learning-Foundation/gradient-descent.html) | Race GD, momentum, RMSProp and Adam on a narrow bowl, a banana valley and a two-minima surface | 00, 05 |
| [Spread and averages](https://sandeshjung.github.io/Machine-Learning-Foundation/normal-distribution.html) | Move μ and σ, draw samples, check the 68–95–99.7 rule and watch the central limit theorem | 00 |

> [!NOTE]
> The site is served by GitHub Pages from the [`docs/`](docs/) folder. To run it offline, open [`docs/index.html`](docs/index.html) in any browser.
>
> They are plain HTML/JavaScript with no libraries. The maths in [`docs/assets/mlmath.js`](docs/assets/mlmath.js) is checked against NumPy, scikit-learn, SciPy and PyTorch by `node docs/tests/mlmath.test.js`.

---

## How the notebooks are built

- **Verified implementations.** Every from-scratch result is compared with a library reference using `check_close` / `check_agreement`, which raise an error if they disagree. For example:
  - autograd vs `torch.autograd`
  - PCA vs `sklearn.decomposition.PCA`
  - attention vs `F.scaled_dot_product_attention`
  - Adam vs `torch.optim.Adam`
  - the VAE's KL term vs `torch.distributions`

  Exact methods must match to float32 precision. Iterative ones, such as gradient descent, use a tolerance that is explained next to each check.
- **Shared helpers.** Plumbing used by several notebooks (image grids, decision-region plots, graph drawing, the checks) lives in the small [`mlf_utils`](mlf_utils/) package. The algorithms themselves always stay in the notebooks.
- **Paired `.py` files.** Every notebook has a plain-Python twin, kept in sync by [Jupytext](https://jupytext.readthedocs.io) (see [`jupytext.toml`](jupytext.toml)). The twins make diffs and code review readable. Jupyter updates both files when you save. After editing a notebook elsewhere, for example in Colab, run `jupytext --sync */*.ipynb`.

---

## Repository structure

```text
Machine-Learning-Foundation/
├── 00_Mathematical_foundation/
│   ├── README.md              ← theory and derivations
│   ├── CHEATSHEET.md          ← one-page summary
│   ├── assets/                ← figures used by the README
│   ├── linear_algebra.ipynb   ← notebook
│   ├── linear_algebra.py      ← its Jupytext twin
│   └── ...
├── 01_Supervised_Regression/        (same layout for every module)
├── 02_Supervised_classification/
├── 03_Model_evaluation_selection/
├── 04_Unsupervised_learning/
├── 05_Neural_networks_deep_learning/
├── 06_Generative_models/
├── 07_Reinforcement_learning/
├── 08_Other_topics/
├── docs/                      ← interactive explorables (HTML/JS, served by GitHub Pages)
├── mlf_utils/                 ← shared helpers: checks, plotting, graph drawing
├── jupytext.toml              ← .ipynb ↔ .py pairing
├── pyproject.toml             ← makes mlf_utils installable (locally and on Colab)
├── requirements.txt
└── LICENSE
```

---

## Contact

**Sandesh Jung Kunwar**
- GitHub: [sandeshjung](https://github.com/sandeshjung)
- Email: [sandeshjkunwar@gmail.com](mailto:sandeshjkunwar@gmail.com)
- LinkedIn: [Sandesh Jung Kunwar](https://www.linkedin.com/in/sandeshjung/)

## License

This project is licensed under the [MIT License](LICENSE).