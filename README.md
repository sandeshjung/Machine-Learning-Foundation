# Machine Learning Foundation

This repository is my personal collection of resources for learning the foundations of machine learning. It goes through the essential math, core algorithms, and practical implementations that make up the basics of modern machine learning and AI.

Whether you’re just starting out or want to refresh your understanding, you’ll find:

* **Clear explanations** of important ML concepts, with **mathematical derivations** in each module's README
* **Step-by-step implementations** from scratch, each **verified against a library** (PyTorch, scikit-learn, SciPy)
* **Interactive widgets** for building intuition: learning rate, polynomial degree, regularisation strength, SVM `C`/`gamma`, K, perplexity, a VAE latent space and more
* **One-page cheat sheets** per module with the key equations, hyperparameters and common pitfalls
* **Practical examples** with real datasets, from linear models, trees and ensembles to CNNs, Transformers, VAEs/GANs and RL
* **The practical side:** evaluation metrics, imbalance and calibration, preprocessing pipelines, data leakage, fairness and interpretability

I put this together mainly to document what I’ve studied and to make it easier to revise these concepts. It’s not a complete course from scratch, but it’s a good starting point if you want to get your hands dirty with ML.

## Getting Started

### Option 1: Google Colab (no setup)
Click the <a href="https://colab.research.google.com"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="16"></a> badge at the top of any notebook, or the ↗ links in the table below. The first cell installs everything the notebook needs. Switch to a GPU runtime (*Runtime → Change runtime type*) for the deep learning, generative and RL notebooks.

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

Notes:
- The autograd notebooks in `00_Mathematical_foundation` draw computation graphs with Graphviz, which also needs the system binary (`brew install graphviz`, `sudo apt install graphviz`, or see [graphviz.org/download](https://graphviz.org/download/)).
- Datasets (CIFAR-10, Fashion-MNIST, EMNIST, UCI Adult) are downloaded automatically into a `data/` folder next to each notebook on first run.
- The deep learning, generative and RL notebooks train models and run much faster on a GPU, but all of them work on CPU.
- Interactive widgets need a live kernel (locally or in Colab). GitHub only renders the static notebook.

## Notebooks

Suggested order: read a module's README (theory), work through its notebooks (implementation), then keep its cheat sheet for revision. ↗ opens the notebook in Colab.

| # | Module | Notebooks | Summary |
|---|---|---|---|
| 00 | [Mathematical foundation](00_Mathematical_foundation/README.md) | [autograd(scalar)](00_Mathematical_foundation/autograd%28scalar%29.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/autograd%28scalar%29.ipynb) · [autograd(tensor)](00_Mathematical_foundation/autograd%28tensor%29.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/autograd%28tensor%29.ipynb) · [calculus_optimization](00_Mathematical_foundation/calculus_optimization.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/calculus_optimization.ipynb) · [linear_algebra](00_Mathematical_foundation/linear_algebra.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/linear_algebra.ipynb) · [optimization_algorithms](00_Mathematical_foundation/optimization_algorithms.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/optimization_algorithms.ipynb) · [probability_statistics](00_Mathematical_foundation/probability_statistics.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/00_Mathematical_foundation/probability_statistics.ipynb) | [cheat sheet](00_Mathematical_foundation/CHEATSHEET.md) |
| 01 | [Supervised regression](01_Supervised_Regression/README.md) | [linear_regression](01_Supervised_Regression/linear_regression.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/linear_regression.ipynb) · [polynomial_overfitting](01_Supervised_Regression/polynomial_overfitting.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/polynomial_overfitting.ipynb) · [regularization](01_Supervised_Regression/regularization.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/01_Supervised_Regression/regularization.ipynb) | [cheat sheet](01_Supervised_Regression/CHEATSHEET.md) |
| 02 | [Supervised classification](02_Supervised_classification/README.md) | [logistic_regression](02_Supervised_classification/logistic_regression.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/logistic_regression.ipynb) · [naive_bayes](02_Supervised_classification/naive_bayes.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/naive_bayes.ipynb) · [svm_kernels](02_Supervised_classification/svm_kernels.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/svm_kernels.ipynb) · [knn](02_Supervised_classification/knn.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/knn.ipynb) · [decision_trees](02_Supervised_classification/decision_trees.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/decision_trees.ipynb) · [ensembles](02_Supervised_classification/ensembles.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/02_Supervised_classification/ensembles.ipynb) | [cheat sheet](02_Supervised_classification/CHEATSHEET.md) |
| 03 | [Model evaluation & selection](03_Model_evaluation_selection/README.md) | [bias_variance](03_Model_evaluation_selection/bias_variance.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/bias_variance.ipynb) · [cross_validation](03_Model_evaluation_selection/cross_validation.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/cross_validation.ipynb) · [hyperparameter_tuning](03_Model_evaluation_selection/hyperparameter_tuning.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/hyperparameter_tuning.ipynb) · [classification_metrics](03_Model_evaluation_selection/classification_metrics.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/classification_metrics.ipynb) · [preprocessing_pipelines](03_Model_evaluation_selection/preprocessing_pipelines.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/03_Model_evaluation_selection/preprocessing_pipelines.ipynb) | [cheat sheet](03_Model_evaluation_selection/CHEATSHEET.md) |
| 04 | [Unsupervised learning](04_Unsupervised_learning/README.md) | [kmeans_hierarchical](04_Unsupervised_learning/kmeans_hierarchical.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/04_Unsupervised_learning/kmeans_hierarchical.ipynb) · [pca_tsne](04_Unsupervised_learning/pca_tsne.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/04_Unsupervised_learning/pca_tsne.ipynb) | [cheat sheet](04_Unsupervised_learning/CHEATSHEET.md) |
| 05 | [Neural networks & deep learning](05_Neural_networks_deep_learning/README.md) | [cnn](05_Neural_networks_deep_learning/cnn.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/cnn.ipynb) · [mlp_backprop](05_Neural_networks_deep_learning/mlp_backprop.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/mlp_backprop.ipynb) · [optimizers_regularization](05_Neural_networks_deep_learning/optimizers_regularization.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/optimizers_regularization.ipynb) · [rnn](05_Neural_networks_deep_learning/rnn.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/rnn.ipynb) · [transformer](05_Neural_networks_deep_learning/transformer.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/05_Neural_networks_deep_learning/transformer.ipynb) | [cheat sheet](05_Neural_networks_deep_learning/CHEATSHEET.md) |
| 06 | [Generative models](06_Generative_models/README.md) | [gan](06_Generative_models/gan.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/gan.ipynb) · [vae](06_Generative_models/vae.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/06_Generative_models/vae.ipynb) | [cheat sheet](06_Generative_models/CHEATSHEET.md) |
| 07 | [Reinforcement learning](07_Reinforcement_learning/README.md) | [dqn_and_actor_critic](07_Reinforcement_learning/dqn_and_actor_critic.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/07_Reinforcement_learning/dqn_and_actor_critic.ipynb) · [rl_basics](07_Reinforcement_learning/rl_basics.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/07_Reinforcement_learning/rl_basics.ipynb) | [cheat sheet](07_Reinforcement_learning/CHEATSHEET.md) |
| 08 | [Fairness & interpretability](08_Other_topics/README.md) | [fairness_interpretability](08_Other_topics/fairness_interpretability.ipynb) [↗](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/08_Other_topics/fairness_interpretability.ipynb) | [cheat sheet](08_Other_topics/CHEATSHEET.md) |

## How the notebooks are built

- **Verified implementations.** Every from-scratch implementation is checked against a library reference with `check_close` / `check_agreement`, which raise an error on mismatch. For example: autograd against `torch.autograd`, PCA against `sklearn.decomposition.PCA`, attention against `F.scaled_dot_product_attention`, Adam against `torch.optim.Adam`, and the VAE's KL term against `torch.distributions`. Exact methods must match to float32 precision. Iterative ones (gradient descent) are compared with a tolerance explained next to each check.
- **Shared helpers.** Plumbing that several notebooks need (image grids, decision-region plots, autograd graph drawing, the checks) lives in the small [`mlf_utils`](mlf_utils/) package. The algorithms being taught stay inside the notebooks.
- **Paired `.py` files.** Each notebook has a plain-Python twin (via [Jupytext](https://jupytext.readthedocs.io), configured in [`jupytext.toml`](jupytext.toml)) for readable diffs and code review. With `jupytext` installed (it's in `requirements.txt`), Jupyter keeps both files in sync when you save. After editing a notebook elsewhere, for example in Colab, run `jupytext --sync */*.ipynb`.

## Structure

```
Machine-Learning-Foundation
├── 00_Mathematical_foundation/
│   ├── README.md                 # theory & derivations
│   ├── CHEATSHEET.md             # one-page summary
│   ├── assets/                   # figures used by the README
│   ├── linear_algebra.ipynb      # implementation
│   ├── linear_algebra.py         # Jupytext twin of the notebook
│   └── ...
├── 01_Supervised_Regression/     # same layout for every module
├── 02_Supervised_classification/
├── 03_Model_evaluation_selection/
├── 04_Unsupervised_learning/
├── 05_Neural_networks_deep_learning/
├── 06_Generative_models/
├── 07_Reinforcement_learning/
├── 08_Other_topics/
├── mlf_utils/                    # shared helpers (checks, plotting, graph drawing)
├── jupytext.toml                 # .ipynb <-> .py pairing
├── pyproject.toml                # makes mlf_utils installable (locally and on Colab)
├── requirements.txt
├── LICENSE
└── README.md
```

## Contact

**Sandesh Jung Kunwar**
- GitHub: [sandeshjung](https://github.com/sandeshjung)
- Email: [sandeshjkunwar@gmail.com](mailto:sandeshjkunwar@gmail.com)
- LinkedIn: [Sandesh Jung Kunwar](https://www.linkedin.com/in/sandeshjung/)

## License

This project is licensed under the [MIT License](LICENSE).