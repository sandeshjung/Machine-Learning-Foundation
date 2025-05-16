# Mathematical Foundation

## Linear Algebra Fundamentals for Machine Learning

This provides a practical demonstration of key Linear Algebra concepts crucial for understanding and implementing Machine Learning algorithms. For Linear Algebra, scalars, vectors, matrices, along with essential operations and decompositions like Eigen-decomposition and Singular Value Decomposition (SVD) are explored.

The examples are primarily shown using [NumPy](https://numpy.org/) and [PyTorch](https://pytorch.org/).

### 1. Core Concepts

- **Scalars**
    - A scalar is a single numerical value. 
    - Representation: 
        - Python: Standard Numbers (e.g., `5`, `3.14`, etc.).
        - NumPy: 0-dimensional `ndarray` (e.g., `np.array(5)`).
        - PyTorch: 0-dimensional `torch.Tensor` (e.g., `torch.tensor(5.0)`).
    - ML Relevance: Learning rates, regularization parameters, individual feature values, loss values.

- **Vectors**
    - A 2-dimensional array (grid) of numbers, arranged in rows and columns.
    - Representation:
        - NumPy: 2-dimensional `ndarray` (e.g., `np.array([[1,2],[3,4]])`).
        - PyTorch: 2-dimensional `torch.Tensor` (e.g., `torch.tensor([[1.,2.],[3.,4.]])`).
    - ML Relevance: Datasets (samples x features), weight matrices in neural networks, covariance matrices, transformation matrices.

- **Matrices**
    - A 2-dimensional array (grid) of numbers, arranged in rows and columns.
    - Representation: 
        - NumPy: 2-dimensional `ndarray` (e.g., `np.array([[1,2],[3,4]])`).
        - PyTorch: 2-dimensional `torch.Tensor` (e.g., `torch.tensor([[1.,2.],[3.,4.]])`).
    - ML Relevance: Datasets (samples × features), weight matrices in neural networks, covariance matrices, transformation matrices.

### 2. Vector Operations

- **Vector Addition & Substraction**
    - Element-wise addition or subtraction. Vectors must have the same dimensions.
    - `c = a + b` => `c_i = a_i + b_i`

- **Vector Scalar Multiplication**
    - Multiplying each element of the vector by a scalar.
    - `b = s * a` => `b_i = s * a_i`   

- **Dot Product**
    - Sum of the products of corresponding elements of two vectors. The result is a scalar.
    - `a · b = Σ (a_i * b_i)` = a₁b₁ + a₂b₂ + ... + aₙbₙ`
    - Geometrically: `a · b = ||a|| ||b|| cos(θ)`, where `θ` is the angle between `a` and `b`. 
    - ML Relevance: Core of linear models (`w^T * x`), calculating projections, similarity measures.

- **Norm (Magnitude/Length)**
    - A function that assigns a strictly positive length or size to each vector in a vector space (except for the zero vector, which has a norm of zero).
    - L2 Norm (Euclidean Norm): `√(x₁² + x₂² + ...)`. Standard notion of length.
    - L1 Norm (Manhattan Norm): `|x₁| + |x₂| + ...`. Sum of absolute values.
    - ML Relevance: Regularization (L1/L2 penalties in Lasso/Ridge regression), distance calculations, error measurement.

### 3. Matrix Operations

- **Matrix Addition & Subtraction**
    - Element-wise addition or subtraction. Matrices must have the same dimensions.
    - `C = A + B` => `C_ij = A_ij + B_ij`

- **Matrix Scalar Multiplication**
    - Multiplying each element of the matrix by a scalar.
    - `B = s * A` => `B_ij = s * A_ij`

- **Matrix Transpose**
    - Swaps rows and columns of a matrix. `(A^T)_ij = A_ji`.
    - If `A` is `m × n`, then `A^T` is `n × m`.

- **Matrix Multiplication**
    - If `A` is `m × n` and `B` is `n × p`, their product `C = A @ B` is `m × p`.
    - `C_ij = Σ_k (A_ik * B_kj)` (sum over common dimension `k`).
    - The number of columns in the first matrix must equal the number of rows in the second.
    - ML Relevance: Fundamental for linear transformations, composing operations in neural networks (layer outputs `y = Wx + b`), solving systems of equations.

- **Identity Matrix (`I`)**
    - A square matrix with `1`s on the main diagonal and `0`s elsewhere.
    - Multiplying any matrix `A` by an appropriately sized identity matrix `I` leaves `A` unchanged (`A @ I = A`, `I @ A = A`).

- **Matrix Inverse (`A⁻¹`)**
    - For a square matrix `A`, its inverse `A⁻¹` is a matrix such that `A @ A⁻¹ = I` and `A⁻¹ @ A = I`.
    - Only non-singular (determinant ≠ 0) square matrices have an inverse.
    - ML Relevance: Solving systems of linear equations (e.g., Normal Equation in Linear Regression: `θ = (X^T X)⁻¹ X^T y`), theoretical derivations.

### 4. Matrix Decomposition & Advanced Concepts
- **Eigenvalues and Eigenvectors**
    - For a square matrix `A`, an **eigenvector** `v` is a non-zero vector that, when transformed by `A`, only changes in scale (not direction). The scaling factor is the **eigenvalue** `λ`.
    - Equation: `A @ v = λ * v`
    - Eigen-decomposition: If a matrix `A` is diagonalizable, it can be written as `A = V @ diag(Λ) @ V⁻¹`, where:
        - `V` is a matrix whose columns are the eigenvectors of `A`.
        - `diag(Λ)` is a diagonal matrix whose diagonal entries are the corresponding eigenvalues.
        - For symmetric matrices, `V` is orthogonal, so `V⁻¹ = V^T`.
    - ML Relevance: Principal Component Analysis (PCA) uses eigenvalues/vectors of the covariance matrix to find directions of maximum variance. Understanding stability of systems, graph analysis (spectral clustering).

- **Singular Value Decomposition (SVD)**
    - A factorization of *any* `m × n` matrix `A` into three matrices: `A = U @ Σ @ Vᵀ` (or `U @ Σ @ Vh` in PyTorch where `Vh` is conjugate transpose).
        - `U`: `m × m` orthogonal matrix (left singular vectors).
        - `Σ` (Sigma): `m × n` diagonal matrix with non-negative real numbers called singular values on its diagonal, sorted in descending order.
        - `Vᵀ` (or `Vh`): `n × n` orthogonal matrix (transpose of right singular vectors).
    - Properties:
        - Singular values are the square roots of the non-zero eigenvalues of `A^T @ A` (or `A @ A^T`).
        - Always exists for any matrix.
    - ML Relevance:
        - Dimensionality Reduction: PCA can be performed using SVD. Truncated SVD (keeping top `k` singular values) gives the best rank-`k` approximation of the matrix.
        - Recommender Systems: Matrix factorization techniques often rely on SVD-like methods.
        - Noise Reduction: Smaller singular values often correspond to noise.
        - Solving Linear Systems: Calculating pseudo-inverse.

### 5. Why is this important for ML?

Linear Algebra is the language of data and models in Machine Learning:

- Data Representation: Datasets are typically represented as matrices (samples × features). Individual data points are vectors.
- Model Parameters: Many models (Linear Regression, Logistic Regression, Neural Networks) have parameters (weights, biases) organized as vectors or matrices.
- Transformations: Linear transformations (matrix multiplications) are fundamental to how many models process data (e.g., layers in a neural network).
- Optimization: Gradients (vectors) are used in optimization algorithms (like Gradient Descent) to update model parameters. Hessians (matrices of second derivatives) can also be used.
- Dimensionality Reduction: Techniques like PCA rely heavily on eigendecomposition or SVD to find more compact representations of data.
- Similarity & Distance: Dot products and norms are used to measure similarity or distance between data points or vectors.
- Underlying Theory: Many ML algorithms have their roots and derivations firmly planted in linear algebra. Understanding it allows for a deeper comprehension of *why* algorithms work and how to troubleshoot or improve them.

## Probability and Statistics Fundamentals

This provides a practical demonstration of Probability and Statistics that are essential for understanding and building Machine Learning models.Common probability distributions, the intuition and application of Bayes' Theorem, and various sampling techniques are explored.

### 1. Introduction to Probability & Statistics in ML
Probability theory provides a framework for quantifying uncertainty, while statistics offers tools to analyze and interpret data, draw inferences, and make predictions. In Machine Learning:
- Probability helps us model the inherent randomness or uncertainty in data and in model predictions.
- Statistics provides methods for learning from data, estimating model parameters, evaluating model performance, and testing hypotheses.

### 2. Probability Distributions

A *probability distribution* is a mathematical function that describes the likelihood of different possible outcomes for a random variable.

#### 2.1 Discrete Distributions

For random variables that can take on a finite or countably infinite number of distinct values.

- **Bernoulli Distribution**
    - Models a single trial with two possible outcomes (e.g., success/failure, 0/1, head/tail).
    - Parameter: $p$ (probability of success, i.e., outcome 1).
    - PMF: $\large P(X=k) = p^k (1-p)^{1-k}$ for $\large k \in \{0, 1\}$.
    - ML Relevance: Modeling binary classification outputs, presence/absence of a feature.

- **Binomial Distribution**
    - Models the number of successes in a fixed number, $n$, of independent Bernoulli trials.
    - Parameters: $n$ (total number of trials), $p$ (probability of success in each trial).
    - PMF: $\large P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$ for $\large k \in \{0, 1, ..., n\}$.
    - ML Relevance: Analyzing success counts in experiments, modeling click-through rates over multiple impressions.

- **Categorical Distribution**
    - Generalizes the Bernoulli distribution to a single trial with $\large K$ possible outcomes (categories).
    - Parameter: A vector of $\large K$ probabilities $\large \mathbf{p} = [p_1, p_2, ..., p_K]$, where $\large p_i \ge 0$ and $\large \sum p_i = 1$.
    - PMF: $\large P(X=k_i) = p_i$.
    - ML Relevance: Modeling outputs of multi-class classification (e.g., Softmax output), representing discrete latent variables.

#### 2.2 Continuous Distributions

For random variables that can take on any value within a continuous range.

- **Uniform Distribution**
    - All values within a given range $\large [a, b]$ are equally likely.
    - Parameters: $\large a$ (lower bound), $\large b$ (upper bound).
    - PDF: $\large f(x) = \frac{1}{b-a}$ for $\large a \le x \le b$, and $\large 0$ otherwise.
    - ML Relevance: Initializing weights in neural networks within a certain range, representing a lack of prior knowledge over an interval.

- **Normal (Gaussian) Distribution**
    - The ubiquitous "bell curve," characterized by its mean and standard deviation. Central Limit Theorem states that the sum/average of many independent random variables tends towards a normal distribution.
    - Parameters: $\large \mu$ (mean, location), $\large \sigma^2$ (variance; $\large \sigma$ is standard deviation, scale).
    - PDF: $\large f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.
    - ML Relevance: Modeling noise in data, prior distributions for model parameters (Bayesian ML), basis for Gaussian Mixture Models, assumption in many statistical tests.

### Key Concepts for Distributions

#### Probability Mass Function (PMF) / Probability Density Function (PDF)
- PMF (Discrete): $\large P(X=x)$ gives the probability that a discrete random variable $\large X$ is exactly equal to some value $\large x$.
    - Properties: $\large P(X=x) \ge 0$ for all $\large x$, and $\large \sum_x P(X=x) = 1$.
- PDF (Continuous): $\large f(x)$ describes the relative likelihood for a continuous random variable $X$ to take on a given value $\large x$. The probability of $X$ falling within a range $\large [c,d]$ is given by the integral of the PDF over that range: $\large P(c \le X \le d) = \int_c^d f(x)dx$.
    - Properties: $\large f(x) \ge 0$ for all $\large x$, and $\large \int_{-\infty}^{\infty} f(x)dx = 1$. Note that for continuous variables, the probability of $X$ being exactly equal to a single point $\large x$ is $\large P(X=x) = 0$.

#### Cumulative Distribution Function (CDF)
- The CDF, denoted $\large F(x)$, gives the probability that the random variable $X$ takes on a value less than or equal to $\large x$.
    $\large F(x) = P(X \le x)$
- For Discrete Variables:
    $\large F(x) = \sum_{x_i \le x} P(X=x_i)$
- For Continuous Variables:
    $\large F(x) = \int_{-\infty}^{x} f(t)dt$
    (where $\large f(t)$ is the PDF).
- Properties:
    1. $\large 0 \le F(x) \le 1$.
    2. $\large F(x)$ is a non-decreasing function (i.e., if $\large a < b$, then $\large F(a) \le F(b)$).
    3. $\large \lim_{x\to-\infty} F(x) = 0$.
    4. $\large \lim_{x\to\infty} F(x) = 1$.
    5. For a continuous random variable, $\large P(a < X \le b) = F(b) - F(a)$.
    6. The PDF can be obtained from the CDF (for continuous variables) by differentiation: $\large f(x) = \frac{dF(x)}{dx}$.
- ML Relevance: Generating random samples via inverse transform sampling, calculating p-values.

### 3. Bayes' Theorem

#### Formula and Intuition
Bayes' Theorem is a fundamental theorem in probability that describes how to update the probability of a hypothesis based on new evidence.

- **Formula:**
    $\large  P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$
  
    Where:
    - $\large P(H|E)$: **Posterior probability** – Probability of hypothesis $\large H$ given evidence $\large E$.
    - $\large P(E|H)$: **Likelihood** – Probability of observing evidence $\large E$ if hypothesis $\large H$ is true.
    - $\large P(H)$: **Prior probability** – Initial belief in hypothesis $\large H$ before observing $\large E$.
    - $\large P(E)$: **Evidence (or Marginal Likelihood)** – Total probability of observing evidence $\large E$.
        It acts as a normalization constant: $\large P(E) = \sum_i P(E|H_i) P(H_i)$ over all possible mutually exclusive hypotheses $\large H_i$. For a single hypothesis $\large H$ and its complement $\large \neg H$: $\large P(E) = P(E|H)P(H) + P(E|\neg H)P(\neg H)$.

#### Example Application
The notebook demonstrates a classic medical diagnosis example, showing how a prior belief about a disease's prevalence is updated given the result of a diagnostic test with known sensitivity and specificity.

- **ML Relevance:**
    - Foundation of Naive Bayes classifiers.
    - Core of Bayesian Machine Learning (e.g., Bayesian Linear Regression, Bayesian Neural Networks) where parameters are treated as random variables with prior distributions, updated to posterior distributions given data.
    - Used in spam filtering, A/B testing analysis, and many other areas involving updating beliefs.

### 4. Sampling Techniques

Sampling is the process of selecting a subset of individuals, items, or data points from a larger population to estimate characteristics of or draw inferences about the whole population.

#### Sampling from Distributions
- Generating random numbers that follow a specific probability distribution.
- Code: The notebook shows how to use `torch.distributions.Distribution.sample()` for various distributions.
- ML Relevance: Generating synthetic data, Monte Carlo methods (e.g., for integration or optimization), initializing model parameters, dropout in neural networks.

#### Simple Random Sampling from a Dataset
- Each item in the population has an equal chance of being selected. Can be done with or without replacement.
- Code: Demonstrated using `torch.randperm` and `numpy.random.choice` on tensor indices.
- ML Relevance: Creating training/testing splits (though often stratified sampling is preferred for classification), bootstrapping.

#### Stratified Sampling (Conceptual)
- The population is divided into homogeneous subgroups (strata), and simple random samples are drawn from each stratum, often proportionally to the stratum's size in the population.
- Benefit: Ensures that the sample accurately reflects the population's structure with respect to the stratification variable(s), especially useful for small or imbalanced subgroups.
- ML Relevance: Crucial for creating representative training and testing sets in classification tasks, especially with imbalanced classes, to ensure model evaluation is reliable. (Often implemented using libraries like `scikit-learn`).

### 5. Why is this important for ML?

A solid grasp of probability and statistics is indispensable for a Machine Learning practitioner:

- **Understanding Algorithms:** Many ML algorithms are derived from probabilistic principles (e.g., Naive Bayes, Logistic Regression as a GLM, Gaussian Mixture Models, Hidden Markov Models).
- **Model Building & Interpretation:** Choosing appropriate likelihood functions for loss functions, understanding uncertainty in predictions (e.g., confidence intervals, prediction intervals).
- **Data Analysis & Preprocessing:** Understanding data distributions helps in feature engineering, outlier detection, and choosing data transformations.
- **Model Evaluation:** Statistical hypothesis testing is used to compare models, assess significance of results, and interpret p-values.
- **Bayesian Methods:** A whole subfield of ML relies on Bayesian statistics for more robust modeling, uncertainty quantification, and incorporating prior knowledge.
- **Generative Models:** Models that learn the underlying probability distribution of the data (e.g., GANs, VAEs) are inherently probabilistic.
- **Reinforcement Learning:** Involves agents learning in probabilistic environments.
