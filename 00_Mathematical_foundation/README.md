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