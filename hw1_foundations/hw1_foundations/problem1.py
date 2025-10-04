import numpy as np

np.random.seed(42)
n, d = 5, 3
X = np.random.randn(n, d)
w = np.random.randn(d)

print("X shape: ", X.shape)
print("w shape: ", w.shape)
print("=" * 50)

print("(i) Xw")
einsum_result = np.einsum('nd, d->n', X, w)
matmul_result = X @ w 

print(einsum_result)
print(matmul_result)
print(np.allclose(einsum_result, matmul_result))
print(einsum_result.shape)

print("(ii) XX^T")

einsum_result = np.einsum('nd, md->nm', X, X)
matmul_result = X @ X.T
print(einsum_result)
print(matmul_result)
print(np.allclose(einsum_result, matmul_result))
print(einsum_result.shape)

print("(iii) diag(X^T X): ")
einsum_result = np.einsum('nd, nd->d', X, X)
manual_result = np.diag(X.T @ X)
column_norms = np.sum(X**2, axis=0)

print(einsum_result)
print(manual_result)
print(column_norms)
print(np.allclose(einsum_result, manual_result))
print(np.allclose(einsum_result, column_norms))
print(einsum_result.shape)