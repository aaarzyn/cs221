#!/usr/bin/env python3

import collections
import grader_util
import random

grader = grader_util.Grader()
submission = grader.load('submission')

############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
    warnings.warn("Must be using Python 3.12 \n")



############################################################
##### Problem 1 (Linear Algebra) ###########################
############################################################

grader.add_manual_part('1a', max_points=2, description='NumPy tutor session link')
grader.add_manual_part('1b', max_points=3, description='Matrix multiplication complexity')
grader.add_manual_part('1c', max_points=2, description='Einsum tutor session link')
grader.add_manual_part('1d', max_points=3, description='Einstein summation (written)')

# Problem 1e: linear_project
def test1e0():
    import numpy as np
    x = np.array([[1.0, 2.0, 3.0],
                  [0.0, -1.0, 4.0]])  # (B=2, Din=3)
    W = np.array([[1.0, 0.0],
                  [0.5, -1.0],
                  [2.0, 3.0]])        # (Din=3, Dout=2)
    b = np.array([0.1, -0.2])         # (Dout=2)
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)


def test1e1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        B, Din, Dout = 4, 5, 3
        x = rng.standard_normal((B, Din))
        W = rng.standard_normal((Din, Dout))
        b = rng.standard_normal(Dout)
        expected = x @ W + b
        out = submission.linear_project(x, W, b)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1e-0-basic', test1e0, max_points=1, description='linear_project with bias (small deterministic)')
grader.add_basic_part('1e-1-basic', test1e1, max_points=2, description='linear_project randomized, with bias')

# Problem 1f: split_last_dim pattern string
def test1f0():
    import numpy as np
    from einops import rearrange
    x = np.arange(12, dtype=float).reshape(2, 6)  # (B=2, D=6)
    num_groups = 3
    expected = x.reshape(2, num_groups, 6 // num_groups)
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    grader.require_is_equal(expected, out)


def test1f1():
    import numpy as np
    from einops import rearrange
    rng = np.random.default_rng(1)
    for _ in range(3):
        B, num_groups = 3, 4
        D = 20
        x = rng.standard_normal((B, D))
        expected = x.reshape(B, num_groups, D // num_groups)
        pattern = submission.split_last_dim_pattern()
        out = rearrange(x, pattern, g=num_groups)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1f-0-basic', test1f0, max_points=1, description='split_last_dim pattern applies correctly')
grader.add_basic_part('1f-1-basic', test1f1, max_points=2, description='split_last_dim pattern randomized')

# Problem 1g: normalized_inner_products
def test1g0():
    import numpy as np
    A = np.array([[[1., 0.], [0., 1.]]])  # (B=1, M=2, D=2)
    Bm = np.array([[[1., 2.], [3., 4.], [0., 1.]]])  # (B=1, N=3, D=2)
    expected = np.einsum('bmd,bnd->bmn', A, Bm)
    out = submission.normalized_inner_products(A, Bm, normalize=False)
    grader.require_is_equal(expected, out)
    # normalized
    D = A.shape[-1]
    expected_norm = expected / np.sqrt(D)
    out_norm = submission.normalized_inner_products(A, Bm, normalize=True)
    grader.require_is_equal(expected_norm, out_norm)


def test1g1():
    import numpy as np
    rng = np.random.default_rng(2)
    for _ in range(3):
        B, M, N, D = 2, 3, 4, 5
        A = rng.standard_normal((B, M, D))
        Bm = rng.standard_normal((B, N, D))
        exp = np.einsum('bmd,bnd->bmn', A, Bm)
        out = submission.normalized_inner_products(A, Bm, normalize=False)
        grader.require_is_equal(exp, out)
        expn = exp / np.sqrt(D)
        outn = submission.normalized_inner_products(A, Bm, normalize=True)
        grader.require_is_equal(expn, outn)


grader.add_basic_part('1g-0-basic', test1g0, max_points=1, description='normalized_inner_products small case + normalization')
grader.add_basic_part('1g-1-basic', test1g1, max_points=2, description='normalized_inner_products randomized')

# Problem 1h: mask_strictly_upper
def test1h0():
    import numpy as np
    B, L = 1, 4
    scores = np.arange(B * L * L, dtype=float).reshape(B, L, L)
    out = submission.mask_strictly_upper(scores.copy())
    expected = scores.copy()
    triu_rows, triu_cols = np.triu_indices(L, k=1)
    expected[:, triu_rows, triu_cols] = -np.inf
    # Check non-inf values first
    non_inf_mask = ~np.isinf(expected)
    grader.require_is_equal(expected[non_inf_mask], out[non_inf_mask])
    # Check inf values separately
    inf_mask = np.isinf(expected)
    grader.require_is_equal(np.all(np.isinf(out[inf_mask])), True)


def test1h1():
    import numpy as np
    rng = np.random.default_rng(3)
    for _ in range(3):
        B, L = 2, 5
        scores = rng.standard_normal((B, L, L))
        out = submission.mask_strictly_upper(scores.copy())
        expected = scores.copy()
        rr, cc = np.triu_indices(L, k=1)
        expected[:, rr, cc] = -np.inf
        # Check non-inf values first
        non_inf_mask = ~np.isinf(expected)
        grader.require_is_equal(expected[non_inf_mask], out[non_inf_mask])
        # Check inf values separately
        inf_mask = np.isinf(expected)
        grader.require_is_equal(np.all(np.isinf(out[inf_mask])), True)


grader.add_basic_part('1h-0-basic', test1h0, max_points=1, description='mask_strictly_upper sets upper triangle to -inf')
grader.add_basic_part('1h-1-basic', test1h1, max_points=2, description='mask_strictly_upper randomized')

# Problem 1i: prob_weighted_sum einsum string
def test1i0():
    import numpy as np
    from einops import einsum
    P = np.array([[0.25, 0.25, 0.5]])  # (B=1, N=3)
    V = np.array([[[1., 0.], [0., 1.], [2., 2.]]])  # (B=1, N=3, D=2)
    expected = einsum(P, V, 'b n, b n d -> b d')
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    grader.require_is_equal(expected, out)


def test1i1():
    import numpy as np
    from einops import einsum
    rng = np.random.default_rng(4)
    for _ in range(3):
        B, N, D = 2, 5, 3
        P = rng.random((B, N))
        P = P / P.sum(axis=1, keepdims=True)
        V = rng.standard_normal((B, N, D))
        expected = einsum(P, V, 'b n, b n d -> b d')
        pattern = submission.prob_weighted_sum_einsum()
        out = einsum(P, V, pattern)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1i-0-basic', test1i0, max_points=1, description='prob_weighted_sum einsum string deterministic')
grader.add_basic_part('1i-1-basic', test1i1, max_points=2, description='prob_weighted_sum einsum string randomized')
############################################################
##### Problem 2 (Calculus & Gradients) #####################
############################################################

grader.add_manual_part('2a', max_points=2, description='Gradient warmup')
grader.add_manual_part('2c', max_points=3, description='Matrix multiplication gradient')

# Problem 2b: gradient_warmup implementation
def test2b0():
    import numpy as np
    w = np.array([1.0, -2.0, 3.0])
    c = np.array([0.0, 1.0, -1.0])
    expected = 2.0 * (w - c)
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)


def test2b1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.integers(2, 8)
        w = rng.standard_normal(d)
        c = rng.standard_normal(d)
        expected = 2.0 * (w - c)
        out = submission.gradient_warmup(w, c)
        grader.require_is_equal(expected, out)


grader.add_basic_part('2b-0-basic', test2b0, max_points=1, description='gradient_warmup deterministic')
grader.add_basic_part('2b-1-basic', test2b1, max_points=2, description='gradient_warmup randomized')

# Problem 2d: matrix_grad implementation
def test2d0():
    import numpy as np
    A = np.array([[2., 1., 3.],
                  [4., 5., 6.]])  # (m=2, p=3)
    B = np.array([[7., 8.],
                  [9., 0.],
                  [1., 2.]])      # (p=3, n=2)
    grad_A, grad_B = submission.matrix_grad(A, B)
    # Expected gradients
    row_sum_B = B.sum(axis=1)  # (3,)
    col_sum_A = A.sum(axis=0)  # (3,)
    expected_grad_A = np.ones((A.shape[0], 1)) @ row_sum_B[None, :]
    expected_grad_B = col_sum_A[:, None] @ np.ones((1, B.shape[1]))
    grader.require_is_equal(expected_grad_A, grad_A)
    grader.require_is_equal(expected_grad_B, grad_B)


def test2d1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(3):
        m = rng.integers(2, 5)
        p = rng.integers(2, 5)
        n = rng.integers(2, 5)
        A = rng.standard_normal((m, p))
        B = rng.standard_normal((p, n))
        grad_A, grad_B = submission.matrix_grad(A, B)

        # Numeric check via central differences (function is linear; should match exactly)
        eps = 1e-6
        # Check a subset or all entries for small sizes
        num_grad_A = np.zeros_like(A)
        for i in range(m):
            for k in range(p):
                E = np.zeros_like(A)
                E[i, k] = eps
                s_plus = np.sum((A + E) @ B)
                s_minus = np.sum((A - E) @ B)
                num_grad_A[i, k] = (s_plus - s_minus) / (2 * eps)
        num_grad_B = np.zeros_like(B)
        for k in range(p):
            for j in range(n):
                E = np.zeros_like(B)
                E[k, j] = eps
                s_plus = np.sum(A @ (B + E))
                s_minus = np.sum(A @ (B - E))
                num_grad_B[k, j] = (s_plus - s_minus) / (2 * eps)
        grader.require_is_equal(num_grad_A, grad_A)
        grader.require_is_equal(num_grad_B, grad_B)

# Problem 2e: finite differences vs analytic gradient
def test2e0():
    import numpy as np
    rng = np.random.default_rng(42)
    n, d = 5, 4
    A = rng.standard_normal((n, d))
    b = rng.standard_normal(n)
    w = rng.standard_normal(d)
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    grader.require_is_equal(g_analytic, g_numeric)


def test2e1():
    import numpy as np
    rng = np.random.default_rng(99)
    for _ in range(5):
        n = rng.integers(3, 8)
        d = rng.integers(3, 8)
        A = rng.standard_normal((n, d))
        b = rng.standard_normal(n)
        w = rng.standard_normal(d)
        g_analytic = submission.lsq_grad(w, A, b)
        g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=3e-6)
        grader.require_is_equal(g_analytic, g_numeric)

grader.add_basic_part('2d-0-basic', test2d0, max_points=1, description='matrix_grad deterministic')
grader.add_basic_part('2d-1-basic', test2d1, max_points=2, description='matrix_grad randomized + numeric check')

grader.add_basic_part('2e-0-basic', test2e0, max_points=1, description='finite differences matches analytic gradient (single case)')
grader.add_basic_part('2e-1-basic', test2e1, max_points=2, description='finite differences matches analytic gradient (random cases)')


############################################################
##### Problem 3 (Optimization) #############################
############################################################

grader.add_manual_part('3a', max_points=3, description='Weighted scalar quadratic minimizer')
grader.add_manual_part('3b', max_points=2, description='Gradient descent tutor session link')

# Problem 3c: gradient_descent_quadratic (code)
def test3c0():
    import numpy as np
    # Simple deterministic case: weighted average should be the minimizer
    x = np.array([0.0, 10.0])
    w = np.array([1.0, 3.0])
    theta_star = (w * x).sum() / w.sum()
    theta0 = 100.0
    lr = 0.25 / w.sum()  # stable stepsize (< 1/(2*sum w))
    theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=200)
    grader.require_is_equal(theta_star, theta)


def test3c1():
    import numpy as np
    rng = np.random.default_rng(5)
    for _ in range(5):
        n = rng.integers(2, 8)
        x = rng.standard_normal(n)
        w = rng.random(n) + 0.1  # strictly positive
        theta_star = (w * x).sum() / w.sum()
        theta0 = rng.standard_normal()
        lr = 0.25 / w.sum()
        theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=300)
        grader.require_is_equal(theta_star, theta)


grader.add_basic_part('3c-0-basic', test3c0, max_points=1, description='gradient descent converges to weighted average (deterministic)')
grader.add_basic_part('3c-1-basic', test3c1, max_points=2, description='gradient descent converges on random instances')


############################################################
##### Problem 4 (Ethics in AI): written parts ##############
############################################################

grader.add_manual_part('4a', max_points=2, description='Ethics in AI part a')
grader.add_manual_part('4b', max_points=2, description='Ethics in AI part b')
grader.add_manual_part('4c', max_points=2, description='Ethics in AI part c')
grader.add_manual_part('4d', max_points=2, description='Ethics in AI part d')

############################################################
##### Code Extra Test Cases Written for the Vibes ##########
############################################################
def test1e2():
    import numpy as np
    x = np.array([[5.0]])
    W = np.array([[2.0]])
    b = np.array([1.0])
    expected = np.array([[11.0]])
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)

def test1e3():
    import numpy as np
    x = np.array([[1.0, 2.0]])
    W = np.array([[3.0], [4.0]])
    b = np.array([0.0])
    expected = np.array([[11.0]])
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)

grader.add_basic_part('1e-2-basic', test1e2, max_points=1, description='linear_project single dimension')
grader.add_basic_part('1e-3-basic', test1e3, max_points=1, description='linear_project zero bias')

def test1h2():
    import numpy as np
    B, L = 1, 2
    scores = np.array([[[1., 2.], [3., 4.]]], dtype=float)
    out = submission.mask_strictly_upper(scores.copy())
    grader.require_is_equal(out[0, 0, 0], 1.0)
    grader.require_is_equal(np.isinf(out[0, 0, 1]), True)
    grader.require_is_equal(out[0, 1, 0], 3.0)
    grader.require_is_equal(out[0, 1, 1], 4.0)

grader.add_basic_part('1h-2-basic', test1h2, max_points=1, description='mask_strictly_upper 2x2 simple case')

def test2b2():
    import numpy as np
    w = np.array([1.0, 2.0])
    c = np.array([1.0, 2.0])
    expected = np.array([0.0, 0.0])
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

def test3c2():
    import numpy as np
    x = np.array([5.0])
    w = np.array([1.0])
    theta0 = 0.0
    lr = 0.1
    theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=50)
    grader.require_is_equal(5.0, theta, tolerance=1e-3)

grader.add_basic_part('2b-2-basic', test2b2, max_points=1, description='gradient_warmup zero case')
grader.add_basic_part('3c-2-basic', test3c2, max_points=1, description='gradient descent single point')

def test2e2():
    import numpy as np
    A = np.eye(2)
    b = np.array([1., 2.])
    w = np.array([3., 4.])
    expected = w - b
    out = submission.lsq_grad(w, A, b)
    grader.require_is_equal(expected, out)

grader.add_basic_part('2e-2-basic', test2e2, max_points=1, description='lsq_grad identity matrix case')

def test1f2():
    pattern = submission.split_last_dim_pattern()
    grader.require_is_equal(pattern, "b (g d) -> b g d")

def test1i2():
    pattern = submission.prob_weighted_sum_einsum()
    grader.require_is_equal(pattern, "b n, b n d -> b d")

grader.add_basic_part('1f-2-basic', test1f2, max_points=1, description='split_last_dim pattern string check')
grader.add_basic_part('1i-2-basic', test1i2, max_points=1, description='prob_weighted_sum pattern string check')

def test1e_edge_cases():
    import numpy as np
    
    x = np.array([[-1.0, -2.0, -3.0]])
    W = np.array([[-1.0, 2.0], [3.0, -4.0], [0.5, 1.5]])
    b = np.array([-0.5, 0.5])
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)
    
    batch_size = 10
    x = np.random.randn(batch_size, 5)
    W = np.random.randn(5, 3)
    b = np.random.randn(3)
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)

def test1e_extreme_values():
    import numpy as np
    
    x = np.array([[1e-10, 2e-10]])
    W = np.array([[1e10], [2e10]])
    b = np.array([1.0])
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)

grader.add_basic_part('1e-edge-basic', test1e_edge_cases, max_points=1, description='linear_project edge cases')
grader.add_basic_part('1e-extreme-basic', test1e_extreme_values, max_points=1, description='linear_project extreme values')

def test1f_edge_cases():
    import numpy as np
    from einops import rearrange
    
    x = np.arange(6, dtype=float).reshape(1, 6)
    num_groups = 1
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    expected = x.reshape(1, 1, 6)
    grader.require_is_equal(expected, out)
    
    x = np.arange(8, dtype=float).reshape(2, 4)
    num_groups = 4
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    expected = x.reshape(2, 4, 1)
    grader.require_is_equal(expected, out)

def test1f_large_dimensions():
    import numpy as np
    from einops import rearrange
    
    B, D = 5, 24
    num_groups = 6
    x = np.random.randn(B, D)
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    expected = x.reshape(B, num_groups, D // num_groups)
    grader.require_is_equal(expected, out)

grader.add_basic_part('1f-edge-basic', test1f_edge_cases, max_points=1, description='split_last_dim edge cases')
grader.add_basic_part('1f-large-basic', test1f_large_dimensions, max_points=1, description='split_last_dim large dimensions')

def test1g_edge_cases():
    import numpy as np
    
    A = np.zeros((1, 2, 3))
    B = np.ones((1, 4, 3))
    out = submission.normalized_inner_products(A, B, normalize=False)
    expected = np.zeros((1, 2, 4))
    grader.require_is_equal(expected, out)
    
    A = np.array([[[1., 0., 0.]]])
    B = np.array([[[1., 0., 0.], [0., 1., 0.]]])
    out = submission.normalized_inner_products(A, B, normalize=False)
    expected = np.array([[[1., 0.]]])
    grader.require_is_equal(expected, out)

def test1g_normalization_effect():
    import numpy as np
    
    A = np.ones((1, 1, 4))
    B = np.ones((1, 1, 4))
    
    out_unnorm = submission.normalized_inner_products(A, B, normalize=False)
    out_norm = submission.normalized_inner_products(A, B, normalize=True)
    
    grader.require_is_equal(out_unnorm[0, 0, 0], 4.0)
    grader.require_is_equal(out_norm[0, 0, 0], 2.0)

grader.add_basic_part('1g-edge-basic', test1g_edge_cases, max_points=1, description='normalized_inner_products edge cases')
grader.add_basic_part('1g-norm-basic', test1g_normalization_effect, max_points=1, description='normalized_inner_products normalization effect')

def test1h_different_sizes():
    import numpy as np
    
    scores = np.arange(9, dtype=float).reshape(1, 3, 3)
    out = submission.mask_strictly_upper(scores.copy())
    
    grader.require_is_equal(out[0, 0, 0], 0.0)
    grader.require_is_equal(out[0, 1, 1], 4.0)
    grader.require_is_equal(out[0, 2, 2], 8.0)
    grader.require_is_equal(out[0, 1, 0], 3.0)
    grader.require_is_equal(out[0, 2, 0], 6.0)
    grader.require_is_equal(out[0, 2, 1], 7.0)
    
    grader.require_is_equal(np.isinf(out[0, 0, 1]), True)
    grader.require_is_equal(np.isinf(out[0, 0, 2]), True)
    grader.require_is_equal(np.isinf(out[0, 1, 2]), True)

def test1h_batch_consistency():
    import numpy as np
    
    B, L = 3, 4
    scores = np.random.randn(B, L, L)
    out = submission.mask_strictly_upper(scores.copy())
    
    for b in range(B):
        for i in range(L):
            for j in range(L):
                if j > i:
                    grader.require_is_equal(np.isinf(out[b, i, j]), True)
                else:
                    grader.require_is_equal(out[b, i, j], scores[b, i, j])

grader.add_basic_part('1h-sizes-basic', test1h_different_sizes, max_points=1, description='mask_strictly_upper different sizes')
grader.add_basic_part('1h-batch-basic', test1h_batch_consistency, max_points=1, description='mask_strictly_upper batch consistency')

def test1i_edge_cases():
    import numpy as np
    from einops import einsum
    
    P = np.ones((2, 4)) / 4
    V = np.random.randn(2, 4, 3)
    
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    expected = V.mean(axis=1)
    grader.require_is_equal(expected, out)

def test1i_single_hot():
    import numpy as np
    from einops import einsum
    
    P = np.array([[1., 0., 0.], [0., 0., 1.]])
    V = np.array([[[1., 2.], [3., 4.], [5., 6.]], 
                  [[7., 8.], [9., 10.], [11., 12.]]])
    
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    
    expected = np.array([[1., 2.], [11., 12.]])
    grader.require_is_equal(expected, out)

grader.add_basic_part('1i-edge-basic', test1i_edge_cases, max_points=1, description='prob_weighted_sum edge cases')
grader.add_basic_part('1i-onehot-basic', test1i_single_hot, max_points=1, description='prob_weighted_sum one-hot')

def test2b_large_dimensions():
    import numpy as np
    
    d = 100
    w = np.random.randn(d)
    c = np.random.randn(d)
    expected = 2.0 * (w - c)
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

def test2b_extreme_values():
    import numpy as np
    
    w = np.array([1e6, -1e6, 0.0])
    c = np.array([-1e6, 1e6, 1e-10])
    expected = 2.0 * (w - c)
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

grader.add_basic_part('2b-large-basic', test2b_large_dimensions, max_points=1, description='gradient_warmup large dimensions')
grader.add_basic_part('2b-extreme-basic', test2b_extreme_values, max_points=1, description='gradient_warmup extreme values')

def test2d_edge_cases():
    import numpy as np
    
    A = np.array([[5.0]])
    B = np.array([[3.0]])
    grad_A, grad_B = submission.matrix_grad(A, B)
    grader.require_is_equal(grad_A, np.array([[3.0]]))
    grader.require_is_equal(grad_B, np.array([[5.0]]))

def test2d_rectangular():
    import numpy as np
    
    A = np.ones((1, 5))
    B = np.ones((5, 1))
    grad_A, grad_B = submission.matrix_grad(A, B)
    
    grader.require_is_equal(grad_A, np.ones((1, 5)))
    grader.require_is_equal(grad_B, np.ones((5, 1)))

grader.add_basic_part('2d-edge-basic', test2d_edge_cases, max_points=1, description='matrix_grad edge cases')
grader.add_basic_part('2d-rect-basic', test2d_rectangular, max_points=1, description='matrix_grad rectangular matrices')

def test2e_consistency_check():
    import numpy as np
    
    A = np.array([[1., 2.], [3., 4.]])
    b = np.array([1., -1.])
    w = np.array([0.5, -0.5])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    
    error = np.linalg.norm(g_analytic - g_numeric)
    grader.require_is_true(error < 1e-4)

def test2e_different_points():
    import numpy as np
    
    A = np.array([[2., 1.], [1., 2.]])
    b = np.array([3., 4.])
    
    test_points = [
        np.array([0., 0.]),
        np.array([1., 1. ]),
        np.array([-1., 2.]),
        np.array([0.1, -0.3])
    ]
    
    for w in test_points:
        g_analytic = submission.lsq_grad(w, A, b)
        g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
        error = np.linalg.norm(g_analytic - g_numeric)
        grader.require_is_true(error < 1e-4)

def test2e_degenerate_case():
    import numpy as np
    
    A = np.zeros((3, 2))
    b = np.array([1., 2., 3.])
    w = np.array([1., 1.])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    
    grader.require_is_equal(g_analytic, np.zeros(2))
    grader.require_is_equal(g_numeric, np.zeros(2), tolerance=1e-4)

def test2e_epsilon_range():
    import numpy as np
    
    A = np.array([[1., 0.], [0., 1.]])
    b = np.array([2., 3.])
    w = np.array([1., 1.])
    
    g_analytic = submission.lsq_grad(w, A, b)
    
    epsilons = [1e-4, 1e-5, 1e-6, 1e-7]
    
    for eps in epsilons:
        g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=eps)
        error = np.linalg.norm(g_analytic - g_numeric)
        grader.require_is_true(error < 1e-3)

def test2e_gradient_direction():
    import numpy as np
    
    A = np.array([[1., 0.], [0., 1.]])
    b = np.array([0., 0.])
    w = np.array([1., -1.])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    
    expected = w
    grader.require_is_equal(g_analytic, expected)
    grader.require_is_equal(g_numeric, expected, tolerance=1e-4)

grader.add_basic_part('2e-consistency-basic', test2e_consistency_check, max_points=1, description='finite differences consistency check')
grader.add_basic_part('2e-points-basic', test2e_different_points, max_points=1, description='finite differences at different points')
grader.add_basic_part('2e-degenerate-basic', test2e_degenerate_case, max_points=1, description='finite differences degenerate case')
grader.add_basic_part('2e-epsilon-range-basic', test2e_epsilon_range, max_points=1, description='finite differences epsilon range')
grader.add_basic_part('2e-direction-basic', test2e_gradient_direction, max_points=1, description='finite differences gradient direction')

def test3c_convergence_rate():
    import numpy as np
    
    x = np.array([1., 2., 3.])
    w = np.array([1., 1., 1.])
    theta_star = np.sum(w * x) / np.sum(w)
    
    theta_large_lr = submission.gradient_descent_quadratic(x, w, 0.0, 0.4, 50)
    
    theta_small_lr = submission.gradient_descent_quadratic(x, w, 0.0, 0.05, 200)
    
    error_large = abs(theta_large_lr - theta_star)
    error_small = abs(theta_small_lr - theta_star)
    grader.require_is_true(error_small < 0.01)

def test3c_zero_steps():
    import numpy as np
    
    x = np.array([1., 2.])
    w = np.array([1., 1.])
    theta0 = 5.0
    
    theta = submission.gradient_descent_quadratic(x, w, theta0, 0.1, 0)
    grader.require_is_equal(theta, theta0)

def test3c_weighted_vs_unweighted():
    import numpy as np
    
    x = np.array([1., 3., 5.])
    w_uniform = np.array([1., 1., 1.])
    w_skewed = np.array([10., 1., 1.])
    
    theta_uniform = submission.gradient_descent_quadratic(x, w_uniform, 0.0, 0.1, 100)
    theta_skewed = submission.gradient_descent_quadratic(x, w_skewed, 0.0, 0.1, 100)
    
    grader.require_is_equal(theta_uniform, 3.0, tolerance=1e-3)
    grader.require_is_true(theta_skewed < theta_uniform)

grader.add_basic_part('3c-convergence-basic', test3c_convergence_rate, max_points=1, description='gradient descent convergence rate')
grader.add_basic_part('3c-zerosteps-basic', test3c_zero_steps, max_points=1, description='gradient descent zero steps')
grader.add_basic_part('3c-weighted-basic', test3c_weighted_vs_unweighted, max_points=1, description='gradient descent weighted vs unweighted')

def test_stress_large_dimensions():
    import numpy as np
    
    B, Din, Dout = 100, 500, 200
    x = np.random.randn(B, Din)
    W = np.random.randn(Din, Dout)
    b = np.random.randn(Dout)
    
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)

def test_stress_numerical_stability():
    import numpy as np
    
    x = np.array([[1e-15, 1e15]])
    W = np.array([[1e15], [1e-15]])
    b = np.array([0.0])
    
    out = submission.linear_project(x, W, b)
    grader.require_is_true(np.isfinite(out).all())

grader.add_basic_part('stress-large-basic', test_stress_large_dimensions, max_points=1, description='stress test large dimensions')
grader.add_basic_part('stress-numerical-basic', test_stress_numerical_stability, max_points=1, description='stress test numerical stability')

def test_properties_linear_project():
    import numpy as np
    
    x1 = np.random.randn(2, 3)
    x2 = np.random.randn(2, 3)
    W = np.random.randn(3, 4)
    b = np.random.randn(4)
    
    out_sum = submission.linear_project(x1 + x2, W, b)
    out1 = submission.linear_project(x1, W, b)
    out2 = submission.linear_project(x2, W, b)
    
    expected = out1 + out2 - b
    grader.require_is_equal(expected, out_sum)

def test_properties_gradient_descent():
    import numpy as np

    x = np.random.randn(5)
    w = np.random.random(5) + 0.1
    theta0 = np.random.randn()
    
    theta_final = submission.gradient_descent_quadratic(x, w, theta0, 0.0, 100)
    grader.require_is_equal(theta0, theta_final)

grader.add_basic_part('properties-linear-basic', test_properties_linear_project, max_points=1, description='linear_project linearity property')
grader.add_basic_part('properties-gd-basic', test_properties_gradient_descent, max_points=1, description='gradient descent zero lr property')

def test1e_batch_independence():
    import numpy as np
    
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    W = np.array([[1.0, 0.5], [2.0, 1.5]])
    b = np.array([0.1, 0.2])
    
    out_batch = submission.linear_project(x, W, b)
    
    for i in range(x.shape[0]):
        out_single = submission.linear_project(x[i:i+1], W, b)
        grader.require_is_equal(out_single, out_batch[i:i+1])

def test1e_zero_matrices():
    import numpy as np
    
    x = np.zeros((3, 4))
    W = np.zeros((4, 5))
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    out = submission.linear_project(x, W, b)
    expected = np.tile(b, (3, 1))
    grader.require_is_equal(expected, out)

def test1e_identity_transformation():
    import numpy as np
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    W = np.eye(3)
    b = np.zeros(3)
    
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(x, out)

def test1f_maximum_groups():
    import numpy as np
    from einops import rearrange
    
    x = np.arange(12, dtype=float).reshape(2, 6)
    num_groups = 6
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    expected = x.reshape(2, 6, 1)
    grader.require_is_equal(expected, out)

def test1f_prime_dimensions():
    import numpy as np
    from einops import rearrange
    
    x = np.random.randn(3, 21)
    num_groups = 7
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    expected = x.reshape(3, 7, 3)
    grader.require_is_equal(expected, out)

def test1g_orthogonal_vectors():
    import numpy as np
    
    A = np.array([[[1., 0., 0.], [0., 1., 0.]]])
    B = np.array([[[0., 0., 1.], [1., 0., 0.]]])
    
    out = submission.normalized_inner_products(A, B, normalize=False)
    expected = np.array([[[0., 1.], [0., 0.]]])
    grader.require_is_equal(expected, out)

def test1g_identical_matrices():
    import numpy as np
    
    A = np.array([[[2., 3., 1.], [1., 0., 2.]]])
    B = A.copy()
    
    out = submission.normalized_inner_products(A, B, normalize=False)
    expected_diag = np.array([np.sum(A[0, 0]**2), np.sum(A[0, 1]**2)])
    grader.require_is_equal(out[0, 0, 0], expected_diag[0])
    grader.require_is_equal(out[0, 1, 1], expected_diag[1])

def test1g_scaling_invariance():
    import numpy as np
    
    A = np.random.randn(1, 2, 4)
    B = np.random.randn(1, 3, 4)
    
    out1 = submission.normalized_inner_products(A, B, normalize=True)
    out2 = submission.normalized_inner_products(2*A, 3*B, normalize=True)
    
    grader.require_is_equal(out1 * 6, out2)

def test1h_single_element():
    import numpy as np
    
    scores = np.array([[[5.0]]], dtype=float)
    out = submission.mask_strictly_upper(scores.copy())
    grader.require_is_equal(out[0, 0, 0], 5.0)

def test1h_large_matrix():
    import numpy as np
    
    L = 10
    scores = np.random.randn(2, L, L)
    out = submission.mask_strictly_upper(scores.copy())
    
    for i in range(L):
        for j in range(L):
            if j > i:
                grader.require_is_true(np.isinf(out[0, i, j]))
                grader.require_is_true(np.isinf(out[1, i, j]))
            else:
                grader.require_is_equal(out[0, i, j], scores[0, i, j])
                grader.require_is_equal(out[1, i, j], scores[1, i, j])

def test1h_negative_infinity_input():
    import numpy as np
    
    scores = np.array([[[-np.inf, 1.0], [2.0, 3.0]]], dtype=float)
    out = submission.mask_strictly_upper(scores.copy())
    
    grader.require_is_true(np.isinf(out[0, 0, 0]) and out[0, 0, 0] < 0)
    grader.require_is_true(np.isinf(out[0, 0, 1]))
    grader.require_is_equal(out[0, 1, 0], 2.0)
    grader.require_is_equal(out[0, 1, 1], 3.0)

def test1i_sparse_probabilities():
    import numpy as np
    from einops import einsum
    
    P = np.array([[0.9, 0.1, 0.0], [0.0, 0.5, 0.5]])
    V = np.array([[[1., 2.], [3., 4.], [5., 6.]], 
                  [[7., 8.], [9., 10.], [11., 12.]]])
    
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    
    expected = np.array([[1.2, 2.2], [10., 11.]])
    grader.require_is_equal(expected, out)

def test1i_single_sequence_element():
    import numpy as np
    from einops import einsum
    
    P = np.array([[1.0], [1.0]])
    V = np.array([[[5., 6.]], [[7., 8.]]])
    
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    
    expected = np.array([[5., 6.], [7., 8.]])
    grader.require_is_equal(expected, out)

def test2b_identical_vectors():
    import numpy as np
    
    w = np.array([3.0, -1.0, 2.5])
    c = w.copy()
    expected = np.zeros_like(w)
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

def test2b_opposite_vectors():
    import numpy as np
    
    w = np.array([1.0, -2.0, 3.0])
    c = -w
    expected = 4.0 * w
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

def test2b_single_dimension():
    import numpy as np
    
    w = np.array([5.0])
    c = np.array([2.0])
    expected = np.array([6.0])
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)

def test2d_non_square_matrices():
    import numpy as np
    
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) 
    B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) 
    
    grad_A, grad_B = submission.matrix_grad(A, B)

    expected_grad_A = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])

    expected_grad_B = np.array([[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]])
    
    grader.require_is_equal(grad_A, expected_grad_A)
    grader.require_is_equal(grad_B, expected_grad_B)

def test2d_zero_matrices():
    import numpy as np
    
    A = np.zeros((2, 3))
    B = np.zeros((3, 2))
    
    grad_A, grad_B = submission.matrix_grad(A, B)
    
    grader.require_is_equal(grad_A, np.zeros((2, 3)))
    grader.require_is_equal(grad_B, np.zeros((3, 2)))

def test2d_symmetric_case():
    import numpy as np
    
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = A.copy()
    
    grad_A, grad_B = submission.matrix_grad(A, B)

    expected_grad_A = np.array([[3.0, 7.0], [3.0, 7.0]])

    expected_grad_B = np.array([[4.0, 4.0], [6.0, 6.0]])
    
    grader.require_is_equal(grad_A, expected_grad_A)
    grader.require_is_equal(grad_B, expected_grad_B)

def test2e_ill_conditioned_matrix():
    import numpy as np
    
    A = np.array([[1.0, 1.0], [1.0, 1.001]])
    b = np.array([1.0, 1.001])
    w = np.array([0.5, 0.5])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-4)
    
    error = np.linalg.norm(g_analytic - g_numeric)
    grader.require_is_true(error < 1e-2)

def test2e_overdetermined_system():
    import numpy as np
    
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    b = np.array([1.0, 2.0, 3.0, -1.0])
    w = np.array([0.0, 0.0])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    
    error = np.linalg.norm(g_analytic - g_numeric)
    grader.require_is_true(error < 1e-4)

def test2e_underdetermined_system():
    import numpy as np
    
    A = np.array([[1.0, 2.0, 3.0]])
    b = np.array([6.0])
    w = np.array([1.0, 1.0, 1.0])
    
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    
    error = np.linalg.norm(g_analytic - g_numeric)
    grader.require_is_true(error < 1e-4)

def test3c_oscillatory_behavior():
    import numpy as np
    
    x = np.array([0.0, 10.0])
    w = np.array([1.0, 1.0])
    theta0 = 0.0
    
    theta_stable = submission.gradient_descent_quadratic(x, w, theta0, 0.1, 100)
    theta_unstable = submission.gradient_descent_quadratic(x, w, theta0, 0.8, 20)
    
    expected = 5.0
    grader.require_is_equal(theta_stable, expected, tolerance=1e-2)
    grader.require_is_true(abs(theta_unstable - expected) > abs(theta_stable - expected))

def test3c_negative_weights():
    import numpy as np
    
    x = np.array([1.0, 3.0, 5.0])
    w = np.array([-1.0, 2.0, 1.0])  # sum = 2, not zero
    theta0 = 0.0
    
    theta = submission.gradient_descent_quadratic(x, w, theta0, 0.1, 100)
    expected = np.sum(w * x) / np.sum(w)  # (-1 + 6 + 5) / 2 = 5.0
    grader.require_is_equal(theta, expected, tolerance=1e-3)

def test3c_large_learning_rate():
    import numpy as np
    
    x = np.array([2.0])
    w = np.array([1.0])
    theta0 = 10.0
    
    theta = submission.gradient_descent_quadratic(x, w, theta0, 0.9, 50)
    grader.require_is_true(abs(theta) < 100)

def test_boundary_empty_arrays():
    import numpy as np
    
    try:
        x = np.array([]).reshape(0, 3)
        W = np.random.randn(3, 2)
        b = np.random.randn(2)
        out = submission.linear_project(x, W, b)
        grader.require_is_equal(out.shape, (0, 2))
    except:
        pass

def test_boundary_very_small_values():
    import numpy as np
    
    x = np.array([[1e-100, 2e-100]])
    W = np.array([[1e100], [1e100]])
    b = np.array([0.0])
    
    out = submission.linear_project(x, W, b)
    expected = 3.0
    grader.require_is_equal(out[0, 0], expected, tolerance=1e-10)

def test_boundary_alternating_signs():
    import numpy as np
    
    n = 10
    x = np.array([(-1)**i for i in range(n)])
    w = np.ones(n)
    theta0 = 0.0
    
    theta = submission.gradient_descent_quadratic(x, w, theta0, 0.1, 100)
    expected = 0.0 if n % 2 == 0 else -1.0/n
    grader.require_is_equal(theta, expected, tolerance=1e-3)

def test_advanced_einsum_broadcasting():
    import numpy as np
    from einops import einsum
    
    P = np.array([[0.2, 0.3, 0.5]])
    V = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    
    expected = 0.2 * V[0, 0] + 0.3 * V[0, 1] + 0.5 * V[0, 2]
    grader.require_is_equal(expected, out[0])

grader.add_basic_part('1e-batch-independence', test1e_batch_independence, max_points=1, description='linear_project batch independence')
grader.add_basic_part('1e-zero-matrices', test1e_zero_matrices, max_points=1, description='linear_project zero matrices')
grader.add_basic_part('1e-identity-transform', test1e_identity_transformation, max_points=1, description='linear_project identity transformation')
grader.add_basic_part('1f-max-groups', test1f_maximum_groups, max_points=1, description='split_last_dim maximum groups')
grader.add_basic_part('1f-prime-dims', test1f_prime_dimensions, max_points=1, description='split_last_dim prime dimensions')
grader.add_basic_part('1g-orthogonal', test1g_orthogonal_vectors, max_points=1, description='normalized_inner_products orthogonal vectors')
grader.add_basic_part('1g-identical', test1g_identical_matrices, max_points=1, description='normalized_inner_products identical matrices')
grader.add_basic_part('1g-scaling', test1g_scaling_invariance, max_points=1, description='normalized_inner_products scaling invariance')
grader.add_basic_part('1h-single-element', test1h_single_element, max_points=1, description='mask_strictly_upper single element')
grader.add_basic_part('1h-large-matrix', test1h_large_matrix, max_points=1, description='mask_strictly_upper large matrix')
grader.add_basic_part('1h-neg-inf-input', test1h_negative_infinity_input, max_points=1, description='mask_strictly_upper negative infinity input')
grader.add_basic_part('1i-sparse-probs', test1i_sparse_probabilities, max_points=1, description='prob_weighted_sum sparse probabilities')
grader.add_basic_part('1i-single-seq', test1i_single_sequence_element, max_points=1, description='prob_weighted_sum single sequence element')
grader.add_basic_part('2b-identical-vectors', test2b_identical_vectors, max_points=1, description='gradient_warmup identical vectors')
grader.add_basic_part('2b-opposite-vectors', test2b_opposite_vectors, max_points=1, description='gradient_warmup opposite vectors')
grader.add_basic_part('2b-single-dim', test2b_single_dimension, max_points=1, description='gradient_warmup single dimension')
grader.add_basic_part('2d-non-square', test2d_non_square_matrices, max_points=1, description='matrix_grad non-square matrices')
grader.add_basic_part('2d-zero-matrices', test2d_zero_matrices, max_points=1, description='matrix_grad zero matrices')
grader.add_basic_part('2d-symmetric', test2d_symmetric_case, max_points=1, description='matrix_grad symmetric case')
grader.add_basic_part('2e-ill-conditioned', test2e_ill_conditioned_matrix, max_points=1, description='finite differences ill conditioned matrix')
grader.add_basic_part('2e-overdetermined', test2e_overdetermined_system, max_points=1, description='finite differences overdetermined system')
grader.add_basic_part('2e-underdetermined', test2e_underdetermined_system, max_points=1, description='finite differences underdetermined system')
grader.add_basic_part('3c-oscillatory', test3c_oscillatory_behavior, max_points=1, description='gradient descent oscillatory behavior')
grader.add_basic_part('3c-negative-weights', test3c_negative_weights, max_points=1, description='gradient descent negative weights')
grader.add_basic_part('3c-large-lr', test3c_large_learning_rate, max_points=1, description='gradient descent large learning rate')
grader.add_basic_part('boundary-empty', test_boundary_empty_arrays, max_points=1, description='boundary case empty arrays')
grader.add_basic_part('boundary-small-vals', test_boundary_very_small_values, max_points=1, description='boundary case very small values')
grader.add_basic_part('boundary-alternating', test_boundary_alternating_signs, max_points=1, description='boundary case alternating signs')
grader.add_basic_part('advanced-einsum', test_advanced_einsum_broadcasting, max_points=1, description='advanced einsum broadcasting')

############################################################
grader.grade()
