import numpy as np
from numpy.linalg import norm, inv


def compute_b(A, s):
    n = A.shape[0]
    b = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i][j]
    return b


def backward_substitution(R, b, eps):
    n = R.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < eps:
            raise ValueError("singular matrix")
        s = b[i]
        for j in range(i + 1, n):
            s -= R[i][j] * x[j]
        x[i] = s / R[i][i]
    return x


def householder_qr(A_mat, b_vec, n, eps):
    Q_mat = np.eye(n, dtype=float)

    for r in range(n - 1):
        sigma = 0.0
        for i in range(r, n):
            sigma += A_mat[i][r] ** 2

        if sigma < eps:
            continue

        k = np.sqrt(sigma)
        if A_mat[r][r] > 0:
            k = -k

        beta = sigma - k * A_mat[r][r]

        u_vec = np.zeros(n, dtype=float)
        u_vec[r] = A_mat[r][r] - k
        u_vec[r+1:] = A_mat[r+1:n, r]

        for j in range(r + 1, n):
            gamma = 0.0
            for i in range(r, n):
                gamma += u_vec[i] * A_mat[i][j]
            gamma /= beta
            for i in range(r, n):
                A_mat[i][j] -= gamma * u_vec[i]

        A_mat[r][r] = k
        for i in range(r + 1, n):
            A_mat[i][r] = 0.0

        gamma = 0.0
        for i in range(r, n):
            gamma += u_vec[i] * b_vec[i]
        gamma /= beta
        for i in range(r, n):
            b_vec[i] -= gamma * u_vec[i]

        for j in range(n):
            gamma = 0.0
            for i in range(r, n):
                gamma += u_vec[i] * Q_mat[i][j]
            gamma /= beta
            for i in range(r, n):
                Q_mat[i][j] -= gamma * u_vec[i]

    return Q_mat


def compute_inverse_householder(R, Q_T, n, eps):
    for i in range(n):
        if abs(R[i][i]) < eps:
            raise ValueError("singular matrix - inverse does not exist")

    A_inv = np.zeros((n, n), dtype=float)
    for j in range(n):
        rhs = Q_T[:, j].copy()
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            suma = 0
            for k in range(i + 1, n):
                suma += R[i][k] * x[k]
            x[i] = (rhs[i] - suma) / R[i][i]
        A_inv[:, j] = x
    return A_inv


def run_test(A_input, s_input, label, eps=1e-7):
    n = A_input.shape[0]
    print()
    print(label, " n =", n)
    print()

    A_orig = A_input.astype(float).copy()
    s_vec = s_input.astype(float).copy()

    b_vec = compute_b(A_orig, s_vec)
    b_orig = b_vec.copy()
    print("b =", b_orig)

    Q_bib, R_bib = np.linalg.qr(A_orig)
    print()
    print("Library QR")
    print("Q_bib =")
    print(Q_bib)
    print("R_bib =")
    print(R_bib)

    Qt_b_lib = Q_bib.T @ b_orig
    x_bib = backward_substitution(R_bib, Qt_b_lib.copy(), eps)
    print("x_QR =", x_bib)

    A_mat = A_orig.copy()
    b_house = b_orig.copy()
    Q_T = householder_qr(A_mat, b_house, n, eps)

    print()
    print("Householder QR")
    print("R =")
    print(A_mat)
    print("Q^T =")
    print(Q_T)

    x_house = backward_substitution(A_mat, b_house.copy(), eps)
    print("x_Householder =", x_house)

    diff_norm = norm(x_bib - x_house)
    print()
    print("norm(x_QR - x_Householder) =", diff_norm)

    res_house = norm(A_orig @ x_house - b_orig)
    res_bib = norm(A_orig @ x_bib - b_orig)
    rel_house = norm(x_house - s_vec) / norm(s_vec)
    rel_bib = norm(x_bib - s_vec) / norm(s_vec)

    print()
    print("norm(A*x_Householder - b) =", res_house)
    print("norm(A*x_QR - b)          =", res_bib)
    print("norm(x_Householder - s)/norm(s) =", rel_house)
    print("norm(x_QR - s)/norm(s)          =", rel_bib)

    A_inv_house = compute_inverse_householder(A_mat, Q_T, n, eps)
    print()
    print("A inverse Householder =")
    print(A_inv_house)

    A_inv_lib = inv(A_orig)
    inv_diff = norm(A_inv_house - A_inv_lib)
    print()
    print("Verificare (A * A_inv):")
    print(np.round(A_orig @ A_inv_house, 8))
    print()
    print("norm(A_inv_Householder - A_inv_library) =", inv_diff)


np.random.seed(42)

for n_rand in [5, 10, 50]:
    A_rand = np.random.randint(-10, 11, size=(n_rand, n_rand)).astype(float)
    A_rand += n_rand * np.eye(n_rand)
    s_rand = np.random.randint(-10, 11, size=n_rand).astype(float)
    run_test(A_rand, s_rand, "Random test n=" + str(n_rand))
