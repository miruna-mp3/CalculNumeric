import numpy as np

eps = 10**(-7)
n = 5

B = np.random.randint(low=2, size=(n, n))
A = (B @ B.T).astype(float) + np.eye(n) * eps

# 1. Jacobi

U = np.eye(n)
copy_A = np.copy(A)

for k in range(10000):
    max_val = -1
    p = q = 0
    for i in range(n):
        for j in range(i):
            if abs(copy_A[i][j]) > max_val:
                p, q = i, j
                max_val = abs(copy_A[i][j])

    if max_val <= eps:
        print("Diagonal matrix obtained")
        break

    alpha = (copy_A[p][p] - copy_A[q][q]) / (2 * copy_A[p][q])

    if alpha >= 0:
        sign_a = 1
    else:
        sign_a = -1

    t = -alpha + sign_a * np.sqrt(alpha**2 + 1)
    c = 1 / np.sqrt(1 + t**2)
    s = t / np.sqrt(1 + t**2)

    old_p = np.copy(copy_A[p, :])
    old_q = np.copy(copy_A[q, :])

    for j in range(n):
        if j != p and j != q:
            copy_A[p][j] = copy_A[j][p] = c * old_p[j] + s * old_q[j]
            copy_A[q][j] = copy_A[j][q] = -s * old_p[j] + c * old_q[j]

    copy_A[p][p] = old_p[p] + t * old_p[q]
    copy_A[q][q] = old_q[q] - t * old_p[q]
    copy_A[p][q] = copy_A[q][p] = 0

    for i in range(n):
        old_uip = U[i][p]
        U[i][p] = c * U[i][p] + s * U[i][q]
        U[i][q] = -s * old_uip + c * U[i][q]

llambda = np.diag(copy_A)
Lambda = np.diag(llambda)
print(f"||A_init*U - U*Lambda|| = {np.linalg.norm(A @ U - U @ Lambda)}")
print(f"Valori proprii (Jacobi) : {llambda}")
print(f"Valori proprii (numpy)  : {np.linalg.eigvalsh(A)}")

# 2. Cholesky iteration

A_k = np.copy(A).astype(float)

for k in range(10000):
    A_old = np.copy(A_k)
    L = np.linalg.cholesky(A_k)
    A_k = L.T @ L
    if np.linalg.norm(A_k - A_old) < eps:
        print(f"Converged at iteration no. k={k}")
        break

print(f"A={A}")
print(f"A_k={A_k}")

# 3. SVD

p = 5
n = 3
A = np.random.randint(low=5, size=(p, n)).astype(float)

U_svd, sigma, Vt = np.linalg.svd(A, full_matrices=True)
V = Vt.T

print(f"A={A}")
print(f"Valorile singulare ale lui A: {sigma}")

rank_calc = np.sum(sigma > eps)
print(f"rang(A)={rank_calc} (calculat)")
print(f"rang(A)={np.linalg.matrix_rank(A)} (numpy)")

sigma_pos = sigma[sigma > eps]
k2 = sigma_pos[0] / sigma_pos[-1]
print(f"Condition number (k2) : {k2} (calculat)")
print(f"Condition number (k2) : {np.linalg.cond(A)} (numpy)")

S_inv = np.zeros((n, p))
for i in range(len(sigma)):
    if sigma[i] > eps:
        S_inv[i][i] = 1.0 / sigma[i]

A_I = V @ S_inv @ U_svd.T
A_J = np.linalg.inv(A.T @ A) @ A.T

print(f"||AI - AJ||_1 : {np.linalg.norm(A_I - A_J, ord=1)}")
