import numpy as np
from scipy.linalg import lu

n = 5
eps = 1e-7

np.random.seed(42)
M = np.random.randint(-10, 11, size=(n, n)).astype(float)
A_mat = M @ M.T
b_vec = np.random.randint(1, 11, size=n).astype(float)

A_orig = A_mat.copy()

P, L_lu, U_lu = lu(A_mat)
x_ref = np.linalg.solve(A_mat, b_vec)

print("LU Decomposition")
print("P =\n", P)
print("L =\n", L_lu)
print("U =\n", U_lu)
print("\nReference solution = ", x_ref)

d_vec = np.zeros(n)

for p in range(n):
    s = 0.0
    for k in range(p):
        s += d_vec[k] * A_mat[p, k] ** 2
    d_vec[p] = A_mat[p, p] - s

    if abs(d_vec[p]) <= eps:
        raise ValueError("d[" + str(p) + "] is near zero - can't do decomposition'")

    for i in range(p + 1, n):
        s = 0.0
        for k in range(p):
            s += d_vec[k] * A_mat[i, k] * A_mat[p, k]
        A_mat[i, p] = (A_mat[i, p] - s) / d_vec[p]

print("\nLDLT Decomposition")
print("d =", d_vec)
print("A after decomposition (L below diag, original A on/above diag):")
print(A_mat)

det_A = np.prod(d_vec)
print("\ndet(A) =", det_A)
print("det(A) via numpy =", np.linalg.det(A_orig))

z_vec = b_vec.copy()
for i in range(n):
    for j in range(i):
        z_vec[i] -= A_mat[i, j] * z_vec[j]

y_vec = np.zeros(n)
for i in range(n):
    if abs(d_vec[i]) <= eps:
        raise ValueError("d[" + str(i) + "] is near zero - can't solve  system")
    y_vec[i] = z_vec[i] / d_vec[i]

x_sol = y_vec.copy()
for i in range(n - 1, -1, -1):
    for j in range(i + 1, n):
        x_sol[i] -= A_mat[j, i] * x_sol[j]

print("\nx_Chol =", x_sol)

def mat_vec_original(A_mod, x, n):
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            a_ij = A_mod[i, j] if j >= i else A_mod[j, i]
            result[i] += a_ij * x[j]
    return result

Ax = mat_vec_original(A_mat, x_sol, n)
norm1 = np.linalg.norm(Ax - b_vec)
norm2 = np.linalg.norm(x_sol - x_ref)

print("\nVerificare:")
print("||A_init * x_Chol - b||_2 =", format(norm1, ".2e"))
print("||x_Chol - x_lib||_2     =", format(norm2, ".2e"))
