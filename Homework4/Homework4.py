import os

def read_vector(file_name):
    result = []
    f = open(file_name, 'r')
    for line in f:
        line = line.strip()
        if line != "":
            result.append(float(line))
    f.close()
    return result

def solve(d0, d1, d2, b, eps):
    n = len(d0)
    p = n - len(d1)
    q = n - len(d2)

    # Task 1
    print(f"System size n = {n}")

    # Task 2
    print(f"Secondary diagonal offsets: p = {p}, q = {q}")

    # Task 3
    for i in range(n):
        if abs(d0[i]) <= eps:
            print(f"ERROR: d0[{i}] = {d0[i]} is zero. Cannot use Gauss-Seidel.")
            return None

    # Task 4
    kmax = 10000
    x = [0.0] * n

    converged = False
    for k in range(kmax):
        x_old = x.copy()

        for i in range(n):
            s = b[i]

            if i - p >= 0:
                s -= d1[i - p] * x[i - p]

            if i + p < n:
                s -= d1[i] * x[i + p]

            if i - q >= 0:
                s -= d2[i - q] * x[i - q]

            if i + q < n:
                s -= d2[i] * x[i + q]

            x[i] = s / d0[i]

        delta = max(abs(x[i] - x_old[i]) for i in range(n))

        if delta < eps:
            print(f"Converged after {k+1} iterations (delta = {delta:.2e})")
            converged = True
            break

        if delta > 1e10:
            print("DIVERGED.")
            return None

    if not converged:
        print(f"Did not converge after {kmax} iterations.")
        return None

    return x

def compute_ax(d0, d1, d2, x):
    n = len(d0)
    p = n - len(d1)
    q = n - len(d2)

    y = [0.0] * n

    for i in range(n):
        s = d0[i] * x[i]

        if i - p >= 0:
            s += d1[i - p] * x[i - p]

        if i + p < n:
            s += d1[i] * x[i + p]

        if i - q >= 0:
            s += d2[i - q] * x[i - q]

        if i + q < n:
            s += d2[i] * x[i + q]

        y[i] = s

    return y

exp = 8
eps = 10 ** (-exp)
print(f"Precision eps = 1e-{exp}\n")

for i in range(1, 6):
    print(f"{'='*50}")
    print(f"System {i}")
    print(f"{'='*50}")

    f_d0 = f"d0_{i}.txt"
    f_d1 = f"d1_{i}.txt"
    f_d2 = f"d2_{i}.txt"
    f_b  = f"b_{i}.txt"

    d0 = read_vector(f_d0)
    d1 = read_vector(f_d1)
    d2 = read_vector(f_d2)
    b  = read_vector(f_b)

    x = solve(d0, d1, d2, b, eps)

    if x is not None:
        y = compute_ax(d0, d1, d2, x)
        error = max(abs(y[i] - b[i]) for i in range(len(b)))
        print(f"||Ax - b||_inf = {error:.2e}")

        print(f"x[0..4]     = {[round(v,6) for v in x[:5]]}")
        print(f"x[n-5..n-1] = {[round(v,6) for v in x[-5:]]}")

    print()