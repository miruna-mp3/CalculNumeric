import os

def read_vector(filename):
    """Read a list of floats from a file, one per line."""
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def solve(d0, d1, d2, b, eps):
    n = len(d0)
    p = n - len(d1)   # offset of first secondary diagonal
    q = n - len(d2)   # offset of second secondary diagonal

    # --- Task 1: print system size ---
    print(f"System size n = {n}")

    # --- Task 2: print diagonal offsets ---
    print(f"Secondary diagonal offsets: p = {p}, q = {q}")

    # --- Task 3: check all diagonal elements are nonzero ---
    for i in range(n):
        if abs(d0[i]) <= eps:
            print(f"ERROR: d0[{i}] = {d0[i]} is zero. Cannot use Gauss-Seidel.")
            return None

    # --- Task 4: Gauss-Seidel iteration ---
    kmax = 10000
    x = [0.0] * n

    converged = False
    for k in range(kmax):
        x_old = x.copy()

        for i in range(n):
            s = b[i]

            # subtract d1 left neighbor  (column i-p, already updated)
            if i - p >= 0:
                s -= d1[i - p] * x[i - p]

            # subtract d1 right neighbor (column i+p, not yet updated)
            if i + p < n:
                s -= d1[i] * x[i + p]

            # subtract d2 left neighbor  (column i-q, already updated)
            if i - q >= 0:
                s -= d2[i - q] * x[i - q]

            # subtract d2 right neighbor (column i+q, not yet updated)
            if i + q < n:
                s -= d2[i] * x[i + q]

            x[i] = s / d0[i]

        # check convergence: max absolute change
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
    """Compute y = A*x using only the diagonal vectors."""
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

# ----------------------------------------------------------------
# Main: run all 5 systems
# ----------------------------------------------------------------

# precision: change p_exp to 5,6,7,8, or 9
p_exp = 8
eps = 10 ** (-p_exp)
print(f"Precision eps = 1e-{p_exp}\n")

for i in range(1, 6):
    print(f"{'='*50}")
    print(f"System {i}")
    print(f"{'='*50}")

    # build filenames — adjust path if your files are in a subfolder
    f_d0 = f"d0_{i}.txt"
    f_d1 = f"d1_{i}.txt"
    f_d2 = f"d2_{i}.txt"
    f_b  = f"b_{i}.txt"

    # check files exist
    missing = [f for f in [f_d0, f_d1, f_d2, f_b] if not os.path.exists(f)]
    if missing:
        print(f"  Skipping — missing files: {missing}\n")
        continue

    d0 = read_vector(f_d0)
    d1 = read_vector(f_d1)
    d2 = read_vector(f_d2)
    b  = read_vector(f_b)

    x = solve(d0, d1, d2, b, eps)

    if x is not None:
        # Task 5 & 6: compute Ax and the error norm
        y = compute_ax(d0, d1, d2, x)
        error = max(abs(y[i] - b[i]) for i in range(len(b)))
        print(f"||Ax - b||_inf = {error:.2e}")

        # print first and last few solution values
        print(f"x[0..4]     = {[round(v,6) for v in x[:5]]}")
        print(f"x[n-5..n-1] = {[round(v,6) for v in x[-5:]]}")

    print()