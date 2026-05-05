import math

def horner(a, v):
    n = len(a)
    b = [0.0] * n
    b[0] = a[0]
    for i in range(1, n):
        b[i] = a[i] + b[i-1] * v
    return b

def eval_poly(a, v):
    b = horner(a, v)
    Pv = b[-1]

    c = horner(b[:-1], v)
    dPv = c[-1]

    d = horner(c[:-1], v)
    ddPv = d[-1]

    return Pv, dPv, ddPv

def find_interval(a):
    a0 = abs(a[0])
    A = max(abs(x) for x in a[1:])
    R = (a0 + A) / a0
    return R

def newton(a, x0, eps, kmax=1000):
    x = x0
    steps = 0
    for k in range(kmax):
        Pv, dPv, _ = eval_poly(a, x)

        if abs(dPv) <= eps:
            return None, steps  # derivative too small, bad starting point

        dx = Pv / dPv
        x = x - dx
        steps += 1

        if abs(dx) < eps:
            return x, steps
        if abs(dx) > 1e8:
            return None, steps  # diverging

    return None, steps

def olver(a, x0, eps, kmax=1000):
    x = x0
    steps = 0
    for k in range(kmax):
        Pv, dPv, ddPv = eval_poly(a, x)

        if abs(dPv) <= eps:
            return None, steps

        ck = (Pv ** 2 * ddPv) / (dPv ** 3)
        dx = Pv / (dPv - 0.5 * ck * Pv)
        x = x - dx
        steps += 1

        if abs(dx) < eps:
            return x, steps
        if abs(dx) > 1e8:
            return None, steps  # diverging

    return None, steps

def is_new_root(roots, r, eps):
    for existing in roots:
        if abs(existing - r) <= eps:
            return False
    return True

def main():
    print("=== Tema 7 - Metoda Newton si Olver ===\n")

    # --- Input ---
    n = int(input("Gradul polinomului: "))
    print(f"Introduceti cei {n+1} coeficienti (a0, a1, ..., an):")
    a = []
    for i in range(n + 1):
        c = float(input(f"  a[{i}] = "))
        a.append(c)

    eps = float(input("Precizia epsilon: "))

    # --- Interval ---
    R = find_interval(a)
    print(f"\nToate radacinile reale se afla in [-{R:.4f}, {R:.4f}]")

    # --- Try multiple starting points ---
    # We spread starting points evenly across [-R, R]
    num_starts = 20
    step = 2 * R / num_starts
    starting_points = [-R + i * step for i in range(num_starts + 1)]

    roots_newton = []
    roots_olver = []

    print("\n--- Metoda Newton ---")
    total_steps_newton = 0
    for x0 in starting_points:
        root, steps = newton(a, x0, eps)
        if root is not None and -R <= root <= R:
            Pv, _, _ = eval_poly(a, root)
            if abs(Pv) < 1e-6:  # sanity check its actually a root
                if is_new_root(roots_newton, root, eps):
                    roots_newton.append(root)
                    total_steps_newton += steps
                    print(f"  x0={x0:.3f} -> radacina={root:.8f}  (pasi: {steps})")

    print("\n--- Metoda Olver ---")
    total_steps_olver = 0
    for x0 in starting_points:
        root, steps = olver(a, x0, eps)
        if root is not None and -R <= root <= R:
            Pv, _, _ = eval_poly(a, root)
            if abs(Pv) < 1e-6:
                if is_new_root(roots_olver, root, eps):
                    roots_olver.append(root)
                    total_steps_olver += steps
                    print(f"  x0={x0:.3f} -> radacina={root:.8f}  (pasi: {steps})")

    # --- Comparison ---
    print("\n--- Comparatie ---")
    print(f"Newton: {len(roots_newton)} radacini gasite, total pasi: {total_steps_newton}")
    print(f"Olver:  {len(roots_olver)} radacini gasite, total pasi: {total_steps_olver}")

    # --- Save to file ---
    # Use Olver roots if available, otherwise Newton
    final_roots = roots_olver if roots_olver else roots_newton

    with open("rezultate.txt", "w") as f:
        f.write("=== Radacini distincte ale polinomului ===\n\n")
        f.write("Coeficienti: " + str(a) + "\n")
        f.write(f"Epsilon: {eps}\n")
        f.write(f"Interval: [-{R:.4f}, {R:.4f}]\n\n")

        f.write("Radacini (Newton):\n")
        for r in sorted(roots_newton):
            f.write(f"  {r:.8f}\n")

        f.write("\nRadacini (Olver):\n")
        for r in sorted(roots_olver):
            f.write(f"  {r:.8f}\n")

        f.write(f"\nNewton - total pasi: {total_steps_newton}\n")
        f.write(f"Olver  - total pasi: {total_steps_olver}\n")

    print("\nRezultatele au fost salvate in 'rezultate.txt'")

main()