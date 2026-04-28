import numpy as np


def l(w):
    return -np.log(1.0 - 1.0/(1.0 + np.exp(w[1]-w[0]))) - np.log(1.0/(1.0 + np.exp(-w[0]-w[1])))

def grad_l(w):
    a = 1.0/(1.0 + np.exp(w[1]-w[0]))
    b = 1.0/(1.0 + np.exp(-w[0]-w[1]))
    return np.array([a + b - 1.0, b - a - 1.0])


def f1(x):       return x[0]*x[0] + x[1]*x[1] - 2.0*x[0] - 4.0*x[1] - 1.0
def grad_f1(x):  return np.array([2.0*x[0] - 2.0, 2.0*x[1] - 4.0])

def f2(x):       return 3.0*x[0]*x[0] - 12.0*x[0] + 2.0*x[1]*x[1] + 16.0*x[1] - 10.0
def grad_f2(x):  return np.array([6.0*x[0] - 12.0, 4.0*x[1] + 16.0])

def f3(x):       return x[0]*x[0] - 4.0*x[0]*x[1] + 4.5*x[1]*x[1] - 4.0*x[1] + 3.0
def grad_f3(x):  return np.array([2.0*x[0] - 4.0*x[1], -4.0*x[0] + 9.0*x[1] - 4.0])

def f4(x):       return x[0]*x[0]*x[1] - 2.0*x[0]*x[1]*x[1] + 3.0*x[0]*x[1] + 4.0
def grad_f4(x):  return np.array([2.0*x[0]*x[1] - 2.0*x[1]*x[1] + 3.0*x[1],
                                  x[0]*x[0] - 4.0*x[0]*x[1] + 3.0*x[0]])


# parallel tables, indexed the same way
funcs   = [l,        f1,        f2,        f3,         f4       ]
grads   = [grad_l,   grad_f1,   grad_f2,   grad_f3,    grad_f4  ]
starts  = [(0.0,1.0),(0.5,1.5),(1.5,-3.5),(8.15,3.85),(-0.5,0.0)]
labels  = ["l", "f1", "f2", "f3", "f4"]


def G(i, h, x, f):
    # one component of the gradient via 4-point central difference
    y = x.copy()
    y[i] = x[i] + 2.0*h; F1 = f(y)
    y[i] = x[i] + h;     F2 = f(y)
    y[i] = x[i] - h;     F3 = f(y)
    y[i] = x[i] - 2.0*h; F4 = f(y)
    return (-F1 + 8.0*F2 - 8.0*F3 + F4) / (12.0*h)


def gradient_descent_f(idx, lr_mod, eps):
    # uses the analytic gradient
    f    = funcs[idx]
    grad = grads[idx]
    x    = np.array(starts[idx], dtype=float)

    eta  = 10**(-4) if lr_mod == 1 else 1.0
    beta = 0.8
    k    = 0

    for k in range(40000):
        g = grad(x)
        norm_g = np.linalg.norm(g)
        if eta*norm_g < eps or eta*norm_g > 10**10:
            break

        if lr_mod == 2:
            # backtracking, capped at 8 reductions per step
            eta = 1.0
            p = 1
            while f(x - eta*g) > f(x) - eta/2 * (norm_g**2) and p < 8:
                eta *= beta
                p += 1

        x = x - eta*g

    if eta * np.linalg.norm(grad(x)) <= eps:
        return x, k + 1
    return None, k + 1


def gradient_descent_g(idx, lr_mod, eps):
    # uses the approximate gradient
    f = funcs[idx]
    x = np.array(starts[idx], dtype=float)
    h = 10**(-6)

    eta  = 10**(-4) if lr_mod == 1 else 1.0
    beta = 0.8
    k    = 0
    g    = None

    for k in range(40000):
        g = np.array([G(0, h, x, f), G(1, h, x, f)])
        norm_g = np.linalg.norm(g)
        if eta*norm_g < eps or eta*norm_g > 10**10:
            break

        if lr_mod == 2:
            eta = 1.0
            p = 1
            while f(x - eta*g) > f(x) - eta/2 * (norm_g**2) and p < 8:
                eta *= beta
                p += 1

        x = x - eta*g

    if eta * np.linalg.norm(g) <= eps:
        return x, k + 1
    return None, k + 1


def report(method, eps):
    # runs both lr modes across all 5 cases and prints rows
    for lr_mod, tag in [(1, "fixed eta"), (2, "backtracking")]:
        print(f"[{tag}]")
        for idx in range(5):
            sol, steps = method(idx, lr_mod, eps)
            if sol is None:
                print(f"  {labels[idx]}: divergence - steps = {steps}")
            else:
                print(f"  {labels[idx]}: x = ({sol[0]:.10f}, {sol[1]:.10f})  |  steps = {steps}")
        print()


if __name__ == "__main__":
    eps = 10**(-4)

    print("[analytic gradient]\n")
    report(gradient_descent_f, eps)

    print("[approximate gradient]\n")
    report(gradient_descent_g, eps)