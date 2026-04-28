import numpy as np
import matplotlib.pyplot as plt

# the function and derivative
def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

def df(x):
    return 4*x**3 - 36*x**2 + 60*x

# input values
a = 0.0
b = 2.0
x_bar = 1.5
n = 5
m = 3
da = df(a)
db = df(b)

# generate points
np.random.seed(42)
interior = np.sort(np.random.uniform(a, b, n - 1))
x = np.concatenate(([a], interior, [b]))
y = f(x)

# LEAST SQUARES 

# build B and rhs
B = np.zeros((m + 1, m + 1))
rhs = np.zeros(m + 1)
for i in range(m + 1):
    rhs[i] = np.sum(y * x**i)
    for j in range(m + 1):
        B[i, j] = np.sum(x**(i + j))

# solve for coefficients
a_coeffs = np.linalg.solve(B, rhs)

# horner evaluation
def horner(coeffs, x_val):
    c = coeffs[::-1]  # highest degree first
    d = c[0]
    for i in range(1, len(c)):
        d = d * x_val + c[i]
    return d

Pm_x_bar = horner(a_coeffs, x_bar)

print("=== Least Squares ===")
print(f"Pm(x_bar) = {Pm_x_bar:.6f}")
print(f"f(x_bar)  = {f(x_bar):.6f}")
print(f"|Pm(x_bar) - f(x_bar)| = {abs(Pm_x_bar - f(x_bar)):.6f}")
print(f"sum |Pm(xi) - yi| = {sum(abs(horner(a_coeffs, x[i]) - y[i]) for i in range(n+1)):.6f}")

# CUBIC SPLINE

h = np.diff(x)

# build H and rhs
H = np.zeros((n + 1, n + 1))
rhs2 = np.zeros(n + 1)

# first row
H[0, 0] = 2 * h[0]
H[0, 1] = h[0]
rhs2[0] = 6 * ((y[1] - y[0]) / h[0] - da)

# interior rows
for i in range(1, n):
    H[i, i-1] = h[i-1]
    H[i, i]   = 2 * (h[i-1] + h[i])
    H[i, i+1] = h[i]
    rhs2[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

# last row
H[n, n-1] = h[n-1]
H[n, n]   = 2 * h[n-1]
rhs2[n] = 6 * (db - (y[n] - y[n-1]) / h[n-1])

# solve for A
A = np.linalg.solve(H, rhs2)

# compute b and c for each subinterval
b_sp = np.zeros(n)
c_sp = np.zeros(n)
for i in range(n):
    b_sp[i] = (y[i+1] - y[i]) / h[i] - h[i] * (A[i+1] - A[i]) / 6
    c_sp[i] = (x[i+1]*y[i] - x[i]*y[i+1]) / h[i] - h[i] * (x[i+1]*A[i] - x[i]*A[i+1]) / 6

# find which interval x_bar is in
i0 = n - 1
for i in range(n):
    if x[i] <= x_bar <= x[i+1]:
        i0 = i
        break

# evaluate spline at x_bar
Sf_x_bar = ((x_bar - x[i0])**3 * A[i0+1] / (6*h[i0])
          + (x[i0+1] - x_bar)**3 * A[i0] / (6*h[i0])
          + b_sp[i0] * x_bar + c_sp[i0])

print("\n=== Cubic Spline ===")
print(f"Sf(x_bar) = {Sf_x_bar:.6f}")
print(f"f(x_bar)  = {f(x_bar):.6f}")
print(f"|Sf(x_bar) - f(x_bar)| = {abs(Sf_x_bar - f(x_bar)):.6f}")

# PLOT

def eval_spline(x_val):
    i0 = n - 1
    for i in range(n):
        if x[i] <= x_val <= x[i+1]:
            i0 = i
            break
    return ((x_val - x[i0])**3 * A[i0+1] / (6*h[i0])
          + (x[i0+1] - x_val)**3 * A[i0] / (6*h[i0])
          + b_sp[i0] * x_val + c_sp[i0])

x_plot = np.linspace(a, b, 500)
y_plot = f(x_plot)
Pm_plot = [horner(a_coeffs, xi) for xi in x_plot]
Sf_plot = [eval_spline(xi) for xi in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot,  'k-',  label='f(x)')
plt.plot(x_plot, Pm_plot, 'b--', label=f'Pm(x) m={m}')
plt.plot(x_plot, Sf_plot, 'r-',  label='Sf(x)')
plt.scatter(x, y, color='green', zorder=5, label='nodes')
plt.axvline(x_bar, color='gray', linestyle=':')
plt.legend()
plt.grid(True)
plt.title('Tema 6')
plt.show()