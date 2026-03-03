import math
import random
import time

def exercise_1():
    m = 0
    u = 1
    lastu = u
    while ((1.0 + u) != 1.0):
        lastu = u
        m = m + 1
        u = u / 10
    m = m - 1
    u = u * 10
    return u

print("1)", exercise_1())

def exercise_2a():
    u = exercise_1()
    x = 1.0
    y = u / 10
    z = u / 10
    if (((x + y) + z) != (x + (y + z))):
        print("2a) Addition is not associative.")
        left = (x + y) + z
        right = x + (y + z)
        print("   (x + y) + z =", repr(left))
        print("   x + (y + z) =", repr(right))

exercise_2a()

def exercise_2b():
    x = 0.1
    y = 0.2
    z = 0.3
    left  = (x * y) * z
    right = x * (y * z)
    if left != right:
        print("2b) Multiplication is not associative.")
        print("   (x * y) * z =", repr(left))
        print("   x * (y * z) =", repr(right))

exercise_2b()

def tan_poly(x):
    if x < 0:
        return -tan_poly(-x)
    if x > math.pi / 4:
        reduced = math.pi / 2 - x
        return 1.0 / tan_poly(reduced)
    c0 = 1.0
    c1 = 1.0 / 3
    c2 = 2.0 / 15
    c3 = 17.0 / 315
    c4 = 62.0 / 2835
    t = x * x
    P = c4
    P = c3 + t * P
    P = c2 + t * P
    P = c1 + t * P
    P = c0 + t * P
    return x * P

def tan_cf(x, epsilon=1e-10):
    tiny = 1e-30
    max_iter = 100
    f = tiny
    C = tiny
    D = 0.0
    a = x
    b = 1.0
    D = b + a * D
    if abs(D) < tiny:
        D = tiny
    D = 1.0 / D
    C = b + a / C
    if abs(C) < tiny:
        C = tiny
    Delta = C * D
    f = f * Delta
    for j in range(2, max_iter + 1):
        a = -x * x
        b = 2 * j - 1
        D = b + a * D
        if abs(D) < tiny:
            D = tiny
        D = 1.0 / D
        C = b + a / C
        if abs(C) < tiny:
            C = tiny
        Delta = C * D
        f = f * Delta
        if abs(Delta - 1.0) < epsilon:
            break
    return f

def exercise_3():
    print("\n3) Approximating tan(x)")
    N = 10000

    test_xs = []
    for i in range(N):
        test_xs.append(random.uniform(-math.pi/2 + 0.01, math.pi/2 - 0.01))

    start_poly = time.time()
    poly_results = []
    for i in range(N):
        poly_results.append(tan_poly(test_xs[i]))
    end_poly = time.time()
    time_poly = end_poly - start_poly

    start_cf = time.time()
    cf_results = []
    for i in range(N):
        cf_results.append(tan_cf(test_xs[i]))
    end_cf = time.time()
    time_cf = end_cf - start_cf

    max_err_poly = 0
    max_err_cf = 0
    sum_err_poly = 0
    sum_err_cf = 0

    for i in range(N):
        exact = math.tan(test_xs[i])
        err_poly = abs(exact - poly_results[i])
        err_cf = abs(exact - cf_results[i])
        sum_err_poly = sum_err_poly + err_poly
        sum_err_cf = sum_err_cf + err_cf
        if err_poly > max_err_poly:
            max_err_poly = err_poly
        if err_cf > max_err_cf:
            max_err_cf = err_cf

    avg_err_poly = sum_err_poly / N
    avg_err_cf = sum_err_cf / N

    print("\n   Testing on", N, "random values in (-pi/2, pi/2)\n")

    print("ERRORS:")
    print("   Polynomial: worst", "%.2e" % max_err_poly, " average", "%.2e" % avg_err_poly)
    print("   Continued fraction: worst", "%.2e" % max_err_cf, " average", "%.2e" % avg_err_cf)
    ratio_err = avg_err_poly / avg_err_cf
    print("   Continued fraction is ~%.0f" % ratio_err, "times more accurate.")

    print("TIME:")
    print("   Polynomial: %.4f" % time_poly, "seconds")
    print("   Continued fraction: %.4f" % time_cf, "seconds")
    if time_poly < time_cf:
        ratio_time = time_cf / time_poly
        print("   Polynomial is ~%.1f" % ratio_time, "times faster.")
    else:
        ratio_time = time_poly / time_cf
        print("   Continued fraction is ~%.1f" % ratio_time, "times faster.")
    print("\n")
    print("   Continued fraction: slower but much more accurate.")
    print("   Polynomial has faster but limited precision (~10^-4 err)")

exercise_3()
