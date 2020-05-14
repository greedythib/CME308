import numpy as np
from scipy import optimize
from scipy.stats import weibull_min
from math import gamma

# Problem 3
T1 = 44 * 10 ** 3
T2 = 26 * 10 ** 3
T3 = 100 * 10 ** 3
T4 = 19 * 10 ** 3
T5 = 45 * 10 ** 3


# def L(gamma, alpha, t1, t2, t3, t4, t5) :
#     # uncensored contribution
#     A = ((alpha/gamma)**2) * ((1/gamma)**(alpha-1)) * (t1*t2) * np.exp((1/gamma)*((-t1)**alpha + (-t2)**alpha))
#     # censored contribution
#     B = np.exp((1/gamma)**alpha) * ((t3**alpha) + (t4**alpha) + (t5**alpha))
#     return A * B

def neg_log_L(gamma, alpha) -> float:
    A = 2 * np.log(alpha / gamma) + (alpha - 1) * np.log(T1 * T2 / (gamma ** 2)) - \
        ((1 / gamma) ** alpha) * ((T1 ** alpha) + (T2 ** alpha))
    B = -((1 / gamma) ** alpha) * (T3 ** alpha + T4 ** alpha + T5 ** alpha)
    return -A - B


## Numerically solve MLE equations.
epsilon = 1 * 10 ** -12
x0 = [10, 10]
bnds = ((epsilon, np.inf), (epsilon, np.inf))
fun = lambda x: neg_log_L(x[0], x[1])
solver = optimize.minimize(fun, x0=x0, bounds=bnds)
print(" {} iterations \n".format(solver.nit),
      "lambda = {} and alpha = {}".format(1 / solver.x[0], solver.x[1]))

## Parametric Boostrap (1-delta) confidence interval
lmda = 1 / solver.x[0]
alpha = solver.x[1]
# reference parameter
mu_star = (1 / lmda) * gamma(1 + 1 / alpha)
print("In average, {} kms before a reliability problem occurs.".format(mu_star))

m = 100  # number of iterations to aggregate
bootstrap_estimates = []
for i in range(m):
    T_bootstrap = weibull_min.rvs(c=alpha, scale=1 / lmda, size=100)
    bootstrap_estimates.append(np.mean(T_bootstrap))

delta = 0.1
# Upper bound : we estimate prob(estimate - mu_star < x) = 1-delta/2
x = 2000
count = 0
while (count / m != 1 - delta / 2):
    count = 0
    for i in range(m):
        if bootstrap_estimates[i] - mu_star < x:
            count += 1
    print(count / m)
    x += 10
print(" Prob(mu_n - mu_estimate < {}) = {}".format(x, count / m))

# Lowe bound : we estimate prob(estimate - mu_star < y ) = delta/2
y = -10000
count = 0
while (count / m != delta / 2):
    count = 0
    for i in range(m):
        if bootstrap_estimates[i] - mu_star < y:
            count += 1
    print(count / m)
    y += 10
print(" Prob(mu_n - mu_estimate < {}) = {}".format(y, count / m))

print(" Parametric Boostrap {}% confidence interval : [{},{}]".format(1 - delta, -x + np.mean(bootstrap_estimates),
                                                                      - y + np.mean(bootstrap_estimates)))
