import numpy as np
from numpy.random import exponential
import statsmodels

# PROBLEM 3
n = 1000
lambda_1 = 1
lambda_2 = 1 / 2
lambda_3 = 1 / 3
lambda_4 = 1 / 4
lambda_5 = 1 / 15
lambda_6 = 1 / 6

def longest_path_length(z1, z2, z3, z4, z5, z6):
    return max(z1 + z2 + z4, z1 + z3, z5 + z6)

## Simulations
Z1 = exponential(1/lambda_1,n)
Z2 = exponential(1/lambda_2,n)
Z3 = exponential(1/lambda_3,n)
Z4 = exponential(1/lambda_4,n)
Z5 = exponential(1/lambda_5,n)
Z6 = exponential(1/lambda_6,n)

L = []
for i in range(n) :
    L.append(longest_path_length(Z1[i],Z2[i],Z3[i],Z4[i],Z5[i],Z6[i]))
L = np.array(L)

# 1 - MC Standard
print("MC (standard) estimator : ", np.mean(L))
print("MC (standard) estimated variance : ", np.var(L))

# 2 - MC Control Variate
C = np.array([Z1 - np.mean(Z1),Z2 - np.mean(Z2), Z3 - np.mean(Z3),
              Z4 - np.mean(Z4),Z5 - np.mean(Z5), Z6 - np.mean(Z6)])

cov_vec = np.array([np.cov(L,C[0,])[0,1],np.cov(L,C[1,])[0,1],np.cov(L,C[2,])[0,1],
                    np.cov(L,C[3,])[0,1],np.cov(L,C[4,])[0,1],np.cov(L,C[5,])[0,1]])
Sigma = np.cov([C[0,],C[1,],C[2,],C[3,],C[4,],C[5,]])
lambda_star = np.dot(np.linalg.inv(Sigma), cov_vec)

L_var_red = L - np.dot(lambda_star, C)
print("MC (standard) estimator : ", np.mean(L_var_red))
print("MC (standard) estimated variance : ", np.var(L_var_red))


# 3 - MC Antithetic Variate