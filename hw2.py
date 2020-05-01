import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential
import statsmodels
import scipy.stats

# PROBLEM 3
n = 10000
lambda_1 = 1
lambda_2 = 1 / 2
lambda_3 = 1 / 3
lambda_4 = 1 / 4
lambda_5 = 1 / 15
lambda_6 = 1 / 6

def longest_path_length(z1, z2, z3, z4, z5, z6):
    return max(z1 + z2 + z4, z1 + z3, z5 + z6)

def inverse_cdf_exponential_law(x,mean) :
    return mean * np.log(1 / (1-x))

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
print("MC (standard) estimated variance : ", (1/n) * np.var(L))

# 2 - MC Control Variate
C = np.array([Z1 - np.mean(Z1),Z2 - np.mean(Z2), Z3 - np.mean(Z3),
              Z4 - np.mean(Z4),Z5 - np.mean(Z5), Z6 - np.mean(Z6)])

cov_vec = np.array([np.cov(L,C[0,])[0,1],np.cov(L,C[1,])[0,1],np.cov(L,C[2,])[0,1],
                    np.cov(L,C[3,])[0,1],np.cov(L,C[4,])[0,1],np.cov(L,C[5,])[0,1]])
Sigma = np.cov([C[0,],C[1,],C[2,],C[3,],C[4,],C[5,]])
lambda_star = np.dot(np.linalg.inv(Sigma), cov_vec)

L_CV = L - np.dot(lambda_star, C)
print("MC (control variate) estimator : ", np.mean(L_CV))
print("MC (control variate) estimated variance : ", (1/n) * np.var(L_CV))

# (1-alpha) % confidence interval.
alpha = 0.1
z = scipy.stats.norm.ppf(1-alpha/2)
inf = np.mean(L_CV) - (z/np.sqrt(n)) * np.var(L_CV)
sup = np.mean(L_CV) + (z/np.sqrt(n)) * np.var(L_CV)
print("({})% confidence interval : ".format((1-alpha)*100),"[",inf,",", sup, "]")

# 3 - MC Antithetic Variates
# TODO : Better to sample 6 uniforms at each simulation
U = np.random.uniform(0,1,n)

Z1_bis = [] ; Z1_bis_trans = []
Z2_bis = [] ; Z2_bis_trans = []
Z3_bis = [] ; Z3_bis_trans = []
Z4_bis = [] ; Z4_bis_trans = []
Z5_bis = [] ; Z5_bis_trans = []
Z6_bis = [] ; Z6_bis_trans = []

for i in range(n) : 
    Z1_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_1))
    Z2_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_2))
    Z3_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_3))
    Z4_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_4))
    Z5_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_5))
    Z6_bis.append(inverse_cdf_exponential_law(U[i], 1 / lambda_6))
    # antithetic tranformation of the sample U
    Z1_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_1))
    Z2_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_2))
    Z3_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_3))
    Z4_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_4))
    Z5_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_5))
    Z6_bis_trans.append(inverse_cdf_exponential_law(1-U[i], 1 / lambda_6))

L_anti = []
L_anti_trans = []
for i in range(n) :
    L_anti.append(longest_path_length(Z1_bis[i],Z2_bis[i],Z3_bis[i],
                                      Z4_bis[i],Z5_bis[i],Z6_bis[i]))
    L_anti_trans.append(longest_path_length(Z1_bis_trans[i], Z2_bis_trans[i],
                                            Z3_bis_trans[i], Z4_bis_trans[i],
                                            Z5_bis_trans[i], Z6_bis_trans[i]))
L_anti = np.array(L_anti)
L_anti_trans = np.array(L_anti_trans)

print("MC (antithetic variate) estimator : ",
      0.5 * np.mean(L_anti + L_anti_trans))
print("MC (antithetic variate) estimated variance : ",
      (0.25 / n) * np.var(L_anti + L_anti_trans))

# (1-alpha) % confidence interval.
alpha = 0.1
z = scipy.stats.norm.ppf(1-alpha/2)
inf = 0.5 * np.mean(L_anti + L_anti_trans) - (z/np.sqrt(n)) * np.var(L_var_red)
sup = 0.5 * np.mean(L_anti + L_anti_trans) + (z/np.sqrt(n)) * np.var(L_var_red)
print("({})% confidence interval : ".format((1-alpha)*100),"[",inf,",", sup, "]")


# PROBLEM 4
mean = np.array([0,0])
cov = np.array([[1,-0.7],[-0.7,1]])
# Simluations
n = 100000
X_Y = np.random.multivariate_normal(mean, cov, size=n)
# np.cov(X_Y[:,0], X_Y[:,1])
# Plots
plt.scatter(x=X_Y[:,0], y=X_Y[:,1])
plt.vlines(3, ymin=3, ymax = 8, color = "red", linestyles='solid')
plt.hlines(3, xmin=3, xmax = 8, color = "red")
plt.title("n = {} simulations".format(n))
plt.show()

# zone = [(x,y) for x,y in X_Y if x >= 1 and y >= 1]
# while len(zone) == 0 :
#     X_Y = np.random.multivariate_normal(mean, cov, size=n)
#     zone = [(x, y) for x, y in X_Y if x > 3 and y > 3]


# Importance Sampling
n = 100000
mean_2 = np.array([3,3])
Z = np.random.multivariate_normal(mean_2, cov, size=n)
f = scipy.stats.multivariate_normal(mean=[0,0], cov=[[1,-0.7],[-0.7,1]])
g = scipy.stats.multivariate_normal(mean=[3,3], cov=[[1,-0.7],[-0.7,1]])

# Plots
plt.scatter(x=Z[:,0], y=Z[:,1])
plt.vlines(3, ymin=3, ymax = 8, color = "red", linestyles='solid')
plt.hlines(3, xmin=3, xmax = 8, color = "red")
plt.title("n = {} simulations".format(n))
plt.show()

ratio = [] # only if Z belongs to A
for i in range(n) :
    if Z[i][0] >= 3 and Z[i][1] > 3 :
        ratio.append(f.pdf(Z[i]) / g.pdf(Z[i]))
ratio = np.array(ratio)

print("p_n = ", np.mean(ratio))