import math
import numpy as np

# Coefficient of restitution
alpha = 0.8

def atan(a):
    return math.atan(a)
def sqrt(a):
    return np.sqrt(a)

def do_bounce((ui, uj, uk), (xi, xj, xk), (dzi, dzj)):
    theta_in = atan(U[2]/U[0])
    gamma_in = atan(U[2]/U[1])
    omega_i = atan(dzi)
    omega_j = atain(dzj)
    theta_out = theta_in+omega_i
    gamma_out = gamma_in+omega_j
    U = sqrt((ui**2)+(uj**2)+(uk**2))
    V = alpha*U
    return (V, theta_out, gamma_out)