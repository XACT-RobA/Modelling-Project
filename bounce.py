import math
import numpy as np

# Coefficient of restitution
alpha = 0.5

# Import useful maths functions
def atan(a):
    return math.atan(a)
def sqrt(a):
    return np.sqrt(a)

# Function to calculate the new velocity, and angles of the ball after it
# bounces on the surface z
def do_bounce(this_dx, (dzi, dzj)):
    [ui, uj, uk] = this_dx
    # No dividing by zero in my code!
    if ui != 0.0:
        theta_in = atan(uk/ui)
    else:
        theta_in = 0.0
        
    if uj != 0.0:
        gamma_in = atan(uj/ui)
    else:
        gamma_in = 0.0
        
    # Calculate new angles
    omega_i = atan(dzi)
    omega_j = atan(dzj)
    theta_out = theta_in+omega_i
    gamma_out = gamma_in+omega_j
    # Calculate the input speed from input velocity
    U = sqrt((ui**2)+(uj**2)+(uk**2))
    # Calculate output speed
    V = alpha*U
    return (V, theta_out, gamma_out)

