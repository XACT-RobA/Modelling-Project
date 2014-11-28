import math
import numpy
import matplotlib.pyplot as this_plot
from scipy.optimize import fsolve

# Set constants
# theta from degrees to radians
theta = math.radians(20)
# Vector position of the hole
hole = numpy.array([20, 0, 0])
h = hole
# Gravitational constant
g = -9.81
# Air resistance
mu = 0.05
# Mass of golf ball
m = 0.045
# Vector wind coefficient
wind = numpy.array([-10, 0, 0])
# w is an estimate of how much the wind affects the ball
w = mu * wind

# Import useful maths functions
def sin(a):
    return math.sin(a)
def cos(a):
    return math.cos(a)
def tan(a):
    return math.tan(a)
def cosec(a):
    return 1.0/sin(a)
def sec(a):
    return 1.0/cos(a)
def exp(a):
    return math.exp(a)

# Define the function to be solved by fsolve
def F(T):
    # The function is split into parts to make debugging the equation easier
    # pe1 is the value used with e
    pe1 = -(mu/m)*T
    # f is split into twohalves becuase I had errors
    # u0 (i)
    ip1 = (w[0]/mu)*sec(theta)
    ip2 = ((mu*h[0])-(w[0]*T))*sec(theta)
    ip3 = m*(1-exp(pe1))
    fi = ip1+(ip2/ip3)
    # u0 (k)
    kp1 = ((m*g)+w[2])*cosec(theta)/mu
    kp2 = ((mu*h[2])-(T*((m*g)+w[2])))*cosec(theta)
    kp3 = ip3
    fk = kp1+(kp2/kp3)
    # Combine both parts into one equation
    f = fi-fk
    return f

# Use fsolve to find the value of T
# fsolve finds when F(T) = 0 using numerical analysis
def find_T():
    # Use 0 as the starting point
    T = fsolve(F, 0.1)[0]
    return T

# Find the initial velocity from T value
def find_u0(T):
    # The function is split into parts
    # pe1 is the value used with e
    pe1 = -(mu/m)*T
    p1 = (w[0]/mu)*sec(theta)
    p2 = ((mu*h[0])-(w[0]*T))*sec(theta)
    p3 = m*(1-exp(pe1))
    # Combine all of the parts into one equation
    u0 = p1+(p2/p3)
    return u0

# The equation for x(t)
# Returns the vector position of the golf ball at a given time t
def x(u0, t, T):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    # Parts of xi(t)
    ip1 = (w[0]*t)/mu
    ip2 = ((mu*h[0])-(w[0]*T))
    ip3 = 1-exp(pe1)
    ip4 = mu*(1-exp(pe2))
    # Combine the parts of xi(t)
    i = ip1+((ip2*ip3)/ip4)
    # xj = 0 since 2D
    j = 0
    # Parts of xk(t)
    kp1 = ((m*g)+w[2])*t/mu
    kp2 = (mu*h[2])-(T*((m*g)+w[2]))
    kp3 = ip3
    kp4 = ip4
    # Combine the parts of xk(t)
    k = kp1+((kp2*kp3)/kp4)
    # Return x as vector
    this_x = numpy.array([i, j, k])
    return this_x

def dx(u0, t, T):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    # Parts of xi(t)
    ip1 = w[0]/mu
    ip2 = ((mu*h[0])-(w[0]*T))
    ip3 = exp(pe1)
    ip4 = m*(1-exp(pe2))
    # Combine the parts of xi(t)
    i = ip1+((ip2*ip3)/ip4)
    # dxj/dt = 0 since 2D
    j = 0
    # Parts of xk(t)
    kp1 = ((m*g)+w[2])/mu
    kp2 = (mu*h[2])-(T*((m*g)+w[2]))
    kp3 = exp(pe1)
    kp4 = m*(1-exp(pe2))
    # Combine the parts of xk(t)
    k = kp1+((kp2*kp3)/kp4)
    # Return dx/dt as vector
    this_dx = numpy.array([i, j, k])
    return this_dx

# Find T and print the result
T = find_T()
print('T: ' + str(T))
# Find u0 and print the result
u0 = find_u0(T)
print('u0: ' + str(u0))

# Create an array of t values for plotting x(t)
t_range = numpy.arange(0,101*T/100,T/100)

# Create empty arrays for x and dx
x_range = []
dx_range = []

# Populate the x and dx arrays
for t in t_range:
    x_range.append(x(u0, t, T))
    dx_range.append(dx(u0, t, T))

# Create empty arrays for x and y for plotting
xplot = []
yplot = []

# Populate x and y with the values from the array
for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])

# Plot the graph of xi(t) against xk(t) 
# x = xi(t), y = xk(t)
this_plot.plot(xplot, yplot)
this_plot.show()