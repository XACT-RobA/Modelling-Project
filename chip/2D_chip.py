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

# Import useful maths functions
def sin(a):
    return math.sin(a)
def cos(a):
    return math.cos(a)
def tan(a):
    return math.tan(a)
def sec(a):
    return 1.0/cos(a)
def exp(a):
    return math.exp(a)

# Define the function to be solved by fsolve
def F(T):
    # The function is split into parts to make debugging the equation easier
    # pe1 is the value used with e
    pe1 = -(mu/m)*T
    # pn is part n
    p1 = T*mu*m*g
    p2 = (m**2)*g
    p3 = 1-exp(pe1)
    p4 = (mu**2)*h[0]*tan(theta)
    p5 = (mu**2)*h[2]
    # Combine all of the parts into one equation
    f = p1-(p2*p3)+p4-p5
    return f

# Use fsolve to find the value of T
# fsolve finds when F(T) = 0 using numerical analysis
def find_T():
    # Use 0 as the starting point
    T = fsolve(F, 0)[0]
    return T

# Find the initial velocity from T value
def find_u0(T):
    # The function is split into parts
    # pe1 is the value used with e
    pe1 = -(mu/m)*T
    p1 = mu*h[0]*sec(theta)
    p2 = m*(1-exp(pe1))
    # Combine all of the parts into one equation
    u0 = p1/p2
    return u0

# The equation for x(t)
# Returns the vector position of the golf ball at a given time t
def x(u0, t, T):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    # Parts of xi(t)
    ip1 = h[0]*(1-exp(pe1))
    ip2 = 1-exp(pe2)
    # Combine the parts of xi(t)
    i = ip1/ip2
    # Parts of xk(t)
    kp1 = (m*g*t)/mu
    kp2 = (mu*h[2])-(m*g*T)
    kp3 = mu*(1-exp(pe2))
    kp4 = (m*g*T)-(mu*h[2])
    kp5 = exp(pe1)
    kp6 = kp3
    # Combine the parts of xk(t)
    k = kp1+(kp2/kp3)+((kp4*kp5)/kp6)
    # Return x as vector
    this_x = numpy.array([i, 0, k])
    return this_x

def dx(u0, t, T):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    # Parts of xi(t)
    ip1 = mu*h[0]*exp(pe1)
    ip2 = m*(1-exp(pe2))
    # Combine the parts of xi(t)
    i = ip1/ip2
    # xj = 0 since 2D
    j = 0
    # Parts of xk(t)
    kp1 = (m*g)/mu
    kp2 = (mu*h[2])-(T*m*g)
    kp3 = exp(pe1)
    kp4 = m*(1-exp(pe2))
    # Combine the parts of xk(t)
    k = kp1+((kp2*kp3)/kp4)
    # Return dx/dt as vector
    this_dx = numpy.array([i, 0, k])
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
this_plot.xlabel('xi(t) (metres)')
this_plot.ylabel('xj(t) (metres)')
this_plot.show()