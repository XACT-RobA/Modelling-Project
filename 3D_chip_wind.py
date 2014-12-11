import math
import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

# Set constants
# theta from degrees to radians
theta = math.radians(20)
# Vector position of the hole
hole = numpy.array([10, 8, 0.1])
h = hole
# Gravitational constant
g = -9.81
# Air resistance
mu = 0.05
# Mass of golf ball
m = 0.045
# Vector wind coefficient
wind = numpy.array([-5, 0, 0])
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
def atan(a):
    return math.atan(a)
def exp(a):
    return math.exp(a)

# Calculate gamma
if h[0] != 0:
    gamma = atan(h[1]/h[0])
elif h[1] > 0:
    gamma = math.pi/2
else:
    gamma = 3 * math.pi/2

# Define the function to be solved by fsolve
def F(T):
    # The function is split into parts to make debugging the equation easier
    # pe1 is the value used with e
    pe1 = -(mu/m)*T
    # f is split becuase I had errors
    # Use xi and xk if hi does not equal 0
    if h[0] != 0:
        # u0 (i)
        ip1 = (w[0]/mu)*sec(theta)*sec(gamma)
        ip2 = ((mu*h[0])-(w[0]*T))*sec(theta)*sec(gamma)
        ip3 = m*(1-exp(pe1))
        fi = ip1+(ip2/ip3)
        fij = fi
    # Otherwise use xj and xk
    else:
        # u0 (j)
        jp1 = (w[1]/mu)*sec(theta)*cosec(gamma)
        jp2 = ((mu*h[1])-(w[1]*T))*sec(theta)*cosec(gamma)
        jp3 = m*(1-exp(pe1))
        fj = jp1+(jp2/jp3)
        fij = fj
    # u0 (k)
    kp1 = ((m*g)+w[2])*cosec(theta)/mu
    kp2 = ((mu*h[2])-(T*((m*g)+w[2])))*cosec(theta)
    kp3 = m*(1-exp(pe1))
    fk = kp1+(kp2/kp3)
    # Combine both parts into one equation
    f = fij-fk
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
    if h[0] != 0:
        p1 = (w[0]/mu)*sec(theta)*sec(gamma)
        p2 = ((mu*h[0])-(w[0]*T))*sec(theta)*sec(gamma)
    else:
        p1 = (w[1]/mu)*sec(theta)*cosec(gamma)
        p2 = ((mu*h[1])-(w[1]*T))*sec(theta)*cosec(gamma)
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
    # Parts of xj(t)
    jp1 = (w[1]*t)/mu
    jp2 = ((mu*h[1])-(w[1]*T))
    jp3 = ip3
    jp4 = ip4
    # Combine the parts of xj(t)
    j = jp1+((jp2*jp3)/jp4)
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
    # Parts of dxi(t)/dt
    ip1 = w[0]/mu
    ip2 = ((mu*h[0])-(w[0]*T))
    ip3 = exp(pe1)
    ip4 = m*(1-exp(pe2))
    # Combine the parts of dxi(t)/dt
    i = ip1+((ip2*ip3)/ip4)
    # Parts of dxj(t)/dt
    jp1 = w[1]/mu
    jp2 = ((mu*h[1])-(w[1]*T))
    jp3 = ip3
    jp4 = ip4
    # Combine the parts of dxj(t)/dt
    j = jp1+((jp2*jp3)/jp4)
    # Parts of dxk(t)/dt
    kp1 = ((m*g)+w[2])/mu
    kp2 = (mu*h[2])-(T*((m*g)+w[2]))
    kp3 = ip3
    kp4 = ip4
    # Combine the parts of dxk(t)/dt
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

# Create empty arrays for x, y and z for plotting
xplot = []
yplot = []
zplot = []

# Populate x and y with the values from the array
for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])
    zplot.append(x_range[i][1])
    
# Plot the graph of xi(t) and xj(t) against xk(t) 
# x = xi(t), y = xk(t), z = xj(t)
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot(xplot, zplot, yplot)
ax.set_xlabel('xi(t) (metres)')
ax.set_zlabel('xj(t) (metres)')
ax.set_ylabel('xk(t) (metres)')
plt.show()