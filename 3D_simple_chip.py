import math
import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
# Set constants
# theta from degrees to radians
theta = math.radians(20)
# Vector position of the hole
hole = numpy.array([10, 8, 0.1])
h = hole
# Gravitational constant
a = -9.81

# Import useful maths functions
def sin(a):
    return math.sin(a)
def cos(a):
    return math.cos(a)
def tan(a):
    return math.tan(a)
def sec(a):
    return 1.0/cos(a)
def cosec(a):
    return 1.0/sin(a)
def cot(a):
    return 1.0/tan(a)
def atan(a):
    return math.atan(a)
def sqrt(a):
    return math.sqrt(a)

# Calculate gamma
if h[0] != 0:
    gamma = atan(h[1]/h[0])
elif h[1] > 0:
    gamma = math.pi/2
else:
    gamma = 3 * math.pi/2

# Find the value of t when the ball gets to the hole
def find_T():
    # The function is split into two parts to make debugging easier
    p1 = (2*h[2]/a)
    # Use xi and xk if hi does not equal 0
    if h[0] != 0:
        p2 = ((2*h[0]/a)*tan(theta)*sec(gamma))
    # Use xj and xk otherwise
    else:
        p2 = ((2*h[1]/a)*cot(theta)*cosec(gamma))
    # Combine the two parts of the equation
    T = math.sqrt(p1-p2)
    return T

# Find the initial velocity from T value
def find_u0(T):
    if h[0] != 0:
        u0 = h[0]/(T*cos(theta)*cos(gamma))
    else:
        u0 = h[1]/(T*cos(theta)*sin(gamma))
    return u0

# The equation for x(t)
# Returns the vector position of the golf ball at a given time t
def x(u0, t):
    # xi(t)
    i = u0*t*cos(theta)*cos(gamma)
    # xj(t)
    j = u0*t*cos(theta)*sin(gamma)
    # xk(t)
    k = (u0*t*sin(theta))+((a/2)*(t**2))
    # Return x as vector
    this_x = numpy.array([i, j, k])
    return this_x

# The equation for dx(t)/dt
# Returns the vector velocity of the golf ball at a given time t
def dx(u0, t):
    # dxi(t)/dt
    # Constant for simple model
    i = u0*cos(theta)*cos(gamma)
    # dxj(t)/dt
    # Constant for simple model
    j = u0*cos(theta)*sin(gamma)
    # dxk(t)/dt
    k = (u0*sin(theta))+(a*t)
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
    x_range.append(x(u0, t))
    dx_range.append(dx(u0, t))

# Convert x_range to numpy array for graphing
x_range = numpy.array(x_range)

# Create empty arrays for x, y and z for plotting
xplot = []
yplot = []
zplot = []

# Populate x, y and z with the values from the numpy array
for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])
    zplot.append(x_range[i][1])
    
# Plot the graph of xi(t) and xj(t) against xk(t) 
# x = xi(t), y = xk(t), z = xj(t)
fig = plt.figure()
ax = p3.Axes3D(fig)    
ax.plot(xplot, zplot, yplot)
plt.show()