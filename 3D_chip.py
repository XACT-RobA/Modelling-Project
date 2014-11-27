import math
import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

theta = math.radians(20)
hole = numpy.array([10, 8, 0.1])
h = hole
g = -9.81
mu = 0.05
m = 0.045

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
def exp(a):
    return math.exp(a)

if h[0] != 0:
    gamma = atan(h[1]/h[0])
elif h[1] > 0:
    gamma = math.pi/2
else:
    gamma = 3 * math.pi/2

def F(T):
    pe1 = -(mu/m)*T
    p1 = T*mu*m*g
    p2 = (m**2)*g
    p3 = 1-exp(pe1)
    if h[0] != 0: 
        p4 = (mu**2)*h[0]*tan(theta)*sec(gamma)
    else:
        p4 = (mu**2)*h[1]*tan(theta)*cosec(gamma)
    p5 = (mu**2)*h[2]
    f = p1-(p2*p3)+p4-p5
    return f

def find_T():
    T = fsolve(F, 0)
    return T

def find_u0(T):
    pe1 = -(mu/m)*T
    if h[0] != 0:
        p1 = mu*h[0]*sec(theta)*sec(gamma)
    else:
        p1 = mu*h[1]*sec(theta)*cosec(gamma)
    p2 = m*(1-exp(pe1))
    u0 = p1/p2
    return u0

def x(u0, t, T):
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    
    ip1 = h[0]*(1-exp(pe1))
    ip2 = 1-exp(pe2)
    i = ip1/ip2
    
    jp1 = h[1]*(1-exp(pe1))
    jp2 = 1-exp(pe2)
    j = ip1/ip2
    
    kp1 = (m*g*t)/mu
    kp2 = (mu*h[2])-(m*g*T)
    kp3 = mu*(1-exp(pe2))
    kp4 = (m*g*T)-(mu*h[2])
    kp5 = exp(pe1)
    kp6 = kp3
    k = kp1+(kp2/kp3)+((kp4*kp5)/kp6)
    
    this_x = numpy.array([i, j, k])
    return this_x

def dx(u0, t, T):
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    
    ip1 = mu*h[0]*exp(pe1)
    ip2 = m*(1-exp(pe2))
    i = ip1/ip2
    
    jp1 = mu*h[1]*exp(pe1)
    jp2 = m*(1-exp(pe2))
    j = jp1/jp2
    
    kp1 = (m*g)/mu
    kp2 = (mu*h[2])-(T*m*g)
    kp3 = exp(pe1)
    kp4 = m*(1-exp(pe2))
    k = kp1+((kp2*kp3)/kp4)
    
    this_dx = numpy.array([i, j, k])
    return this_dx

T = find_T()[0]
print('T: ' + str(T))
u0 = find_u0(T)
print('u0: ' + str(u0))
    
t_range = numpy.arange(0,101*T/100,T/100)

x_range = []
dx_range = []

for t in t_range:
    x_range.append(x(u0, t, T))
    dx_range.append(dx(u0, t, T))
    
xplot = []
yplot = []
zplot = []

for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])
    zplot.append(x_range[i][1])
    
fig = plt.figure()
ax = p3.Axes3D(fig)
    
ax.plot(xplot, zplot, yplot)
plt.show()




