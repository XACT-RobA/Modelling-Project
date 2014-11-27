import math
import numpy
import matplotlib.pyplot as this_plot
from scipy.optimize import fsolve

theta = math.radians(20)
hole = numpy.array([20, 0, 0])
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
def exp(a):
    return math.exp(a)

def F(T):
    pe1 = -(mu/m)*T
    p1 = T*mu*m*g
    p2 = (m**2)*g*cos(theta)
    p3 = 1-exp(pe1)
    p4 = (mu**2)*h[0]*sec(theta)
    p5 = (mu**2)*h[2]
    f = p1-(p2*p3)+p4-p5
    return f

def find_T():
    T = fsolve(F, 0)
    return T

def find_u0(T):
    pe1 = -(mu/m)*T
    p1 = mu*h[0]*sec(theta)
    p2 = m*(1-exp(pe1))
    u0 = p1/p2
    return u0

def x(u0, t, T):
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    
    ip1 = h[0]*(1-exp(pe1))
    ip2 = 1-exp(pe2)
    i = ip1/ip2
    
    kp1 = (m*g*t)/mu
    kp2 = (mu*h[2])-(m*g*T)
    kp3 = mu*(1-exp(pe2))
    kp4 = (m*g*T)-(mu*h[2])
    kp5 = exp(pe1)
    kp6 = kp3
    k = kp1+(kp2/kp3)+((kp4*kp5)/kp6)
    
    '''
    kp1 = (t*m*g)/mu
    kp2 = (T*m*g)-(mu*h[2])
    kp3 = 1+exp(pe1)
    kp4 = mu*(1-exp(pe2))
    k = kp1-((kp2*kp3)/kp4)
    '''
    
    this_x = numpy.array([i, 0, k])
    return this_x

def dx(u0, t, T):
    pe1 = -(mu/m)*t
    pe2 = -(mu/m)*T
    
    ip1 = mu*h[0]*exp(pe1)
    ip2 = m*(1-exp(pe2))
    
    kp1 = (m*g)/mu
    kp2 = (mu*h[2])-(T*m*g)
    kp3 = exp(pe1)
    kp4 = m*(1-exp(pe2))
    
    this_dx = kp1+((kp2*kp3)/kp4)
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

for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])

this_plot.plot(xplot, yplot)
this_plot.show()
    
    
    
    
    
    
    
