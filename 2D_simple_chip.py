import math
import numpy
import matplotlib.pyplot as this_plot

theta = math.radians(20)
hole = numpy.array([20, 0, 0])
a = -9.81
sin = math.sin(theta)
cos = math.cos(theta)
tan = math.tan(theta)
h = hole

# x = h when t = T
def find_T():
    p1 = 2*h[2]/a
    p2 = (2*h[0]/a)*tan
    T = math.sqrt(p1 - p2)
    return T

def find_u0(T):
    u0 = h[0]/(T*cos)
    return u0

def x(u0, t):
    this_x = numpy.array([(u0*t*cos), 0, (u0*t*sin)+((a/2)*(t**2))])
    return this_x

def dx(u0, t):
    this_dx = numpy.array([(u0*cos), 0, (u0*sin)+(a*t)])
    
T = find_T()
print('T: ' + str(T))
u0 = find_u0(T)
print('u0: ' + str(u0))

t_range = numpy.arange(0,101*T/100,T/100)

x_range = []
dx_range = []

for t in t_range:
    x_range.append(x(u0, t))
    dx_range.append(dx(u0, t))
    
x_range = numpy.array(x_range)
#print(x_range)

xplot = []
yplot = []

for i in range(len(x_range)):
    xplot.append(x_range[i][0])
    yplot.append(x_range[i][2])

this_plot.plot(xplot, yplot)
#this_plot.plot(t_range, x_range)
this_plot.show()