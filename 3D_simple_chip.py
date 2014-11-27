import math
import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

theta = math.radians(20)
hole = numpy.array([10, 8, 0.1])
a = -9.81
sin = math.sin(theta)
cos = math.cos(theta)
tan = math.tan(theta)
cot = 1.0/tan
h = hole

if h[0] != 0:
    gamma = math.atan(h[1]/h[0])
elif h[1] > 0:
    gamma = math.pi/2
else:
    gamma = 3 * math.pi/2
    
cosg = math.cos(gamma)
sing = math.sin(gamma)
secg = 1.0/cosg
cosecg = 1.0/sing

# x = h when t = T
def find_T():
    if h[0] != 0:
        p1 = (2*h[2])/a
        p2 = ((2*h[0]/a)*tan*secg)
        T = math.sqrt(p1-p2)
    else:
        p1 = (2*h[2]/a)
        p2 = ((2*h[1]/a)*cot*cosecg)
        T = math.sqrt(p1-p2)
    return T

def find_u0(T):
    if h[0] != 0:
        u0 = h[0]/(T*cos*cosg)
    else:
        u0 = h[1]/(T*cos*sing)
    return u0

def x(u0, t):
    this_x = numpy.array([(u0*t*cos*cosg), (u0*t*cos*sing), ((u0*t*sin)+((a/2)*(t**2)))])
    return this_x

def dx(u0, t):
    this_dx = numpy.array([(u0*cos*cosg), (u0*cos*sing), ((u0*sin)+(a*t))])
    return this_dx

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