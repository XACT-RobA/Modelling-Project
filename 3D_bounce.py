import math
import time
import numpy as np
from bounce import do_bounce as bn
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def z(i, j):
    return ((0.02*i)+(0.02*j))

def dz(i,j):
    return (0.02, 0.02)

starttime = time.time()

# Vector position of the hole
h = hole = np.array([20, 0, z(20, 0)])
# Gravitational constant
a = -9.81
# Air resistance
mu = 0.05
# Mass of golf ball
m = 0.045
wind = np.array([-5, 0, 0])
w = wind*mu

# Number of bounces
n = 1

def sin(a):
    return math.sin(a)
def cos(a):
    return math.cos(a)
def tan(a):
    return math.tan(a)
def atan(a):
    return math.atan(a)
def exp(a):
    return math.exp(a)
def sqrt(a):
    return np.sqrt(a)

pi = math.pi

if h[0] != 0:
    gammai = atan(h[1]/h[0])
elif h[1] > 0:
    gammai = math.pi/2
else:
    gammai = 3 * math.pi/2

# The equation for x(t)
# Returns the vector position of the golf ball at a given time t
def x(u0, theta, gamma, t):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    # Parts of xi(t)
    ip1 = w[0]*t/mu
    ip2 = m/mu
    ip3 = u0*cos(theta)*cos(gamma)
    ip4 = w[0]/mu
    ip5 = 1-exp(pe1)
    # Combine the parts of xi(t)
    i = ip1+(ip2*(ip3-ip4)*ip5)
    # Parts of xj(t)
    jp1 = w[1]*t/mu
    jp2 = ip2
    jp3 = u0*cos(theta)*sin(gamma)
    jp4 = w[1]/mu
    jp5 = ip5
    # Combine the parts of xi(t)
    j = jp1+(jp2*(jp3-jp4)*jp5)
    # Parts of xk(t)
    kp1 = ((m*a)+w[2])*t/mu
    kp2 = ip2
    kp3 = u0*sin(theta)
    kp4 = ((m*a)+w[2])/mu
    kp5 = 1-exp(pe1)
    # Combine the parts of xk(t)
    k = kp1+(kp2*(kp3-kp4)*kp5)
    #print(k)
    # Return x as vector
    this_x = np.array([i, j, k])
    return this_x

# The equation for dx(t)/dt
# Returns the vector velocity of the golf ball at a given time t
def dx(u0, theta, gamma, t):
    # The functions are split into parts
    # pe1 and pe2 are values used with e
    pe1 = -(mu/m)*t
    # Parts of xi(t)
    ip1 = w[0]/mu
    ip2 = u0*cos(theta)*cos(gamma)
    ip3 = exp(pe1)
    # Combine the parts of dxi(t)/dt
    i = ip1+((ip2-ip1)*ip3)
    # xj = 0 since 2D
    jp1 = w[1]/mu
    jp2 = u0*cos(theta)*sin(gamma)
    jp3 = ip3
    # Combine the parts of dxj(t)/dt
    j = jp1+((jp2-jp1)*jp3)
    # Parts of xk(t)
    kp1 = ((m*a)+w[2])/mu
    kp2 = u0*sin(theta)
    kp3 = ip3
    # Combine the parts of dxk(t)/dt
    k = kp1+((kp2-kp1)*kp3)
    # Return dx/dt as vector
    this_dx = np.array([i, j, k])
    return this_dx

def get_distance(new_x):
    distance = sqrt(((h[0]-new_x[0])**2)+((h[1]-new_x[1])**2))
    return distance

def get_total_distance(new_x):
    distance = sqrt(((h[0]-new_x[0])**2)+((h[1]-new_x[1])**2)+((h[2]-new_x[2])**2))
    return distance

def test_speed(u0, gamma):
    x_total_array = []
    iter_count = 0
    # theta from degrees to radians
    theta = math.radians(20)
    u0_range = [u0]
    closest_to_hole = []
    for i in range(len(u0_range)):
        this_u0 = u0_range[i]
        this_t = 0
        x_array = []
        getting_closer = True
        bounces = 0
        bounce_times = np.array([0])
        bounce_positions = [(0, 0, 0)]
        old_x = np.array([0, 0, 0])
        distance_to_hole = get_distance(old_x)
        closest_distance_to_hole = distance_to_hole
        closest_total_distance_to_hole = distance_to_hole
        while getting_closer == True:
            iter_count += 1
            this_t += 0.01
            if bounces == 0:
                new_x = x(this_u0, theta, gamma, (this_t-np.sum(bounce_times)))
            else:
                new_x = x(this_u0, theta, gamma, (this_t-bounce_times[-1])) + bounce_positions[-1][0]
            this_z = z(new_x[0], new_x[1])
            this_dz = dz(new_x[0], new_x[1])
            if new_x[2] < this_z:
                new_x[2] = this_z
                bounces += 1
                bounce_times = np.append(bounce_times, this_t)
                bounce_positions.append([new_x])
                this_dx = dx(this_u0, theta, gamma, (this_t-bounce_times[-1]))
                (this_u0, theta, gamma) = bn(this_dx, this_dz)
            distance_to_hole = get_distance(new_x)
            total_distance_to_hole = get_total_distance(new_x)
            if total_distance_to_hole < closest_total_distance_to_hole:
                closest_total_distance_to_hole = total_distance_to_hole
            if distance_to_hole >= closest_distance_to_hole:
                getting_closer = False
            else:
                closest_distance_to_hole = distance_to_hole
            x_array.append((new_x, this_t, closest_distance_to_hole))
        closest_to_hole.append(closest_total_distance_to_hole)
        x_total_array.append(np.array(x_array))
    return (x_total_array, iter_count, closest_to_hole)

def do_loop(u0_lower, u0_upper, gamma):
    u0_range = np.arange(u0_lower, u0_upper, ((u0_upper-u0_lower)/500.0))
    x_total_array = [None]*len(u0_range)
    iter_count = [None]*len(u0_range)
    closest_to_hole = [None]*len(u0_range)
    for i in range(len(u0_range)):
        (x_total_array[i], iter_count[i], closest_to_hole[i]) = test_speed(u0_range[i], gamma)
    return (x_total_array, iter_count, closest_to_hole, u0_range)

gamma_range = np.arange(gammai-(pi/4), gammai+(pi/4), (pi/2)/100)

total_range = []
x_total_array_array = []
iter_count_array = []
closest_to_hole_array = []
u0_range_array = []

for g in range(len(gamma_range)):
    gamma = gamma_range[g]
    (x_total_array, iter_count, closest_to_hole, u0_range) = do_loop(0, 40, gamma)
    x_total_array_array.append(x_total_array)
    iter_count_array.append(iter_count)
    closest_to_hole_array.append(closest_to_hole)
    u0_range_array.append(u0_range)
    
#print(gamma_range)
#print(closest_to_hole_array)
    
endtime = time.time()
total_iter = 0
for i in range(len(iter_count_array)):
    total_iter += sum(iter_count_array[i])
print(str(total_iter) + ' iterations in ' + str(endtime-starttime) + ' seconds')
putted = []

for i in range(len(closest_to_hole_array)):
    for j in range(len(closest_to_hole_array[i])):
        #print(closest_to_hole[j][i])
        if closest_to_hole[j][i] <= 0.054:
            putted.append('u0: ' + str(u0_range[i]) + ' , gamma: ' + str(gamma_range[j]))

print(putted)
        
        
'''
(x_array, iter_count, closest_to_hole) = test_speed(34.3)
        
xplot = []
yplot= []
zplot = []
print(len(x_array[0]))
for i in range(len(x_array[0])):
    xplot.append(x_array[0][i][0][0])
    yplot.append(x_array[0][i][0][2])
    zplot.append(x_array[0][i][0][1])
    
print yplot

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot(xplot, zplot, yplot)
plt.show()
'''