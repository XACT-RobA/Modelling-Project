import math
import numpy as np
from bounce import do_bounce as bn
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def z(i, j):
    return ((0.02*i)+(0.0*j))

def dz(i,j):
    return (0.02, 0.0)

# Vector position of the hole
h = hole = np.array([20, 0, z(20, 0)])
# Gravitational constant
a = -9.81

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
def sqrt(a):
    return np.sqrt(a)

# The equation for x(t)
# Returns the vector position of the golf ball at a given time t
def x(u0, theta, gamma, t):
    # xi(t)
    i = u0*t*cos(theta)*cos(gamma)
    # xj(t)
    j = u0*t*cos(theta)*sin(gamma)
    # xk(t)
    k = (u0*t*sin(theta))+((a/2)*(t**2))
    # Return x as vector
    this_x = np.array([i, j, k])
    return this_x

# The equation for dx(t)/dt
# Returns the vector velocity of the golf ball at a given time t
def dx(u0, theta, gamma, t):
    # dxi(t)/dt
    i = u0*cos(theta)*cos(gamma)
    # dxj(t)/dt
    j = u0*cos(theta)*sin(gamma)
    # dxk(t)/dt
    k = (u0*sin(theta))+(a*t)
    # Return dx/dt as vector
    this_dx = np.array([i, j, k])
    return this_dx

def get_distance(new_x):
    distance = sqrt(((h[0]-new_x[0])**2)+((h[1]-new_x[1])**2))
    return distance

def get_total_distance(new_x):
    distance = sqrt(((h[0]-new_x[0])**2)+((h[1]-new_x[1])**2)+((h[2]-new_x[2])**2))
    return distance

def test_speed(u0):
    x_total_array = []
    iter_count = 0
    # theta from degrees to radians
    theta = math.radians(20)
    # Calculate gamma
    if h[0] != 0:
        gamma = atan(h[1]/h[0])
    elif h[1] > 0:
        gamma = math.pi/2
    else:
        gamma = 3 * math.pi/2
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
            #print(total_distance_to_hole)
            if total_distance_to_hole < closest_total_distance_to_hole:
                closest_total_distance_to_hole = total_distance_to_hole
            if distance_to_hole >= closest_distance_to_hole:
                getting_closer = False
            else:
                closest_distance_to_hole = distance_to_hole
            x_array.append((new_x, this_t, closest_distance_to_hole))
        closest_to_hole.append(closest_total_distance_to_hole)
        #print(closest_total_distance_to_hole)
        x_total_array.append(np.array(x_array))
    return (x_total_array, iter_count, closest_to_hole)

def do_loop(u0_lower, u0_upper):
    u0_range = np.arange(u0_lower, u0_upper, ((u0_upper-u0_lower)/500.0))
    x_total_array = [None]*len(u0_range)
    iter_count = [None]*len(u0_range)
    closest_to_hole = [None]*len(u0_range)
    for i in range(len(u0_range)):
        (x_total_array[i], iter_count[i], closest_to_hole[i]) = test_speed(u0_range[i])
    return (x_total_array, iter_count, closest_to_hole, u0_range)
        

(x_total_array, iter_count, closest_to_hole, u0_range) = do_loop(0, 20)
print(sum(iter_count))
putted = []

for i in range(len(closest_to_hole)):
    if closest_to_hole[i][0] <= 0.05:
        putted.append(True)
    else:
        putted.append(False)

for i in range(len(putted)):
    if putted[i]:
        print(u0_range[i])

'''
xplot = []
yplot= []
zplot = []
for i in range(len(x_total_array[0])):
    xplot.append(x_total_array[0][i][0][0])
    yplot.append(x_total_array[0][i][0][2])
    zplot.append(x_total_array[0][i][0][1])

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.plot(xplot, zplot, yplot)
plt.show()
'''