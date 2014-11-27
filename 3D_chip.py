import math
import numpy
from scipy.optimize import fsolve

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
def asin(a):
    return math.asin(a)
def acos(a):
    return math.acos(a)
def atan(a):
    return math.atan(a)
def exp(a):
    return math.exp(a)

