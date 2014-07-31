import sys

import lcm
from senlcm import *
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA
import time
import matplotlib.pyplot as plt
from mysaturation import mysaturation

eps = sys.float_info.epsilon

kk = 1

l0 = 0.3
d11 = 1
m11 = 1
d33 = 1
m33 = 1
l = 0.2
c0 = 500
k11 = -d11/m11
k21 = -d33/m33
k31 = 1/m11
k41 = 1/(m33*l)
A  = diag( [k11,k21] )
B  = ([k31, k31] ,[-k41, k41])

zreal = matrix([[11],[24]])
theta = 2*pi*random.rand(1,1)
x_sur = zreal-l0*matrix([[float(cos(theta))],[float(sin(theta))]])
xvector = zeros((2,0))
u = matrix([ [.5], [.5] ])
vd = matrix([ [0], [0] ])
xytheta = matrix([[float(x_sur[0])],[float(x_sur[1])],[theta]])
