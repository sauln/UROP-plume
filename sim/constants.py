import sys

import lcm
from senlcm import *
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA
import time
import matplotlib.pyplot as plt
#from mysaturation import mysaturation

eps = sys.float_info.epsilon



r = 0.2












kk = 1

l0 = 0.3; l = 0.2; c0 = 500
d11 = 1;m11 = 1; d33 = 1; m33 = 1
k11 = -d11/m11;k21 = -d33/m33
k31 = 1/m11;k41 = 1/(m33*l)
A  = diag( [k11,k21] )
B  = ([k31, k31] ,[-k41, k41])
zreal = matrix([[23.1],[11]])


theta = 2*pi*0
x_sur = zreal-l0*matrix([[float(cos(theta))],[float(sin(theta))]])
xvector = zeros((2,0))
u = matrix([ [.1], [.1] ])
vd = matrix([ [.1], [.1] ])
xytheta = matrix([[float(x_sur[0])],[float(x_sur[1])],[theta]])




n = 1
t0 = 0;					#start time
T_thresh = 2;			#time point to release robot
ts = 11;				#end time
dt = 0.001;             #Time step
Dt = 0.002;				#visualization period
T_leader 	= 6000;		#time to start leader control
Dis_thresh 	= 2000;		#distance to activate leader control

k1 = 0.75;                  
k2 = 0;    				#estimator gradient gain
k3 = 5;                 #adaptive control for estimator gradient gain
k4 = 5000; 				#rotation gain--along the tangent direction

Xhatdot_max		= 50; 	#maximum velocity -- estimated value 
V0_robot_max	= 50; 	#robot velcity
v_compensate	= 0;    #velocity compensation   --- either 0 or 1, (on or off)
free_speed		= 40; 	#leader free speed for partroling
c_leader		= 20; 	#leader gain to drive the two leader close
c_r     		= 50; 	#robot gain to observed value
U_s			= 3;        #Concentration of polutant at the source 
threshold 	= 0.1*U_s;  #threshold of concentration detection

c1 = free_speed


