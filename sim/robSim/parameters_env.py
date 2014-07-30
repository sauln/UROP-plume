import sys
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA

#what do these have to do with the environment?



mat_contents 	= sio.loadmat('matlab/constants.mat')
ts 				= float(mat_contents['ts'])
dt 				= float(mat_contents['dt'])
T_thresh 		= float(mat_contents['T_thresh'])
n 				= float(mat_contents['n'])
k1 				= float(mat_contents['k1'])
k2 				= float(mat_contents['k2'])
k3 				= float(mat_contents['k3'])
k4 				= float(mat_contents['k4'])
threshold  		= float(mat_contents['threshold'])	
free_speed 		= float(mat_contents['free_speed'])
T_leader   		= float(mat_contents['T_leader'])
Dis_thresh 		= float(mat_contents['Dis_thresh'])
c_leader   		= float(mat_contents['c_leader'])
c1         		= free_speed
c_r       		= float(mat_contents['c_r'])
Dt 				= float(mat_contents['Dt'])
v_compensate    = float(mat_contents['v_compensate'])
Xhatdot_max		= float(mat_contents['Xhatdot_max'])
V0_robot_max	= float(mat_contents['V0_robot_max'])
