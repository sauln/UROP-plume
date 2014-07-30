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




from robot_model_par import kk, l0, d11, m11, d33, m33, l, c0, k11, k21, k31, k41, A, B, zreal, theta, x_sur, xvector, u, vd, xytheta
from parameters_env import Dt

eps = sys.float_info.epsilon
dumMsg = positionSim_t()
X0 = matrix(dumMsg.X0).H






def conHandler(channel, data):


	#we have to clean up all these global variables
	#most of them aren't ever used anyways...
	global X0
	global u 
	global xytheta
	global theta
	global A
	global B
	global zreal
	global l0
	global Dt

	#print "message received on channel: %s" % channel
	msg  = positionSim_t.decode(data)
	u   = matrix(msg.u).H		#[2]

	dxytheta = Dt*matrix([[float(cos(theta)),0],[float(sin(theta)),0],[0,1]])*u
	xytheta[0] = xytheta[0]+dxytheta[1]
	xytheta[1] = xytheta[1]+dxytheta[0]
	xytheta[2] = xytheta[2]+dxytheta[2]
	
	#%control law ends here

	if xytheta[2]>2*pi:
		xytheta[2] = xytheta[2]-2*pi
	elif xytheta[2]<0:
		xytheta[2] = xytheta[2]+2*pi
	X0[1]	= xytheta[0]
	X0[0]	= xytheta[1]
	theta 	= xytheta[2]

	#return the Xhat and X0
	retMsg = positionSim_t()

	retMsg.X0   = X0 #make X0 three dimensional to include angle information
	retMsg.theta = theta#Should be changed with 3 dimensional X0 containing theta

	lcm.publish("conReturn",retMsg.encode())



lcm = lcm.LCM()
subscription = lcm.subscribe("controlCommand", conHandler)  

print "Robot_model.py is ready to rock and roll"



try:
    while True:
        lcm.handle()
except KeyboardInterrupt:
    pass
