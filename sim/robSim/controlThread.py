import sys
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA
import time
import matplotlib.pyplot as plt





from mysaturation import mysaturation
import lcm
from senlcm import *



#this is weird - do we have to do it like this...?
from parameters_env import ts,dt,T_thresh,n,k1,k2,k3,k4,threshold,free_speed, T_leader, Dis_thresh,c_leader,c1,c_r,Dt,v_compensate,Xhatdot_max,V0_robot_max

from robot_model_par import kk, l0, d11, m11, d33, m33, l, c0, k11, k21, k31, k41, A, B, zreal, theta, xvector, u, vd, xytheta




eps = sys.float_info.epsilon

mat_contents 	= sio.loadmat('matlab/constants.mat')
dumMsg 			= positionSim_t()
X0 				= matrix(dumMsg.X0).H
Xhat 			= matrix(dumMsg.Xhat).H
u 				= matrix(dumMsg.u).H


# the message handling function.
def conHandler(channel, data):
	
	global u
	global X0
	global Xhat
	#print "message received on channel: %s" % channel
	msg  = positionSim_t.decode(data)
	DU   = matrix(msg.DU).H		#[2]
	U0   = matrix(msg.U0)		#[2]
	DU_p = matrix(msg.DU_p).H	#[2]
	V0   = matrix(msg.V0).H		#[2]
	D2U0 = matrix(msg.D2U0)
	
	
	X0_rec = matrix(msg.X0).H	
	theta = matrix(msg.theta).H
	#calculate the control
	
	dotXhat_1 		= float(  -( (V0.H * DU) + k1 *D2U0 + k2 * U0 ) ) \
							 * divide(DU, (LA.norm(DU)**2))
	dotXhat_2 		= -DU*(DU.H * (Xhat[:, 0] - X0[:,0]) + U0 - threshold)
	dotXhat_2 		= k3*dotXhat_2/ LA.norm(dotXhat_2)
	dotXhat_3 		= c1*DU_p
	dotXhat   		= dotXhat_1+dotXhat_2+dotXhat_3
	Xhatdot 		= dotXhat
	
	#this is the control
	V0_robot 		= -c_r*(X0-Xhat)+v_compensate*Xhatdot;
		
	Xhat 			= Xhat + (Dt * mysaturation(Xhatdot, Xhatdot_max)).H
	X0   			= X0   + (Dt * mysaturation(V0_robot, V0_robot_max)).H 
	Xhat_diff=(Dt * mysaturation(Xhatdot, Xhatdot_max)).H
	X0_diff= (Dt * mysaturation(V0_robot, V0_robot_max)).H

	vd[0] = X0_diff[0]/Dt
	vd[1] = X0_diff[1]/Dt


	Dinv = matrix([ [float(cos(theta)), float(-l0*sin(theta))], \
					[float(sin(theta)), float(l0*cos(theta))]])
	D    = matrix([ [float(cos(theta)),     float(sin(theta))], \
					[float(-sin(theta)/l0), float(cos(theta)/l0)]])


	#%Control law part generates the control input corresponding to u in equation 3 and 4
	F    = LA.inv(B)  *  (-A*u-c0*(u-D*vd))
	du   = (A*u+B*F)*Dt
	u = du+u

	retMsg = positionSim_t()
	retMsg.u = u
	lcm.publish("controlCommand",retMsg.encode())



#start lcm and subscribe to the environment channel
lcm = lcm.LCM()
subscription = lcm.subscribe("conUpdate", conHandler)  


print "controlThread.py ready to rock and roll"
# indefinitely wait for a message - e
# process received messages with envHandler to process
try:
    while True:
        lcm.handle()
except KeyboardInterrupt:
    pass



