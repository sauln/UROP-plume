
import sys

import lcm
from senlcm import *
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA
import time
import matplotlib.pyplot as plt



from constants import *


dumMsg = positionSim_t()
X0 = matrix(dumMsg.X0).H
Xhat = matrix(dumMsg.Xhat).H
u = matrix(dumMsg.u).H
# the message handling function.

def mysaturation( x, max_val ):
	newx      = x.H
	x         = sqrt(  power(newx[:,0], 2) + power(newx[:,1], 2)  )
	myy       = (x >= max_val ) * max_val    +  (x<max_val) * x;
	newxangle = diag((x+eps)**(-1))*newx;
	y 		  = diag(myy)*newxangle;
	return y

#def gradientDivergence(self, y, x, vx, vy, r, norm):
def gradientDivergence(c, xR, xL, yU, yD, vx, vy):#yU,yD,xR,xL, vx, vy):# y, x, vx, vy, r, norm):


	#print "using gradDiv in observer"
	#print "VY: %s VX: %s\nY: %s X: %s" %(vy, vx, y, x)

	#print "find gradient and Divergence"

	#yU = self.concentration(y+2*r, x, r)	* norm
	#yD = self.concentration(y-2*r, x, r)	* norm
	#xU = self.concentration(y, x+2*r, r)  * norm
	#xD = self.concentration(y, x-2*r, r)  * norm
	#c = self.concentration(y,x,r)       * norm

	#print "yU: %s yD: %s\nxU: %s xD: %s\nc: %s "%(yU, yD, xU, xD, c)
	#print "vx: %s vy: %s"%(vx,vy)

	""" There is absolutely no reason why these 2 should be the same ever"""

	#print "concentrations: %s, %s, %s, %s" %(yU, yD, xR, xL)
	###print vx, vy
	#print r






	DU_dy0 = (vy>=0)*(yU-c)/(r)+(vy<0)*(c-yD)/(r)
  	DU_dx0 = (vx>=0)*(xR-c)/(r)+(vx<0)*(c-xL)/(r)

	D2U0 = (yU+yD-2*c)/r**2+(xR+xL-2*c)/r**2
	return DU_dx0, DU_dy0, D2U0




def conHandler(channel, data):
	
	global u
	global X0
	global Xhat
	print "message received on channel: %s" % channel
	msg  = positionSim_t.decode(data)

	c = msg.con
	V0   = matrix(msg.V0).H		#[2]

	#for each in c:
	#	print "concentrations: %s" %each
	#print V0

	DU_dx0, DU_dy0, D2U0 = gradientDivergence(c[0], c[1],c[2],c[3],c[4], V0[0], V0[1])
	
	DU = asmatrix([float(DU_dx0), float(DU_dy0)]).H
	DU_p = asmatrix([float(-DU_dy0), float(DU_dx0)]).H
	U0 = c[0]
	#print DU
	X0_rec = matrix(msg.X0).H	
	theta = matrix(msg.theta).H

	
	dotXhat_1 		= float(  -( (V0.H * DU) + k1 *D2U0 + k2 * U0 ) ) \
							 * divide(DU, (LA.norm(DU)**2))
	dotXhat_1 = nanCheck(dotXhat_1)
	
	dotXhat_2 		= -DU*(DU.H * (Xhat[:, 0] - X0[:,0]) + U0 - threshold)
	dotXhat_2 = nanCheck(dotXhat_2)
	
	dotXhat_2 		= k3*dotXhat_2/ LA.norm(dotXhat_2)
	dotXhat_2 = nanCheck(dotXhat_2)
	
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


	#Control law part generates the control input corresponding to u in equation 3 and 4



	F    = LA.inv(B)  *  (-A*u-c0*(u-D*vd))
	du   = (A*u+B*F)*Dt
	u = du+u

	retMsg = positionSim_t()
	retMsg.u = u
	lcm.publish("controlReturn",retMsg.encode())




def nanCheck(dotXhat):
	for e in xrange(shape(dotXhat)[0]):
		if isnan(dotXhat[e]):
			print "corrected dotXhat"
			dotXhat[e] = 0	
	return dotXhat



#start lcm and subscribe to the environment channel
lcm = lcm.LCM()
subscription = lcm.subscribe("conUpdate", conHandler)  



# indefinitely wait for a message - e
# process received messages with envHandler to process
try:
    while True:
        lcm.handle()
except KeyboardInterrupt:
    pass



