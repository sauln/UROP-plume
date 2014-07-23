
#necessary inputs are      Xhat, Xhatdot, X0, DU, DU_p, V0, D2U0, U0
#necessary returns are     Xhat, X0, Xhatdot

import sys

import lcm
from senlcm import *
from math import *
from numpy import *
import scipy.io as sio
import scipy.linalg as LA
import time
import matplotlib.pyplot as plt

eps = sys.float_info.epsilon


plotTrue = True



mat_contents = sio.loadmat('constants.mat')
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


#surface vessel control
A				= matrix(mat_contents['A_kin'])
B				= matrix(mat_contents['B_kin'])
l0				= float(mat_contents['l0'])
xytheta			= matrix(mat_contents['xytheta'])
u				= matrix(mat_contents['u_kin'])
c0				= float(mat_contents['c01'])
l				= float(mat_contents['l_kin'])


kk 			= 1.

print "We are trying to find: %s" %threshold
threshold = 0.5
print "Now we are trying to find: %s"% threshold
if plotTrue:
	print "initiate plot"
	fig = plt.figure(1)
	plt.clf()
	plt.show()

	fig.ax = fig.add_subplot(121, aspect='equal')
	fig.ax2 = fig.add_subplot(122, aspect='equal')
	scale = .03
	fig.ax.axis([-scale,scale,-scale,scale])
	#fig.ax2.axis([0,20,0,30])
	#fig.ax2.sc = None

def mysaturation( x, max_val ):
	newx      = x.H
	x         = sqrt(  power(newx[:,0], 2) + power(newx[:,1], 2)  )
	myy       = (x >= max_val ) * max_val    +  (x<max_val) * x;
	newxangle = diag((x+eps)**(-1))*newx;
	y 		  = diag(myy)*newxangle;
	return y

#def surfaceVesselControl(V0_robot, X0, X0_diff):
def control_SurfaceVessel(X0, Xhat, Xhatdot): #X0_diff):
	global u, xytheta

	V0_robot 		= -c_r*(X0-Xhat)+v_compensate*Xhatdot;
	X0_diff   		= (Dt * mysaturation(V0_robot, V0_robot_max)).H	

	theta = xytheta[2]
	vd = zeros(shape=(2,1))
	vd[0] = X0_diff[0]/Dt
	vd[1] = X0_diff[1]/Dt
	Dinv = matrix([ [float(cos(theta)), float(-l0*sin(theta))], \
					[float(sin(theta)), float(l0*cos(theta))]])
	D    = matrix([ [float(cos(theta)),     float(sin(theta))], \
					[float(-sin(theta)/l0), float(cos(theta)/l0)]])
	F    = LA.inv(B)  *  (-A*u-c0*(u-D*vd))
	du   = (A*u+B*F)*Dt
	dxytheta = Dt*matrix([[float(cos(theta)),0],[float(sin(theta)),0],[0,1]])*u
	u = du+u
	xytheta[0] = xytheta[0]+dxytheta[1]
	xytheta[1] = xytheta[1]+dxytheta[0]
	xytheta[2] = xytheta[2]+dxytheta[2]
	if xytheta[2]>2*pi:
		xytheta[2] = xytheta[2]-2*pi
	elif xytheta[2]<0:
		xytheta[2] = xytheta[2]+2*pi
	X0[0] = xytheta[1]
	X0[1] = xytheta[0]
	return X0

def control_SingleIntegrator(X0, Xhat, Xhatdot): #,X0_diff):
	V0_robot 		= -c_r*(X0-Xhat)+v_compensate*Xhatdot;
	X0_diff   		= (Dt * mysaturation(V0_robot, V0_robot_max)).H	
	X0   			= X0   + X0_diff
	#print "X0_diff: %s" %X0_diff
	return X0


def updatePlot(Xhat_diff, Xhat):
	global fig
	#print "update the plot"
	#print "Xhat_diff: %s" %Xhat_diff
	#print "Xhat: %s" %Xhat

	#if fig.ax2.sc != None:
	#	fig.ax2.sc.remove()

	#sc = fig.ax.scatter(Xhat_diff[0], Xhat_diff[1])
	#sc = fig.ax.arrow(0.0, 0.0, Xhat_diff[0], Xhat_diff[1])
	sc = fig.ax.scatter(Xhat_diff[0], Xhat_diff[1])
	cs = fig.ax2.scatter(Xhat[0], Xhat[1])
	fig.ax.set_title('Xhat_diff:\n(%s, %s)'%(Xhat_diff[0], Xhat_diff[1]) )
	fig.ax2.set_title('Xhat: (%s, %s)'%(Xhat[0], Xhat[1]))
	plt.draw()

	


def observer(X0, Xhat, Xhatdot, V0, DU, DU_p, U0, D2U0):
	#this currently cannot handle when some of the values are zerod
	#we expect this to happen a lot

	print "V0: %s X0: %s Xhat: %s Xhatdot: %s"%(V0[1], X0[1], Xhat[1], Xhatdot[1])


	dotXhat_1 		= float(  -( (V0.H * DU) + k1 *D2U0 + k2 * U0 ) ) \
							 * divide(DU, (LA.norm(DU)**2))

	#print shape(dotXhat_1)
	for e in xrange(shape(dotXhat_1)[0]):
		if isnan(dotXhat_1[e]):
			print "corrected dotXhat_1"
			dotXhat_1[e] = 0
				
	dotXhat_2 		= -DU*(DU.H * (Xhat[:, 0] - X0[:,0]) + U0 - threshold)
	for e in xrange(shape(dotXhat_2)[0]):
		if isnan(dotXhat_2[e]):
			print "corrected dotXhat_2"
			dotXhat_2[e] = 0

	dotXhat_2 		= k3*dotXhat_2/ LA.norm(dotXhat_2)
	for e in xrange(shape(dotXhat_2)[0]):
		if isnan(dotXhat_2[e]):
			print "corrected dotXhat_2.2"
			dotXhat_2[e] = 0

	dotXhat_3 		= c1*DU_p
	dotXhat   		= dotXhat_1+dotXhat_2+dotXhat_3
	Xhatdot 		= dotXhat

	#this is the control
	Xhat_diff 		= (Dt * mysaturation(Xhatdot, Xhatdot_max)).H
	Xhat 			= Xhat + Xhat_diff
	
	print "Xhat_diff: %s: "%Xhat_diff
	if plotTrue:
		updatePlot(Xhat_diff, Xhat)
	return Xhat, Xhatdot

# the message handling function.
def conHandler(channel, data):
	#print "message received on channel: %s" % channel
	msg  = positionSim_t.decode(data)
	DU   = matrix(msg.DU).H		#[2]
	U0   = matrix(msg.U0)		#[2]
	DU_p = matrix(msg.DU_p).H	#[2]
	V0   = matrix(msg.V0).H		#[2]
	D2U0 = matrix(msg.D2U0)

	Xhatdot = matrix(msg.Xhatdot).H
	X0 = matrix(msg.X0).H	
	Xhat = matrix(msg.Xhat).H

	#calculate the control
	Xhat, Xhatdot = observer(X0, Xhat, Xhatdot, V0, DU, DU_p, U0, D2U0)

	#X0 = control_SurfaceVessel(X0, Xhat, Xhatdot)
	X0 = control_SingleIntegrator(X0, Xhat, Xhatdot)


	#return the Xhat and X0
	retMsg = positionSim_t()
	retMsg.Xhat = Xhat
	retMsg.X0   = X0
	retMsg.Xhatdot = Xhatdot

	lcm.publish("conReturn",retMsg.encode())



#start lcm and subscribe to the environment channel
lcm = lcm.LCM()
subscription = lcm.subscribe("conUpdate", conHandler)  


print "controlThread_Kin.py ready for take off"
# indefinitely wait for a message - e
# process received messages with envHandler to process
try:
    while True:
        lcm.handle()
except KeyboardInterrupt:
    pass




