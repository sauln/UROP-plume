import sys
from numpy import linalg as LA
import numpy as np

#from pyqtgraph.Qt import QtGui, QtCore
#import pyqtgraph

from pylab import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import lcm
from senlcm import *

import plumeClass
reload(plumeClass)


''' Setup Plume '''
#T = 5 #seconds 
dt = 0.002
density = 300
flow = 'simple'
std = 1
sourcex = 12; sourcey = 25
plum = plumeClass.setupPlume(density, flow, std, dt, sourcex, sourcey)
fig = plt.figure(1)
rix = 12; riy = 24
pltAniR = []; pltAniS = []

lc = lcm.LCM()


def updatePoints(flow, ptx, pty):
	global rix, riy
	plt.clf()
	ax = fig.add_subplot(111, aspect='equal')
	plt.axis([0,20,0,30])
	axes().set_aspect('equal')
	#Q = plt.quiver( flow.x[:-2:dif,::dif], flow.y[:-2:dif,:-2:dif], \
	#	flow.vy[:-2:dif, :-2:dif], flow.vx[:-2:dif, :-2:dif], 
	#	color = 'k', units='x',pivot=( 'head' ),
	#	linewidths=(0,0), edgecolors=('k'), headaxislength=0.1 )
	r = plt.scatter(rix, riy,  s = 50, c = 'r', marker='o', zorder = 1)
	s = plt.scatter(pty,ptx, zorder = -1)
	pltAniR.append(r)
	pltAniS.append(s)
	plt.title('A very slow simulation that demonstrates that lcm works')
	plt.draw()

def confirmUpdate():
	msg = positionSim_t()
	lc.publish("envUpdateConfirm", msg.encode() )

def tick(channel, data):

	print np.shape(plum.puffs)
	plum.tick2()
	xp, yp = plum.getPoints()
	updatePoints(plum.flow, xp, yp)
	confirmUpdate()

def retrieve(channel, data):
	global rix, riy
	plum.movePuffs()

	normalizingValue = plum.settings.density / 3.0
	dx = 0.2
	dy = dx
	msg  = positionSim_t.decode(data)
	x = msg.X0[1];  y = msg.X0[0]
	rix = y; riy = x
	print "\n"
	print x,y

	c, x1, x2, x3, x4 = cOnP(x, y, dx, normalizingValue)
	vy, vx = plum.flow.getVector(x, y) 
	DU, DU_p, V0, D2U0, U0 = calcConGradDiv(c, x1, x2, x3, x4, vy, vx)

	retMsg = positionSim_t();
	retMsg.DU = DU;
	retMsg.DU_p = DU_p;
	retMsg.V0 = V0;
	retMsg.D2U0 = D2U0;
	retMsg.U0 = U0;
	
	lc.publish("dataReturn", retMsg.encode() );

def cOnP(x,y,dx, norm):
	#normalizingValue = 10.0 # just trying to bring the puff count to a reasonable	
	#value.  it looks like the concentration of like 0.3 is reasonable.
	c = plum.concentration(x,y, dx)/norm
	x1 = plum.concentration(x+dx, y, dx)/norm
	x2 = plum.concentration(x-dx, y, dx)/norm
	x3 = plum.concentration(x, y+dx, dx)/norm
	x4 = plum.concentration(x, y-dx, dx)/norm

	print "            x1 Up: %2.2f \nx3 Left: %2.2f c Center: %2.2f x4 Right: %2.2f \n            x2 Down: %2.2f" %(x1, x3, c, x4, x2)

	return c, x1, x2, x3, x4	

def calcConGradDiv(c, x1, x2, x3, x4, vy, vx):
	#input c, x1, x2, x3, x4, vy, vx
	#output DU, DU_p, V0, D2U0, U0

	dx = 0.2#this is given from the original environment thread
	dy = dx

	DU_dy0 = (vy>=0)*(x3-c)/(dx)+(vy<0)*(c-x4)/(dx)
  	DU_dx0 = (vx>=0)*(x1-c)/(dx)+(vx<0)*(c-x2)/(dx)
	DU = (DU_dx0, DU_dy0)
	DU_p = (-DU_dy0, DU_dx0)
	DU_p = DU_p/LA.norm(DU_p + np.finfo(float).eps)

	V0 = (vx, vy)
	D2U0 = (x3+x4-2*c)/dx**2+(x1+x2-2*c)/dy**2
	U0 = c

	print "DU: %s DU_p: %s\nV0: %s D2U0: %s U0: %s" \
		%(DU, DU_p, V0, D2U0, U0)

	return DU, DU_p, V0, D2U0, U0

def main( ):
	global fig, lc
	''' Setup plot '''
	dif = 12
	plt.close('all')
	
	ax = fig.add_subplot(111, aspect='equal')
	plt.axis([0,20,0,30])
	plt.show()

	subs2 = lc.subscribe("envUpdate", tick) 
	subs1 = lc.subscribe("envRetrieve", retrieve) 

	print "env model is respondent"
	try:
		while True:
		    lc.handle()
	except KeyboardInterrupt:
		pass


if __name__ == '__main__':
    main() 




