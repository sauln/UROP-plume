import sys
from numpy import linalg as LA
import numpy as np


import lcm
from senlcm import *
import auxiliary

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph


import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


''' Setup Plume '''
T = 5 #seconds 
dt = 0.002
steps = int( np.ceil( T/dt ) )
density = 0.2
flow = 'simple'
std = 1
xi = 12; yi = 25

plum = auxiliary.setupPlumeExp(density, steps, flow, std, dt, xi, yi)

''' Setup plot '''
dif = 12
plt.close('all')
fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
plt.axis([0,20,0,30])
plt.show()

def updatePoints(flow, ptx, pty):
	global xi, yi
	#print "updatePoints"
	plt.clf()
	ax = fig.add_subplot(111, aspect='equal')
	plt.axis([0,20,0,30])
	#Q = plt.quiver( flow.x[:-2:dif,::dif], flow.y[:-2:dif,:-2:dif], \
	#	flow.vy[:-2:dif, :-2:dif], flow.vx[:-2:dif, :-2:dif], 
	#	color = 'k', units='x',pivot=( 'head' ),
	#	linewidths=(0,0), edgecolors=('k'), headaxislength=0.1 )
	r = plt.scatter(xi, yi,  s = 100,c = 'r', marker='o', zorder=1)
	s = plt.scatter(pty,ptx, zorder = -1)

		
	plt.title('A very slow simulation that demonstrates that lcm works')
	plt.draw()



def confirmFinish():
	msg = positionSim_t()
	lcm.publish("envUpdateConfirm", msg.encode() )



def tick(channel, data):
	plum.tick2()
	xp, yp = plum.getPoints()
	#print xp
	updatePoints(plum.flow, xp, yp)
	confirmFinish()


def retrieve(channel, data):
	global xi, yi
	#print "retrieve"
	#global trap

	plum.movePuffs()

	
	dx = 1
	dy = dx
	msg  = positionSim_t.decode(data)
	x = msg.X0[1];  y = msg.X0[0]
	xi = x
	yi = y
	print x,y

	#print x,y
	normalizingValue = 10.0 # just trying to bring the puff count to a reasonable	
	#value.  it looks like the concentration of like 0.3 is reasonable.
	c = plum.concentration(x,y, dx)/normalizingValue
	x1 = plum.concentration(x+dx, y, dx)/normalizingValue
	x2 = plum.concentration(x-dx, y, dx)/normalizingValue
	x3 = plum.concentration(x, y-dx, dx)/normalizingValue
	x4 = plum.concentration(x, y-dx, dx)/normalizingValue
	vy, vx = plum.flow.getVector(x, y) 
	
	#print "return the concentration at (%s, %s)" %(x, y)
	#print x1, x2, x3, x4, vx, vy, c

	DU_dy0 = (vy>=0)*(x1-c)/(dx)+(vy<0)*(c-x2)/(dx)
  	DU_dx0 = (vx>=0)*(x3-c)/(dx)+(vx<0)*(c-x4)/(dx)
	DU = (DU_dx0, DU_dy0)
	DU_p = (-DU_dy0, DU_dx0)
	DU_p = DU_p/LA.norm(DU_p+ np.finfo(float).eps)
	V0 = (vx, vy)
	D2U0 = (x3+x4-2*c)/dx**2+(x1+x2-2*c)/dy**2
	U0 = c

	print "DU: %s  DU_p: %s\nV0: %s   D2U0: %s\nU0: %s"%(DU, DU_p, V0, D2U0, U0)

	retMsg = positionSim_t();
	retMsg.DU = DU;
	retMsg.DU_p = DU_p;
	retMsg.V0 = V0;
	retMsg.D2U0 = D2U0;
	retMsg.U0 = U0;
	
	lcm.publish("dataReturn", retMsg.encode() );


	
#lc.subscribe('envUpdate', aggregator);
#lc.subscribe('finishSim', aggregator);
#lc.subscribe('envRetrieve', aggregator);

lcm = lcm.LCM()
subs2 = lcm.subscribe("envUpdate", tick) 
subs1 = lcm.subscribe("envRetrieve", retrieve) 

#sub = lcm.subscribe("plotIt", plotIt ) 

print "env model is respondent"
try:
    while True:
        lcm.handle()
except KeyboardInterrupt:
    pass












'''

class plotter(QtGui.QMainWindow):

	def __init__(self, quiverField):
		super(plotter, self).__init__()
		self.quiver = quiverField

	def updateP(self, newX, newY):
		pass
		#now we are going to plot the new points



'''



