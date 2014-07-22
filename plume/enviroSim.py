import sys


from numpy import linalg as LA
import numpy as np

from pylab import *
import matplotlib.pyplot as plt

import cPickle

import lcm
from senlcm import *

import plumeClass
reload(plumeClass)
import flowField
reload(flowField)

fileName = "plumeHistMIT"

sc = None

dummyMsg = positionSim_t()


def tick(channel, data):
	#is not necessary anymore

	#updatePlot(12,24)
	pass

def updatePlot(T, rx, ry,c ):
	global fig, sc

	"""this replots everything each time.  I shouldn't have to do that."""
	#plt.clf()
	
	if sc !=None:
		sc.remove()
	

	sc = fig.ax.scatter(plum.plumeHist[int(T%(1/plum.param.dt))].ys[::150],plum.plumeHist[int(T%(1/plum.param.dt))].xs[::150])
	fig.ax.scatter(12, 26, s = 100,c = 'r', marker='o', zorder = 1)#source
	fig.ax.scatter(rx, ry, s = 50,c = 'g', marker='o', zorder = 1)#robot
	fig.ax.set_title('SIMULATION, T=%s'%(T*plum.param.dt) )

	
	fig.ax2.scatter(T, c)
	fig.ax2.set_title('Concentration by time at point\n(%4.3f, %4.3f) c: %4.3f ' % (rx, ry, c))
	plt.draw()

	




def confirmUpdate():
	#this is not needed anymore
	global dummyMsg
	lc.publish("envUpdateConfirm", dummyMsg.encode() )





def retrieve(channel, data):
	global plum

	r = 0.2
	norm = (50.0/plum.param.den)
	print "NORM: %s" %norm

	msg = positionSim_t.decode(data)
	x = msg.X0[0];  y = msg.X0[1]
	T = msg.T
	#c = findConcentration(x,y,plume.plumeHist[index].xs, plume.plumeHist[index].ys)
	print T
	if( T%(1/plum.param.dt) == 0 and T != 0):
		print "load next file"
		t = T/(1/plum.param.dt)
		plum.loadData(fileName, int(t)+1)

	
	#flow vector
	vy, vx = flow.getVal(y,x)
	#concentration
	c = plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y, r) 
	c =  c * norm
	print "concentration at (%s, %s): %s"%(x,y,c)

	#gradient and divergence
	DU_dx0, DU_dy0, D2U0 = \
		plum.plumeHist[int(T%(1/plum.param.dt))].\
		gradientDivergence(x, y, vx, vy, r, norm)
	

	print "DU_dx0: %s DU_dy0: %s\nD2U0: %s U0: %s" \
		%(DU_dx0, DU_dy0, D2U0, c)



	global dummyMsg
	dummyMsg.U0 = c
	dummyMsg.DU = (DU_dx0, DU_dy0)
	dummyMsg.DU_p = (-DU_dy0, DU_dx0)
	dummyMsg.V0 = (vy, vx)
	dummyMsg.D2U0 = D2U0
	lc.publish("dataReturn", dummyMsg.encode() )
	
	updatePlot(T, x,y,c)
	#confirmUpdate()
	#lc.publish("dataReturn", retMsg.encode() );



index = 0

print "initiate lcm"
lc = lcm.LCM()
subs2 = lc.subscribe("envUpdate", tick) 
subs1 = lc.subscribe("envRetrieve", retrieve) 

print "initiate plot"
fig = plt.figure(1)
plt.clf()
show()

fig.ax = fig.add_subplot(121, aspect='equal')
fig.ax2 = fig.add_subplot(122)#, aspect='equal')

fig.ax.axis([0,20,0,30])
#fig.ax2.axis([0,20,0,30])


print "load plume data from file %s"% fileName

plum = plumeClass.plumeEtAl(None, True, fileName )

flow = flowField.flowField(plum.param.flow)


#plume = cPickle.load( open(fileName, "rb" ) )



print "enviroSim is ready"
try:
	while True:
	    lc.handle()
except KeyboardInterrupt:
	pass




"""



def retrieve():
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




























"""

