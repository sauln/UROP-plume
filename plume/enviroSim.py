import sys
import os

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
plotTrue = True


def saveplot(fileName, t, r, norm):
	global fig
	print "SAVING PLOT"

	f = "../simPlots/%s_%s_%s_%s.png" %(fileName, t, r, norm)
	savefig(f, bbox_inches='tight')


def updatePlot(T, rx, ry,c , div, dx, dy):
	global fig, sc

	"""this replots everything each time.  I shouldn't have to do that."""
	#plt.clf()
	
	if sc !=None:
		sc.remove()
	
	if T%100 ==0:
		heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
			ys[::], plum.plumeHist[-1].xs[::], bins=50)

		heatmap = np.rot90(heatmap)
		heatmap = np.flipud(heatmap)
		Hmasked = np.ma.masked_where(heatmap==0,heatmap) # Mask pixels with a value of zero
		extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
		fig.ax.imshow(Hmasked, extent = extent, vmin=0, vmax=200)




	#sc = fig.ax.scatter(plum.plumeHist[int(T%(1/plum.param.dt))].\
	#	ys[::150],plum.plumeHist[int(T%(1/plum.param.dt))].xs[::150])
	fig.ax.scatter(12, 26, s = 100,c = 'r', marker='o', zorder = 1)#source
	fig.ax.scatter(rx, ry, s = 50,c = 'g', marker='o', zorder = 1)#robot
	fig.ax.set_title("Simulation of '%s'\nT=%s"%( fileName, ( T*plum.param.dt)) )

	
	fig.ax2.scatter(T, c)
	fig.ax3.scatter(rx, ry)
	#fig.ax4.scatter(T, dx)
	#fig.ax5.scatter(T, dy)
	#fig.ax4.scatter(T, div)
	#fig.ax5.scatter(dx,dy)
	
	plt.draw()

def confirmUpdate():
	#this is not needed anymore
	global dummyMsg
	lc.publish("envUpdateConfirm", dummyMsg.encode() )

def retrieve(channel, data):
	global plum
	global dummyMsg

	msg = positionSim_t.decode(data)
	x = msg.X0[0];  y = msg.X0[1]
	T = msg.T

	if T == -1:
		os._exit(1)# break
	c, DU_dx0, DU_dy0, vx, vy, D2U0 = findData(T, x, y)

	dummyMsg.U0 = c
	dummyMsg.DU = (DU_dx0, DU_dy0)
	dummyMsg.DU_p = (-DU_dy0, DU_dx0)
	dummyMsg.V0 = (vy, vx)
	dummyMsg.D2U0 = D2U0
	lc.publish("dataReturn", dummyMsg.encode() )
	



def findData(T, x, y):

	print T

	

	if( T%(1/plum.param.dt) == 0 and T != 0):
		print "load next file"
		t = T/(1/plum.param.dt)
		plum.loadData(fileName, int(t)+1)
		if plotTrue:
			saveplot(fileName, int(t)+1, r , norm)

	
	#flow vector
	vy, vx = flow.getVal(y,x)
	#concentration
	c = plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y, r) 
	c =  c * norm
	
	#gradient and divergence
	DU_dx0, DU_dy0, D2U0 = 						  \
		plum.plumeHist[int(T%(1/plum.param.dt))]. \
		gradientDivergence(x, y, vx, vy, r, norm)
	
	#print "concentration at (%s, %s): %s"%(x,y,c)
	#print "DU_dx0: %s DU_dy0: %s\nD2U0: %s U0: %s" \
	#	%(DU_dx0, DU_dy0, D2U0, c)

	if plotTrue:
		updatePlot(T, x,y,c, D2U0, DU_dx0, DU_dy0)

	return c, DU_dx0, DU_dy0, vx, vy, D2U0



print "initiate lcm"
lc = lcm.LCM()

subs1 = lc.subscribe("envRetrieve", retrieve) 


if plotTrue:
	print "initiate plot"
	plt.close("all")
	fig = plt.figure(figsize=(11,6))
	plt.clf()
	show()

	#fig.ax = fig.add_subplot(131, aspect='equal')
	#fig.ax2 = fig.add_subplot(132)
	#fig.ax3 = fig.add_subplot(133)
	fig.ax  = plt.subplot2grid((2,3), (0,0), rowspan= 2)
	fig.ax2 = plt.subplot2grid((2,3), (0,1))
	fig.ax3 = plt.subplot2grid((2,3), (1,1))
	fig.ax4 = plt.subplot2grid((2,3), (0,2))
	fig.ax5 = plt.subplot2grid((2,3), (1,2)) 


	fig.ax2.set_title('Concentration')
	fig.ax3.set_title('Location')
	fig.ax4.set_title('Divergence')
	fig.ax5.set_title('Gradient')


	fig.ax.axis([0,20,0,30])
	plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

print sys.argv[1]

fileName = sys.argv[1]






print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )

""" 										  """
""" Some constants that we use throughout     """
""" 										  """
sc = None
dummyMsg = positionSim_t()
flow = flowField.flowField(plum.param.flow)
r = 0.5
norm = (5.0/plum.param.den)
""" 										  """


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
