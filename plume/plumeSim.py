"""

Nathaniel Saul 2014
UROP plume simulation
Field Robotics Laboratory
University of Hawaii
 
Main executable for the plume simulator 	




"""

import cPickle
import sys
import os
import time

from numpy import *
from pylab import *

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



#import plumeExperimentClass
#reload(plumeExperimentClass)
#import plumeExperiment
#reload(plumeExperiment)

import cPickle as pickle



import plumeClass
reload(plumeClass)
import auxiliary
reload(auxiliary)


close('all')
plotMe = True

if(plotMe):
	fig = plt.figure(1)
	pltAniR = []; pltAniS = []
	show()



def updatePoints(ptx, pty):
	plt.clf()
	ax = fig.add_subplot(111, aspect='equal')
	plt.axis([0,20,0,30])
	#Q = plt.quiver( flow.x[:-2:dif,::dif], flow.y[:-2:dif,:-2:dif], \
	#	flow.vy[:-2:dif, :-2:dif], flow.vx[:-2:dif, :-2:dif], 
	#	color = 'k', units='x',pivot=( 'head' ),
	#	linewidths=(0,0), edgecolors=('k'), headaxislength=0.1 )
	#r = plt.scatter(rix, riy,  s = 50, c = 'r', marker='o', zorder = 1)

	#h = hist2d(pty, ptx, bins=4)
	#h = hexbin(pty,ptx, bins = 'log', cmap=plt.cm.hsv)
	#axis('equal')
	#colorbar()
	s = plt.scatter(pty,ptx)
	sc = scatter(12, 26, s = 100,c = 'r', marker='o', zorder = 1)
	#pltAniR.append(r)
	#pltAniS.append(s)
	plt.title('A very slow simulation that demonstrates that lcm works')
	plt.draw()


def main(): 
	np.random.seed(1)	
	start = time.clock()
	
	#exampleSim()
	print "begin experiment"
	plum = exampleSim()
	#plum = generateData()
	print "end experiemnt"
	elapsed = (time.clock() - start)
	print "Took %s time long (units?)" % elapsed
	#plum.plumeEtAl.printAll()
	#plum.plotPlume()




	#plumeHist = cPickle.load( open("plumeHist.p", "rb" ) )
	#plumeHist.makeMovie(fig)


	"""This works"""


def exampleSim():
	param = auxiliary.Parameters()
	plum = plumeClass.plume(param, "none", False)

	for x in xrange(int(param.T/param.dt)):
	#for x in linspace(0, param.T, param.steps, endpoint=True):
		#print x
		plum.tickSoA(x)
		if plotMe:
			xp, yp = plum.getPointsSoA()
			updatePoints(xp, yp)
	return plum


if __name__ == '__main__':
    main() 





"""
def plumeSim():
	print "run main simulation"

	dt = 0.002
	density = 100
	flow = 'simple'
	std = 1
	xi = 12; yi = 26
	plum = plumeClass.setupPlume(density, flow, std, dt, xi, yi)


	den = 890; ticks = 1/dt

	T = 18 #seconds 
	steps = int( ceil( T/dt ) )
	ratio = den/ticks

	print "There are %s puffs per tick" %ratio

	puffQueue = 0
	for i in xrange(steps):
		puffQueue += ratio
		if puffQueue > 1:
			for x in xrange(int(floor(puffQueue))):
				puffQueue -= 1
				plum.createPuff()
		plum.movePuffs()
		xp, yp = plum.getPoints()
		updatePoints(xp, yp)


	plum.plotPlume()

"""

