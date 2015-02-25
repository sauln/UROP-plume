"""

Nathaniel Saul 2014
UROP plume simulation
Field Robotics Laboratory
University of Hawaii



Plume Experiment


"""

import sys
import os
import time

from numpy import *
from pylab import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt








import auxiliary
reload(auxiliary)
import plumeExperimentClass
reload(plumeExperimentClass)


close('all')
if(True):
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
	s = plt.scatter(pty,ptx)
	sc = scatter(12, 26, s = 100,c = 'r', marker='o', zorder = 1)
	#pltAniR.append(r)
	pltAniS.append(s)
	plt.title('A very slow simulation that demonstrates that lcm works')
	plt.draw()

def main(): 
	np.random.seed(1)	
	start = time.clock()
	
	#exampleSim()
	print "begin experiment"
	#plumeExperiment.plumeExpPDF()
	#plum = exampleSimExp()

	#plum = plumeExpPDF()
	plum = plumeIntermitency()

	print "end experiemnt"
	elapsed = (time.clock() - start)
	print "Took %s time long (units?)" % elapsed

	#plum.plotPlume()



#PDF and intermittency are the 2 stats we need to generate


def plumeExpPDF():
	print "Find the probability density function at certain steps down from the plume source"
	#first step is to generate the points
	#second, plot all those points after the simulation runs.
	param = auxiliary.Parameters()
	plum = plumeExperimentClass.plumeExpC(param)

	#plum = auxiliary.setupPlumeExp(100, 75, 'mit', 1, 0.12, 12, 26)
	nearEnd = param.T * 0.8
	check = 0
	for x in linspace(0, param.T, param.steps, endpoint=True):	
		#if x%1 == 0:
		#print x

		plum.tickSoA()
		if(x - nearEnd) > 0 and (x - nearEnd) < 1:
			if check%55 == 0:
				plum.gatherPointsSoA()
				#print "collect"	
			check += 1
		

	plum.plotPDF()
	return plum	



def exampleSimExp():
	plotAt = [3, 5, 7, 9, 11, 13]
	#keep track of count
	#start count at 0
	#when plotAt[count] < x:
	# 	plotPlume()
	#	save plot as titled "flow, plotAt[count]
	#	destroy plot
	#	count +=1
	#	continue
	
	#this when developed (if the plot, save, destroy/continue works) should
	#give me all the plots I need
	#and i'll only have to click a button once...

	param = auxiliary.Parameters()
	plum = plumeExperimentClass.plumeExpC(param)

	begifn = int( plum.param.steps*0.65 )
	end   = int( plum.param.steps*0.85 ) 
	count = 0
	for x in linspace(0, param.T, param.steps, endpoint=True):		
		print x
		count += 1
		plum.tickSoA()
	return plum


def plumeIntermitency():
	"""

	this is the framework for generating a statistic about the spontaneity
	of the plume.  It takes periodic measurements of the concentration at
	specified points along the plume as it progresses, and then plots them

	This can be used to characterize the instantaneous plume.
 
	"""

	param = auxiliary.Parameters()
	plum = plumeExperimentClass.plumeExpC(param)

	cList = []
	#hardcode points - 
	p = (12,20)
	margin = 0.03


	for x in linspace(0, param.T, param.steps, endpoint=True):		
		print x
		plum.tickSoA()

		xp, yp = plum.getPointsSoA()

		c = len([x for x, y in zip(xp,yp) \
			if (x-margin)<p[1]<(x+margin) \
			and (y-margin)<p[0]<(y+margin)])
		print "c: %s" %c
		cList.append(c)
	
	spon = len([x for x in cList if x==0]) / float(len(cList))


	figure(8)
	plt.plot(linspace(0, param.T, param.steps, endpoint=True), cList)
	title("Margin Size: %s Location (%s, %s)\nDensity:%s Sponaneity: %s"%(margin, p[0],p[1], plum.param.den, spon))	
	#plum.plotPlume()
	show()

	
	print "end spontaneity test"

	return plum

if __name__ == '__main__':
    main() 






"""
def plumeExp():	
	print "basic plume experiments - only difference is that it's based on plumeExperimentClass"

	plum = auxiliary.setupPlumeExp(10, 200, 'simple', 1, 0.12, 12, 26)
	for x in xrange(plum.settings.steps):
		plum.tick(x)
	return plum
	"""
"""
def centerLineDistribution():
	stds = []
	plum = auxiliary.setupPlumeExp(10, 200, 'simple', 1, 0.12, 12, 26)

	for x in xrange(10):
		plum.createPuff()
	for x in xrange(200):
		plum.movePuffs()
		if (x%2 == 0):
			stds.append( plum.horizontalDistribution() )
"""
"""
def distributionAboutCenterLine():
	#This is an interesting
	STDS = zeros((100, 50))
	for i in xrange(50):
		STDS[:,i] = horizontalSTD( i )

	for i in xrange(50):
		scatter(xrange(100), STDS[:,i])
	show()
"""
"""
def horizontalSTD(seedv):
	print "calculating horizontal standard deviation"
	stds = []
	plum = auxiliary.setupPlumeExp(10, 200, 'simple', 1, 0.12, 12, 26)

	for x in xrange(10):
		plum.createPuff()
	for x in xrange(200):
		plum.movePuffs()
		if (x%2 == 0):
			stds.append( plum.horizontalDistribution() )
	return stds


def plumePanelPlot():
	plum = setupPlumeExp(10, 200, 'simple', 1, 0.12, 12, 26)
	plots = 5; i = 1

	for x in xrange(plum.settings.steps):
		plum.tick(x)
		if (x%(plum.settings.steps/plots) == 0):
			figure(1)
			print 'plot %s' %x 
			subplot(1, plots, i , aspect='equal')
			subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
			quiver( plum.flow.x[::6,::6], plum.flow.y[::6,::6], \
				plum.flow.vy[::6, ::6], plum.flow.vx[::6, ::6], 
				color = 'r', units='x',pivot=( 'tail' ), 
				linewidths=(2,), edgecolors=('k'), headaxislength=5 )
			for each in plum.puffs:
				xp.append(each.loc[0])
				yp.append(each.loc[1])
			scatter(yp,xp)
			axis('equal')
			scatter(plum.source[1], plum.source[0], s = 100,c = 'r', marker='o')
			i +=1
"""


'''
def soWhatAreWeDoing():
	plum = auxiliary.setupPlumeExp(10, 200, 'simple', 1, 0.12, 12, 26)

	for x in xrange(plum.settings.steps):
		plum.tick(x)
	plum.calculateCenterLine()
	return plum
'''




'''
def plumeShortTest():
	print "run short test"
	
	plum = auxiliary.setupPlumeExp(100, 20, 'mit', 1, 0.12, 12, 26)


	#total time for a good run is 18 seconds
	time = plum.settings.dt *15 * 10

	for i in xrange(15):
		for x in xrange(200):
			plum.createPuff()
		for x in xrange(10):
			plum.kitanidisNaive(plum.settings.dt)
		plum.gatherPoints()
	plum.plotPDF()

	#plum.plotPDFandData()
	
	return plum
'''
