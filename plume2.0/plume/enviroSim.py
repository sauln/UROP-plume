import sys
import os

from numpy import linalg as LA
import numpy as np

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cPickle

import lcm
from senlcm import *

import plumeClass
reload(plumeClass)
import flowField
reload(flowField)
plotTrue = False
movieTrue = True

from constants import r

""" 
	going to rearrange the control and estimation part so that
	the gradient and divergence calculations are right in front,
	and more apart of the control than apart of the environment

"""


#print sys.argv[1]

#fileName = sys.argv[1]
fileName =  'models/nagheeby/sparse_5000' #'icra2015sim/love_5000'
print "load plume data from file %s"% fileName

""" 										  """
""" Some constants that we use throughout     """
""" 										  """

plum = plumeClass.plumeEtAl(None, True, fileName )

sc = None
dummyMsg = positionSim_t()
flow = flowField.flowField(plum.param.flow)

fileNumber = 1



print "r = %s"%r
r = 1.0


n1 = (-37.3 * r + 20.54)/r
#or
n2 = (-.3364 *r + 1.1182)/(r*r)

print "norm of %s or %s"%(n1, n2)

n = 1.0

norm = (n/plum.param.den)



print "new r = %s"%r

""" 										  """




"""Create an animation """






from pyqtgraph.Qt import QtGui, QtCore
import numpy as np



class movieData():
	def __init__(self):
		self.heat = []
		
		self.ext = []
		self.xx = []
		self.yy = []
		self.T = []
		self.c = []

	def add(self, Hmasked, extent, rx, ry, T,c):
		self.heat.append(Hmasked)
		self.ext.append(extent)
		self.xx.append(rx)
		self.yy.append(ry)
		self.T.append(T)
		self.c.append(c)



movDat = movieData()

def saveplot(fileName, t, r, norm):
	global fig

	f = "../plots/simPlots/%s_%s_%s_%s.png" %(fileName, t, r, norm)
	print "SAVING PLOT as %s"%f
	savefig(f, bbox_inches='tight')





"""

if plotTrue:
	print "initiate plot"
	plt.close("all")
	fig = plt.figure(figsize=(11,6))
	plt.clf()
	#show()

	#fig.ax = fig.add_subplot(131, aspect='equal')
	#fig.ax2 = fig.add_subplot(132)
	#fig.ax3 = fig.add_subplot(133)
	fig.ax  = plt.subplot2grid((2,3), (0,0), rowspan= 2)
	fig.ax2 = plt.subplot2grid((2,3), (0,1))
	fig.ax3 = plt.subplot2grid((2,3), (1,1))
	fig.ax4 = plt.subplot2grid((2,3), (0,2))
	fig.ax5 = plt.subplot2grid((2,3), (1,2), projection='3d') 


	fig.ax2.set_title('Concentration')
	fig.ax3.set_title('Location')
	fig.ax4.set_title('Divergence')
	fig.ax5.set_title('Gradient')


	fig.ax.axis([0,20,0,30])
	plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)





"""
def saveData():
	global movDat
	
	filename = "runs/r%s_%s.p" %(r, plum.param.den)
	#fileName1 = "data/%s_%s_%s.p" %(t, self.param.dt, self.param.den)
	f = open(filename,'wb')
	cPickle.dump(movDat,f,2)
	f.close()
	print "saving file to %s" %filename
	



def genSomePlots(channel = 1, data = 0):

	fig = figure(1)
	
	#cax = fig.add_axes([0.05, 0.1, 0.03, 0.8])
	norm = colors.Normalize()#vmin=0, vmax=1,clip = False)



	print shape(movDat.heat[-1])
	a = movDat.heat[-1]
	a = max(a.max(axis=1))

	movD = movDat.heat[-1] / float(a)
	print movD

	#im = plt.imshow(movDat.heat[-1], extent = movDat.ext[-1], norm=norm)#, vmin=0, vmax=200)
	im = plt.imshow(movD, extent = movDat.ext[-1]) 

	plt.scatter(movDat.xx, movDat.yy, s = 50,c = 'g', marker='o', zorder = 1)#robot
	#plt.axes([0,30, 0, 20])
	ax = fig.add_subplot(111)#, aspect='equal')

	print im
	fig.colorbar(im) #, ax=ax)#im, ax=ax)

	#ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.5)
	#cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm,
                       #norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
	plt.axis([0,20,0,30])
	plt.axis(aspect= 'equal')
	xlabel("$m$")
	ylabel("$m$")


	fig2 = figure(2)

	t = [m/500.0 for m in movDat.T]

	#c = [c/max(movDat.c) for c in movDat.c]#/max(movDat.c)

	plt.plot(t, movDat.c)
	xlabel('$t$')
	ylabel(r"$\frac{c}{c_{max}}$")
	#ylabel("$\frac{c}{c_{max}}$"
	plt.ylim([0,1])





	show()







	pass

def genMovie(channel = 1, data = 0):
	print "save a shitload of frames and generate a movie from them"


	if channel != 1:
		#that means I was signaled by the simulation to begin making move
		print "I'm going to pickle the movie so I can load it later"
		saveData()
	else:
		print "You're going to have to load some data first"
		#movDat = load('runs/r0.5_5000.0.p')
		
	files = []
	#fig = plt.figure()
	#fig2 = plt.figure()
	#plt.clf()
	figure(1)
	plt
	#plt.title('Simplest default with labels')
	fig.ax #.ax  #3= plt.subplot2grid((2,2), (0,0), rowspan= 2)
	fig.ax.axis([0,20,0,30])
	
	fig2.ax1  = plt.subplot2grid((2,2), (0,0), rowspan= 2)
	fig2.ax1.axis([0,20,0,30])
	

	
	#fig.ax1  = plt.subplot2grid((2,2), (1,1), rowspan=2)
	#fig.ax2  = plt.subplot2grid((2,2), (1,1))


	
	print shape(movDat.T)[0]

	for T, heat, ext, x, y, c, i  in \
		zip(movDat.T, movDat.heat, movDat.ext, movDat.xx, movDat.yy, movDat.c, xrange(shape(movDat.T)[0]) ):
		im = fig.ax.imshow(heat, extent = ext)#, vmin=0, vmax=200)
		fig.ax.scatter(x, y, s = 50,c = 'g', marker='o', zorder = 1)#robot
		fig.ax.set_xlabel("m")
		fig.ax.set_ylabel("m")		
		#fig.ax.set_title("Simulation of '%s'\nT=%s" \
		#	%( fileName, ( T*plum.param.dt)) )
		fig2.ax1.scatter(T-1500, c)
		#fig.ax1.set_title("con: %s"%c)
		#fig.ax2.scatter(x,y)
		fig2.ax1.set_xlabel("time")
		fig2.ax1.set_ylabel("c/c max")
		cax = fig.add_axes([0.05, 0.1, 0.03, 0.8])
		fig.colorbar(im, cax)#im, cax=cax)


		fname = 'frames2/_tmp%03d.png'%i
		print 'saving frame', fname
		fig.savefig(fname)
		files.append(fname)
	print "making movie animation.mpg - this may take a while"
	os.system("mencoder 'mf://frames/_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation%s.mpg"%r)




def makeMovie(channel, data = 0):

	print "I don't know if this is any different than the other one..."
	print "received makeMove finsih message"

	if channel == 1:
		f = open("runs/r0.5_5000.0.p",'rb')
		print type(f)
		mD = cPickle.load(f)
	else:

		global movDat

	plt.close("all")
	fig = plt.figure()
	plt.clf()
	show()
	
	#plt.title('Simplest default with labels')
	fig.ax  = plt.subplot2grid((2,2), (0,0), rowspan= 2)
	fig.ax.axis([0,20,0,30])

	fig.ax1  = plt.subplot2grid((2,2), (0,1))

	for T, heat, ext, x, y, c in zip(movDat.T, movDat.heat, movDat.ext, movDat.xx, movDat.yy, movDat.c):
		im = fig.ax.imshow(heat, extent = ext)#, vmin=0, vmax=200)
		fig.ax.scatter(x, y, s = 50,c = 'g', marker='o', zorder = 1)#robot
		#fig.ax.set_title("Simulation of '%s'\nT=%s" \
		#	%( fileName, ( T*plum.param.dt)) )
		fig.ax1.scatter(T, c)
		#fig.ax1.set_title("con: %s"%c)
		#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
		#fig.colorbar(im, cax)#im, cax=cax)
		plt.draw()

	print "finished movie. how do I save it off now?"




def saveMovie(T, rx, ry,c ):#, div, dx, dy):
	global fig, sc, movDat

	"""this replots everything each time.  I shouldn't have to do that."""
	#plt.clf()
	
	if sc !=None:
		sc.remove()
	
	if T%5 == 0:
		heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[int(ceil(T%500))].\
			ys[::], plum.plumeHist[int(ceil(T%500))].xs[::], bins=50)

		heatmap = np.rot90(heatmap)
		heatmap = np.flipud(heatmap)
		Hmasked = np.ma.masked_where(heatmap==0,heatmap) # Mask pixels with a value of zero
		extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
		#fig.ax.imshow(Hmasked, extent = extent, vmin=0, vmax=200)

		
		movDat.add(Hmasked, extent, rx, ry,T, c)

			#add(Hmasked, extent, rx, ry, T,c):



def updatePlot(T, rx, ry,c ):#, div, dx, dy):
	global fig, sc, movDat

	"""this replots everything each time.  I shouldn't have to do that."""
	#plt.clf()
	
	if sc !=None:
		sc.remove()
	
	if T%10 == 0:
		heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[int(ceil(T%500))].\
			ys[::], plum.plumeHist[int(ceil(T%500))].xs[::], bins=50)

		heatmap = np.rot90(heatmap)
		heatmap = np.flipud(heatmap)
		Hmasked = np.ma.masked_where(heatmap==0,heatmap) 
		extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
		fig.ax.imshow(Hmasked, extent = extent, vmin=0, vmax=200)


	movDat.add(Hmasked, extent, rx, ry,T, c)

	#sc = fig.ax.scatter(plum.plumeHist[int(T%(1/plum.param.dt))].\
	#	ys[::150],plum.plumeHist[int(T%(1/plum.param.dt))].xs[::150])
	fig.ax.scatter(plum.param.xi, plum.param.yi, s = 100,c = 'r', marker='o', zorder = 1)#source
	fig.ax.scatter(rx, ry, s = 50,c = 'g', marker='o', zorder = 1)#robot
	fig.ax.set_title("Simulation of '%s'\nT=%s"%( fileName, ( T*plum.param.dt)) )

	
	fig.ax2.scatter(T, c)
	fig.ax2.set_title("C: %s" %c)
	fig.ax3.scatter(rx, ry)
	fig.ax4.scatter(T, dx)
	#fig.ax5.scatter(T, dy)
	#fig.ax4.scatter(T, div)
	#fig.ax5.scatter(T, dx,dy)
	
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

	#if T == -1:
	#	os._exit(1)# break



	"""now find data should only return the 5 cs and the flow vector """
	c, vx, vy = findData(T, x, y)

	dummyMsg.con = c
	#dummyMsg.DU = (DU_dx0, DU_dy0)
	#dummyMsg.DU_p = (-DU_dy0, DU_dx0)
	dummyMsg.V0 = (vy, vx)
	#dummyMsg.D2U0 = D2U0
	lc.publish("dataReturn", dummyMsg.encode() )
	



def findData(T, x, y):

	print T


	""" This check and load deal should be put inside of the 
		plume class, not the simulator					 """
	if( T*plum.param.dt >= fileNumber and T != 0):
		'''if we are inside of the next file, load it'''
		global fileNumber
		fileNumber = int( T/(1/plum.param.dt) ) + 1 
		print "load next file"
		t = T/(1/plum.param.dt)
		plum.loadData(fileName, fileNumber)
		if plotTrue:
			saveplot(fileName, fileNumber, r , norm)

	
	#flow vector
	vy, vx = flow.getVal(y,x)
	#concentration
	c = []


	print "concentration: %s"%plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y, r)
	c.append( plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y, r)  * norm)
	c.append( plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x+(.2*r), y, r)  * norm)
	c.append( plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x-(.2*r), y, r)  * norm)
	c.append( plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y+(.2*r), r) * norm)
	c.append( plum.plumeHist[int(T%(1/plum.param.dt))].concentration(x, y-(.2*r), r)  * norm)

	#print "from environment: %s" %c

	#yU = self.concentration(y+2*r, x, r)	* norm
	#yD = self.concentration(y-2*r, x, r)	* norm
	#xU = self.concentration(y, x+2*r, r)  * norm
	#xD = self.concentration(y, x-2*r, r)  * norm
	#c = self.concentration(y,x,r)       * norm


	#c =  c * norm
	
	#gradient and divergence
	#DU_dx0, DU_dy0, D2U0 = 						  \
	#	plum.plumeHist[int(T%(1/plum.param.dt))]. \
	#	gradientDivergence(x, y, vx, vy, r, norm)
	
	#print "concentration at (%s, %s): %s"%(x,y,c)
	#print "DU_dx0: %s DU_dy0: %s\nD2U0: %s U0: %s" \
	#	%(DU_dx0, DU_dy0, D2U0, c)

	if movieTrue:
		saveMovie(T, x,y,c[0])


	if plotTrue:
		updatePlot(T, x,y,c[0]) #, D2U0, DU_dx0, DU_dy0)
	#update(x,y)

	return c, vx, vy 
	#return c, DU_dx0, DU_dy0, vx, vy, D2U0



print "initiate lcm"
lc = lcm.LCM()

subs1 = lc.subscribe("envRetrieve", retrieve) 
subs2 = lc.subscribe("finishSim", genSomePlots) #genMovie)
#lcm.publish( "finishSim", msg.encode() )

if plotTrue:
	print "initiate plot"
	plt.close("all")
	fig = plt.figure(figsize=(11,6))
	plt.clf()
	#show()

	#fig.ax = fig.add_subplot(131, aspect='equal')
	#fig.ax2 = fig.add_subplot(132)
	#fig.ax3 = fig.add_subplot(133)
	fig.ax  = plt.subplot2grid((2,3), (0,0), rowspan= 2)
	fig.ax2 = plt.subplot2grid((2,3), (0,1))
	fig.ax3 = plt.subplot2grid((2,3), (1,1))
	fig.ax4 = plt.subplot2grid((2,3), (0,2))
	fig.ax5 = plt.subplot2grid((2,3), (1,2), projection='3d') 


	fig.ax2.set_title('Concentration')
	fig.ax3.set_title('Location')
	fig.ax4.set_title('Divergence')
	fig.ax5.set_title('Gradient')


	fig.ax.axis([0,20,0,30])
	plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)



if __name__ == '__main__':
    

	print "enviroSim is ready"
	try:
		while True:
			lc.handle()
	except KeyboardInterrupt:
		pass

else:
	print "I am just going to hold tight and run whatever function you want"


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

