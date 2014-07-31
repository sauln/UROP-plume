import sys
import plumeClass
reload(plumeClass)



from pylab import *
import matplotlib.pyplot as plt


"""
#here we want to plot some intermittency graphs that look good
#here we want to plot some instantaneous distribution and time-averaged distribution

source is at 10, 26

lets go 4 units down and 8 units down

(10, 22) and (10, 18)

here we are going to sample the plume every 5 time steps
we will sample the concentration with varying radius

we will start at time 4 and go to tme 8

"""


fileName = "symetric/dense"
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


plotHex = False
plotSpont = False
plotPDE = True


if plotPDE:
	fNum = 8
	dist = [22, 19, 16]
	plum.loadData(fileName, fNum)
	r = 0.05
	sparse = 1
	num = 20 / (2*r)
	steps = np.linspace(0, 20, num, endpoint = True)
	cL = []
	cL.append(list())
	cL.append(list())
	cL.append(list())

	for d, l in zip(dist, cL):
		for s in steps:
			c = plum.plumeHist[-1].concentration(s, d, r, sparse) 
			l.append(c)
		lb = '%s' %d
		plt.plot(l, label = lb)
	plt.legend()
	savefig('../plots/distrobution/%sand%sand%s.png'%(fNum, dist, size(steps)))

	show()



if plotSpont:
	start = int(4/0.002)
	end   = int(5/0.002)
	step = 10
	fileNumber = 1
	
	""" We need to figure out an equation to relate these"""
	r = 0.015
	norm = (5.0/plum.param.den)
	sparse = 1

	cL = []


	print "start %s, stop %s" %(start, end)

	fig = plt.figure()
	for T in xrange(start, end, step):

		""" This check and load deal should be put inside of the 
			plume class, not the simulator					 """
		if( T*plum.param.dt >= fileNumber and T != 0):
			'''if we are inside of the next file, load it'''
			global fileNumber
			fileNumber = int( T/(1/plum.param.dt) ) + 1 
			print "load next file"
			t = T/(1/plum.param.dt)
			plum.loadData(fileName, fileNumber)
		
		#plt.scatter(plum.plumeHist[int(T%(1/plum.param.dt))].ys, \
		#	plum.plumeHist[int(T%(1/plum.param.dt))].xs)
		#show()

		c = plum.plumeHist[int(T%(1/plum.param.dt))].\
			concentration(10, 22, r, sparse) 
		#c =  c * norm
		#print c
		cL.append(c)



	mean = np.mean(cL)

	plt.plot(cL)
	#plt.plot(([0,0]), ([mean,mean]), 'r')

	plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	plt.title("mean: %s" %mean)
	savefig('../plots/newSpont/%sand%s.png'%(r, sparse))

	show()








if plotHex:

	start = 7
	end = 8
	h = []
	xe = []
	ye = []
	for t in range(start,end):
		print t
		plum.loadData(fileName, int(t)+1)
		heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
				ys[::], plum.plumeHist[-1].xs[::], bins=50)

		heatmap = np.rot90(heatmap)
		heatmap = np.flipud(heatmap)
		Hmasked = np.ma.masked_where(heatmap==0,heatmap) # Mask pixels with a value of zero
	 
		#print "max: %s min: %s" %(heatmap), heatmap))
		print shape(heatmap)
		h.append(Hmasked)
		xe.append(xedges)
		ye.append(yedges)


	total = end - start
	if total == 1:
		si = 1
	
	fig, axes = plt.subplots(nrows=si, ncols = si)
	plt.title("Heatmap of plume from simulation %s" %fileName)

	data = h
	#for dat, ax, x,y, t in zip(data, axes.flat, xe, ye, range(start,end) ):
	for dat,  x,y, t in zip(data,  xe, ye, range(start,end) ):
		# The vmin and vmax arguments specify the color limits
		extent = [x[0], x[-1], y[-1], y[0]]
		#im = axes.imshow(dat, extent = extent, vmin=0, vmax=200)
		im = axes.imshow(dat, extent = extent, vmin = 0, vmax = 5000)
		#plt.pcolormesh(x,y,dat)
		axes.axis([0,20,0,30])
		#redraw()
		axes.set_title('time %s'%(t+1))

	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(im, cax=cax)

	show()




