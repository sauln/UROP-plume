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


fileName = "symetric/sparse"
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


#fig1 = plt.figure(1)
#plt.scatter(plum.plumeHist[-1].ys, plum.plumeHist[-1].xs)
#show()

plotHex = False
plotSpont = True
plotPDE = False
testFilter = False


if testFilter:
	fNum = 7
	dist = 21
	#dist = 21
	plum.loadData(fileName, fNum)
	r = 0.02
	sparse = 1
	num = 20 / (2*r)
	steps = np.linspace(0, 20, num, endpoint = True)
	cL = []

	#ten lists that we will get at different times throughout
	#the simulation
	#then average them all together and see how smooth they are...

	print "setting it up"
	for i in xrange(10):
		cL.append(list())
	xes = range(100, 300, 20)
	for l, sex in zip(cL, xes):
		print "next level"
		for s in steps:
			c = plum.plumeHist[sex].concentration(s, dist, r, sparse) 
			l.append(c)
		lb = '%s' %sex
		plt.plot(l, label = lb)


	z = []

	print "almost finished"
	for a,b,c,d,e,f,g,h,i,j in zip(cL[0], cL[1], cL[2], cL[3], cL[4],  \
		cL[5], cL[6], cL[7], cL[8], cL[9]):
		z.append((a+b+c+d+e+f+g+h+i+j)/10)


	plt.plot(cL[0], label = '0')
	plt.plot(cL[4], label = '4')
	plt.plot(cL[8], label = '8')

	plt.plot(z, label = "average")
	

	plt.legend()
	savefig('../plots/distrobution/%sand%sand%s.png'%(fNum, dist, size(steps)))

	show()



if plotPDE:
	fNum = 8
	dist = [22, 19, 16]
	#dist = 21
	plum.loadData(fileName, fNum)
	r = 0.02
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
	end   = int(6/0.002)
	step = 5
	fileNumber = 1
	
	""" We need to figure out an equation to relate these"""
	r = 0.02
	norm = (25.0/plum.param.den)
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

	fig2 = plt.figure(2)
	plt.scatter(plum.plumeHist[-1].ys, plum.plumeHist[-1].xs)
	show()
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




