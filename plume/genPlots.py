import sys
import plumeClass
reload(plumeClass)
import flowField
reload(flowField)


from pylab import *
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde


import scipy




fileName = 'mit/sparse_5000.0' #'wave/cyclic1_5000' #"lowDisp/mit_Dp1_5000"  #"symetric/sparse_5000.0"
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


#fig1 = plt.figure(1)
#plt.scatter(plum.plumeHist[-1].ys, plum.plumeHist[-1].xs)
#show()

plotHex = False
plotSpont = False
plotPDF = False
testFilter = False
plotFlows = False
plotSpontVaryR = False
plotSpontVaryS = False

plotSpontVaryBoth = False
plotOverlay = True
"""


We need 3 plots that show using 3 different levels of sparsity
a plot at 10 seconds of the sparse plume, heatmap style

and a corresponding sponeneity plot


"""


if plotOverlay:
	'''
	going to overlay the plots from multiple different files
	'''
	


	files = ['mit/sparse_5000.0', 'models/kinzel90/sparse_5000']
	fileNumber = 5



	fig = figure()

	ax = fig.add_subplot(111, aspect='equal')
	ax.axis([0,20,0,30])
	#plt.axes('equal')
	sparse = 50




	plum.loadData(files[0], fileNumber)
	a = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='b')

	plum.loadData(files[1], fileNumber)
	b = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='r')










	show()




if plotFlows:

	print "Here i am going to plot both of the fluid flows side by side, or just 1 at a time is fine.  "



	flowS = flowField.flowField('simple')
	flowM = flowField.flowField('mit')

	gap = 10

	plt.subplot(1, 2, 1)
	#plt.axes('equal')
	plt.quiver( flowS.x[::gap,::gap], flowS.y[::gap,::gap], \
		flowS.vy[::gap, ::gap], flowS.vx[::gap, ::gap], 
		color = 'g', units='x',pivot=( 'tail' ),
        linewidths=(1,), edgecolors=('g'), headaxislength=10 )
	
	plt.subplot(1, 2, 2)

	plt.quiver( flowM.x[::gap,::gap], flowM.y[::gap,::gap], \
		flowM.vy[::gap,::gap], flowM.vx[::gap,::gap], 
		color = 'b', units='x',pivot=( 'tail' ),
        linewidths=(1,), edgecolors=('b'), headaxislength=20)

	show()
	
	











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



if plotPDF:
	fNum = 8
	dist = [22, 19, 16]
	#dist = 21
	plum.loadData(fileName, fNum)
	r = 0.2
	sparse = 1
	num = 20 / (2*r)
	steps = np.linspace(0, 20, num, endpoint = True)
	cL = []
	
	''' setup '''
	#cL.append(list())

	''' gathering the concentrations'''
	
	for s in steps:
		c = plum.plumeHist[-1].concentration(s, 22, r, sparse) 
		cL.append(c)
		
		#plt.plot(l)
	

	print "mean %s" % scipy.stats.mstats.tmean(cL)
	print "variance %s" %scipy.stats.mstats.variation(cL)


	smoothness = 100
	kde = []; distSpace = []; p = []; txt = []
		
	print cL

	y =  scipy.stats.multivariate_normal.pdf(steps, scipy.stats.mstats.tmean(cL), scipy.stats.mstats.variation(cL))

	plt.plot(y)

	kd = gaussian_kde(cL)
	dS = linspace( min(cL), max(cL), smoothness)
	#pl = plt.plot(dS, kd(dS), label = "3 units from source" )

	#pl = plt.plot(cL)

	print kd(dS)

	#		plt.legend()
	savefig('../plots/distrobution/%sand%sand%s.png'%(fNum, dist, size(steps)))

	show()



if plotSpontVaryS:

	#now I am going to plot multiple sponts on the same plot.  

	start = int(5.2/0.002)
	end   = int(6/0.002)
	step = 5
	fileNumber = 1
	x = 8.0; y = 22.0
	

	r = 0.5

	norm = (-.3364 *r + 1.1182)/(r*r) 
	#sparse = [1,5,20,100]
 	sparse = [1,10,50]#, 100]

	cLists = []
	#for r in sparse:
	#	cLists.append(list())
	
	print "start %s, stop %s" %(start, end)

	#for s, cL in zip(sparse, cLists):
	#	tmp = []
	


	for s in sparse:
		tmp = []
		for T in xrange(start, end, step):

			if( T*plum.param.dt >= fileNumber and T != 0):
				'''if we are inside of the next file, load it'''
				global fileNumber
				fileNumber = int( T/(1/plum.param.dt) ) + 1 
				print "load next file"
				t = T/(1/plum.param.dt)
				plum.loadData(fileName, fileNumber)

			c = plum.plumeHist[int(T%(1/plum.param.dt))].\
				concentration(x, y, r, s) * norm

			tmp.append(c)

		cLists.append([t/max(tmp) for t in tmp])


	#con = [t/max(tmp) for t in tmp] 


	#now normalize:
	#print "eror here"
	#print type(cLists)
	#print type(cLists[0])
	#print max(cLists[0])
	#maxx = 0
	#for c in cLists:
	#	print type(c)#
	#	if max(c) > maxx:
	#		maxx = max(c)

	#print maxx

	
	#newCl = []
	#for c in cLists:
	#	newCl.append([v/maxx for v in c])
	#	print [v/maxx for v in c]
	
	


	fig = plt.figure()
	ylim([0,1.5])

	#for cL, r in zip(cLists, sparse):
	#mean = np.mean(cL)
	for con, s in zip(cLists, sparse):
		plt.plot(con, label='%s'%s)

	#plt.plot(([0,0]), ([mean,mean]), 'r')

		#plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	plt.title("Concentration at (%s, %s) with sparsity, not radius."%(x,y))
	plt.legend()
	xlabel("time")
	ylabel("c/c max")
	savefig('../plots/spontCompareSparse.png')
	
	show()


if plotSpontVaryBoth:


	#now I am going to plot multiple sponts on the same plot.  

	start = int(5.5/0.002)
	end   = int(6/0.002)
	step = 5
	fileNumber = 1
	


	#we are going to run with 3 different settings

	rs = [ 0.5, 0.2, 0.1]

	n = [(-.3364 *r + 1.1182)/(r*r) for r in rs]
	sparse = [1,5,20]

 
	cLists = []
	for r in rs:
		
		cLists.append(list())
	

	print "start %s, stop %s" %(start, end)
	x = 8; y = 22
	for r, norm, cL, s in zip(rs, n, cLists,sparse):

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

			c = plum.plumeHist[int(T%(1/plum.param.dt))].\
				concentration(x, y, r, s) * norm

			cL.append(c)
		

	#now normalize:
	maxx = 0
	for c in cLists:
		if max(c) > maxx:
			maxx = max(c)

	print maxx



	newCl = []
	for c in cLists:
		newCl.append([v/maxx for v in c])
	fig = plt.figure()
	#ylim([0,1])

	for cL, r in zip(newCl, rs):
	#mean = np.mean(cL)
	
		plt.plot(cL, label='%s'%r)

	#plt.plot(([0,0]), ([mean,mean]), 'r')

		#plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	plt.title("Concentration at (%s, %s) with no change in sparsity"%(x,y))
	plt.legend()
	xlabel("time")
	ylabel("c/c max")
	savefig('../plots/spontCompareRadius.png')
	
	show()



if plotSpontVaryR:


	#now I am going to plot multiple sponts on the same plot.  

	start = int(5.5/0.002)
	end   = int(6/0.002)
	step = 5
	fileNumber = 1
	


	#we are going to run with 3 different settings

	rs = [ 0.5, 0.2, 0.1, 0.05]
	n = [(-.3364 *r + 1.1182)/(r*r) for r in rs]

	sparse = 1
 
	cLists = []
	for r in rs:
		
		cLists.append(list())
	

	print "start %s, stop %s" %(start, end)
	x = 8; y = 22
	for r, norm, cL in zip(rs, n, cLists):

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

			c = plum.plumeHist[int(T%(1/plum.param.dt))].\
				concentration(x, y, r, sparse) * norm

			cL.append(c)
		

	#now normalize:
	maxx = 0
	for c in cLists:
		if max(c) > maxx:
			maxx = max(c)

	print maxx



	newCl = []
	for c in cLists:
		newCl.append([v/maxx for v in c])
	fig = plt.figure()
	#ylim([0,1])

	for cL, r in zip(newCl, rs):
	#mean = np.mean(cL)
	
		plt.plot(cL, label='%s'%r)

	#plt.plot(([0,0]), ([mean,mean]), 'r')

		#plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	plt.title("Concentration at (%s, %s) with no change in sparsity"%(x,y))
	plt.legend()
	xlabel("time")
	ylabel("c/c max")
	savefig('../plots/spontCompareRadius.png')
	
	show()








if plotHex:

	start = 2
	end = 8
	h = []
	xe = []
	ye = []
	sparse = 1
	for t in range(start,end):
		print t
		plum.loadData(fileName, int(t)+1)
		heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
				ys[::sparse], plum.plumeHist[-1].xs[::sparse], bins=75)

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
	else:
		si = 3
	
	fig, axes = plt.subplots(nrows=3, ncols = 2)
	plt.title("Heatmap of plume from simulation %s" %fileName)

	data = h

	#for dat, ax, x,y, t in zip(data, axes.flat, xe, ye, range(start,end) ):
	for dat,  x,y, t, ax in zip(data,  xe, ye, range(start,end), axes.flat ):
		# The vmin and vmax arguments specify the color limits
		extent = [x[0], x[-1], y[-1], y[0]]
		#im = axes.imshow(dat, extent = extent, vmin=0, vmax=200)
		
		im = ax.imshow(dat, extent = extent)#, vmin = 0, vmax = 200)
		#plt.pcolormesh(x,y,dat)
		ax.axis([0,20,0,30])
		#redraw()
		ax.set_title('time %s'%(t+1))

	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(im, cax=cax)

	show()




