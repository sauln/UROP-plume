import sys
import plumeClass
reload(plumeClass)
import flowField
reload(flowField)


from pylab import *
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde


import matplotlib
matplotlib.rcParams['legend.fancybox'] = True
import scipy




fileName = 'models/nagheeby/sparse_5000'
fileName = 'models/kinzel90/sparse_5000' #'wave/cyclic1_5000' #"lowDisp/mit_Dp1_5000"  #"symetric/sparse_5000.0"
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


#fig1 = plt.figure(1)
#plt.scatter(plum.plumeHist[-1].ys, plum.plumeHist[-1].xs)
#show()



def main():
	concentrationOverSpace()
	#genStats()
	#plotHex()# = False
	#plotSpont()# = False
	#plotPDF()# = False
	#testFilter()# = False
	#plotFlows()# = False
	#plotSpontVaryR()# = False
	#plotSpontVaryS()# = False

	#plotSpontVaryBoth()# = False
	#plotOverlay()# = True
	#now this is going to do the plotting for us...




"""


We need 3 plots that show using 3 different levels of sparsity
a plot at 10 seconds of the sparse plume, heatmap style

and a corresponding sponeneity plot


"""
def concentrationOverSpace():

	#load file 7
	plum.loadData(fileName, 7)

	#we are going to run with 3 different settings
	rs = [0.5, 0.2, 0.05]
	#n = [(-.3364 *r + 1.1182)/(r*r) for r in rs]

	#at line of 20
	length = [2.0*r for r in rs]
	points = [linspace(0,20-(2*r), 100)+r for r in rs]
	
	sparse = 1
 	#print points
	lines = []		
	fig = figure()
	ylim([0,1.2])

	#plt.title("Normalized oncentration across the minor axis of the plume")
	for pts,r in zip(points, rs):
		lines.append(list())
		for p in pts:
			c = plum.plumeHist[-1].concentration(p, 20, r, sparse) #* norm 
			lines[-1].append(c)

		l = normalize(lines[-1])
		plt.plot(pts, l, label='%s'%r)
		

	
	plt.legend(title="sample radius (m)")
	xlabel("Distance (m)")
	ylabel("c/c max")


	show()


def genStats():

	""" I need 4 or 5 different settings with variable radius and puff count

		at each of those levels I want to generate the peak to mean ratio,	
		and the % of readings that are under a certain threshold.
		The threshold should be some ratio of the mean.

	"""

	#turn this into a plot.


	#define the 4 different settings
	
	#for each setting:
		#calculate the concentration across a time interval
		#calculate the mean of this concentration
		#find the peak to mean ratio
		#find the % of readings under a threshold, say 80% of mean


	start = int(8.5/0.002)
	end   = int(9/0.002)
	step = 5
	fileNumber = 1
	x = 8.0; y = 22.0
	print "start %s, stop %s" %(start, end)


	sparse = 1.0  #,4,10, 30, 50]
	radius = linspace(2,.01,500)#[ 1.0, 0.9, 0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.1, 0.05]
	p2ms = []
	means = []
	sparsities = []
	for r in radius:
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
				concentration(x, y, r, sparse) 

			tmp.append(c)

		mean = sum(tmp)/float(len(tmp))
		print "for radius: %s" %r
		p2m = max(tmp)/mean #this is the peak to mean ratio
		print "peak to mean ratio: %s"%p2m
		p2ms.append(p2m)
		means.append(mean)
		print "mean: %s, peak: %s"%(mean, max(tmp))

	#next calculate the number of items under 0.9*mean

		threshold = 0.97 * mean
	
		under = [c for c in tmp if c<threshold]
		print "number under thresh: %s\nnumber total: %s"%(len(under), len(tmp))
		print "Sponenaiety: %s" %(float(len(under))/len(tmp))
		sparsities.append(float(len(under))/len(tmp))


	#fig = figure()
	ax = plt.gca()
	ylim([0,10])
	ax2 = ax.twinx()

	ax.plot(radius, p2ms, 'r-',     label="peak/mean ratio")
	ax2.plot(radius, sparsities, 'b-',label="sparseness".ljust(15))
	ax.set_ylabel("peak/mean ratio", color='r')
	ax2.set_ylabel("% of readings below threshold", color='b')
	ax.set_xlabel("radius (m)")
	
	leg = ax2.legend(fancybox=True,bbox_to_anchor=(0.915,1))
	leg.get_frame().set_alpha(0.0) 
	leg2 = ax.legend(fancybox=True, bbox_to_anchor=(1, 0.93))
	leg2.get_frame().set_alpha(0.0) 


	show()





def plotOverlay():
	'''
	going to overlay the plots from multiple different files
	'''
	


	files = ['models/nagheeby/sparse_5000', 'models/kinzel90/sparse_5000']
	fileNumber = 7


	sparse = 100


	fig = figure()

	ax = fig.add_subplot(121, aspect='equal')
	ax.axis([0,20,0,30])
	fileNumber = 3


	plum.loadData(files[0], fileNumber)
	a = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='b', label="Nagheeby (Chao?)", alpha = 0.5)

	plum.loadData(files[1], fileNumber)
	b = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='r',label="Kinzelbach 1990", alpha = 0.5)

	plt.title("Plume at %s seconds"%fileNumber)
	plt.legend(bbox_to_anchor=(.05, 0.15), loc=2, borderaxespad=0.)
	xlabel("m")
	ylabel("m")





	fileNumber = 10
	ax2 = fig.add_subplot(122, aspect='equal')
	ax2.axis([0,20,0,30])
	plum.loadData(files[0], fileNumber)
	a2 = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='b', label="Nagheeby (Chao?)", alpha = 0.5)

	plum.loadData(files[1], fileNumber)
	b2 = plt.scatter(plum.plumeHist[-1].ys[::sparse], plum.plumeHist[-1].xs[::sparse], c='r',label="Kinzelbach 1990", alpha = 0.5)

	plt.title("Plume at %s seconds"%fileNumber)
	plt.legend(bbox_to_anchor=(.05, 0.15), loc=2, borderaxespad=0.)
	xlabel("m")
	ylabel("m")




	show()




def plotFlows():

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
	
	











def testFilter():
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



def plotPDF():
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



def plotSpontVaryS():

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


def plotSpontVaryBoth():


	#now I am going to plot multiple sponts on the same plot.  

	start = int(6.5/0.002)
	end   = int(7/0.002)
	step = 5
	fileNumber = 1
	


	#we are going to run with 3 different settings

	rs = [ 0.5, 0.2, 0.1]
	sparse = [1,5,10]
	#n = [(-.3364 *r + 1.1182)/((r*r)) for r,s in zip(rs, sparse)]


 	n = [1200.0, 30.0, 3.0]
	cLists = []
	#for r in rs:
		
	#	cLists.append(list())
	

	print "start %s, stop %s" %(start, end)
	x = 8; y = 22
	for r, norm,  s in zip(rs, n, sparse):
		tmp = []
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
				concentration(x, y, r, s) /norm #/ norm

			print "c: %s" %c
			print "c* norm: %s" %(c*norm)
			print "norm: %s" %norm
			#cL.append(c)
		
			tmp.append(c)

		cLists.append(tmp)#[t/max(tmp) for t in tmp])

	#now normalize:
	#maxx = 0
	#for c in cLists:
	#	if max(c) > maxx:
	#		maxx = max(c)

	#print maxx



	#newCl = []
	#for c in cLists:
	#	newCl.append([v/maxx for v in c])
	fig = plt.figure()
	#ylim([0,1])

	for cL, r, s in zip(cLists, rs, sparse):
	#mean = np.mean(cL)
	
		plt.plot(cL, label='%s  $(m)$, %s $(puff/s)$'%(r, 5000/s))

	#plt.plot(([0,0]), ([mean,mean]), 'r')

		#plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	#plt.title("Concentration at (%s, %s) changing both sparsity and radius"%(x,y))
	plt.legend(title="(sample radius, flow rate)")
	xlabel("$t$")
	yl = ylabel(r"$ \frac{c}{c_{max}}$")
	yl.set_fontsize(20)
	savefig('../plots/spontCompareRadius.png')
	
	show()

def normalizeEach(cLists):
	newCl = []
	for c in cLists:
		newCl.append(normalize(c))



	return newCl

def normalize(cList):
	return [float(c)/max(cList) for c in cList]
def normalizeAll(cLists):
	#find the largest in all the lists, normalize it to that

	print shape(cLists)
	if shape(cLists)[0] == 1:
		newCl = [c/max(cLists[0]) for c in cLists]
	else:
		maxx = 0.0
		for c in cLists:
			if max(c) > maxx:
				maxx = max(c)

		print maxx



		newCl = []
		for c in cLists:
			newCl.append([float(v)/maxx for v in c])


	return newCl

def plotSpontVaryR():


	#now I am going to plot multiple sponts on the same plot.  

	start = int(6.5/0.002)
	end   = int(7/0.002)
	step = 5
	fileNumber = 1
	


	#we are going to run with 3 different settings

	rs = [ 0.5, 0.2, 0.05]
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
		
	for c in cLists:
		print c

	newCl = normalizeAll(cLists)
	fig = plt.figure()
	#ylim([0,1])

	for cL, r in zip(newCl, rs):
	#mean = np.mean(cL)
	
		plt.plot(cL, label='%s'%r)

	#plt.plot(([0,0]), ([mean,mean]), 'r')

		#plt.axhline(np.mean(cL), color='b', linestyle='dashed', linewidth=2)
	#plt.title("Normalized oncentration at (%s, %s) \n over time"%(x,y))
	plt.legend(title="sample radius (m)")
	xlabel("Time (t)")
	ylabel("c/c max")
	savefig('../plots/spontCompareRadius.png')
	
	show()








def plotHex():

	start = 7
	end = 9
	h = []
	xe = []
	ye = []
	sparse = 1
	#for t in range(start,end,3):
	t = 8
	plum.loadData(fileName, int(t)+1)
	heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
			ys[::sparse], plum.plumeHist[-1].xs[::sparse], bins=50)

	heatmap = np.rot90(heatmap)
	heatmap = np.flipud(heatmap)
	Hmasked = np.ma.masked_where(heatmap==0,heatmap) # Mask pixels with a value of zero
 
		#print "max: %s min: %s" %(heatmap), heatmap))
		#print shape(heatmap)
		#h.append(Hmasked)
		#xe.append(xedges)
		#ye.append(yedges)



	data = h


	heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
			ys, plum.plumeHist[-1].xs, bins=50)

	heatmap = np.rot90(heatmap)
	#heatmap = np.flipud(heatmap)
	heatmap = np.ma.masked_where(heatmap==0,heatmap)


	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	#plt.clf()
	plt.imshow(heatmap, extent=extent)
	plt.axis([0,20,0,30])
	plt.axis(aspect= 'equal')
	#plt.show()

	#extent = [xe[0], xe[-1], ye[-1], ye[0]]
	#plt.hist2d(X[:, 0], X[:, 1], bins=40, cmap='Greens', norm=LogNorm())
	#im = axes.imshow(data, extent = extent)
	#axes.axis([0,20,0,30])
	#axes.axis(aspect='equal')

	#for dat, ax, x,y, t in zip(data, axes.flat, xe, ye, range(start,end) ):
	#for data,xe,ye,axes:				#in zip(data,  xe, ye):#,  axes.flat ):
		# The vmin and vmax arguments specify the color limits

		#im = axes.imshow(dat, extent = extent, vmin=0, vmax=200)
		
	#im = axes.imshow(data, extent = extent)#, vmin = 0, vmax = 200)
		#plt.pcolormesh(x,y,dat)
	
		#redraw()
		#ax.set_title('time %s'%(t+1))
	#plt.tight_layout()
	#cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
	#fig.colorbar(im, cax=cax)

	show()


if __name__ == '__main__':
   main()

