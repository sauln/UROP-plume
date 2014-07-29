import sys
import plumeClass
reload(plumeClass)


fileName = sys.argv[1]
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


from pylab import *
import matplotlib.pyplot as plt

#fig = plt.figure()#figsize=(11,6))
#fig.ax  = []
#fig.ax.append(fig.add_subplot(131, aspect='equal'))

end = 6

h = []
xe = []
ye = []
for t in range(2,11):
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

fig, axes = plt.subplots(nrows=3, ncols=3)
plt.title("Heatmap of plume from simulation %s" %sys.argv[1])

data = h
for dat, ax, x,y, t in zip(data, axes.flat, xe, ye, range(2,11) ):
    # The vmin and vmax arguments specify the color limits
	extent = [x[0], x[-1], y[-1], y[0]]
	im = ax.imshow(dat, extent = extent, vmin=0, vmax=200)
	#plt.pcolormesh(x,y,dat)
	ax.axis([0,20,0,30])
	#redraw()
	ax.set_title('time %s'%(t+1))

cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)

show()

"""

for t in xrange(end):

	plum.loadData(fileName, int(t)+1)
	ax = fig.add_subplot(2, ceil(end/2), t+1, aspect='equal')
	ax.axis([0,20,0,30])



	#ax.scatter(plum.plumeHist[-1].\
	#	ys[::15],plum.plumeHist[-1].xs[::15])


	heatmap, xedges, yedges = np.histogram2d(plum.plumeHist[-1].\
		ys[::15], plum.plumeHist[-1].xs[::15], bins=15)

	# H needs to be rotated and flipped
	H = np.rot90(heatmap)
	H = np.flipud(H)
	# Mask zeros

	Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 

	plt.pcolormesh(xedges,yedges,Hmasked)
	plt.xlabel('x')
	plt.ylabel('y')
	
	#cbar.ax.set_ylabel('Counts')

	#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	#ax.set_title('time %s'%(t+1))
	fig.ax.append(ax)


cbar = plt.colorbar()

"""



