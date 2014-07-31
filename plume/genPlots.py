import sys
import plumeClass
reload(plumeClass)


fileName = sys.argv[1]
print "load plume data from file %s"% fileName
plum = plumeClass.plumeEtAl(None, True, fileName )


from pylab import *
import matplotlib.pyplot as plt



#here we want to plot some intermittency graphs that look good
#here we want to plot some instantaneous distribution and time-averaged distribution

source is at 10, 26

lets go 4 units down and 8 units down

(10, 22) and (10, 18)

here we are going to sample the plume every 5 time steps
we will sample the concentration with varying radius

we will start at time 4 and go to tme 8

start = 4/0.002
end   = 8/0.002
step = 10
for x in xrange(start, end step):





























plotHex = False
if plotHex:
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




