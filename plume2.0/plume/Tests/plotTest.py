

import numpy as np
import scipy.ndimage as ndi

from flowField import *



import time
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt





flow = flowField('mit')


dif = 12

close('all')
fig = plt.figure(1)
ax = fig.add_subplot(111)
#line1, line2 = ax.plot(x, y, 'r-') # Returns a tuple of line objects, thus the comma





#plt.streamplot(flow.x, flow.y, flow.vy, flow.vx, linewidth=2, cmap=plt.cm.autumn)
#plt.colorbar()



def update_lines(ptx, pty):
	plt.clf()
	Q = quiver( flow.x[:-2:dif,::dif], flow.y[:-2:dif,:-2:dif], \
		flow.vy[:-2:dif, :-2:dif], flow.vx[:-2:dif, :-2:dif], 
		color = 'k', units='x',pivot=( 'head' ),
		linewidths=(0,0), edgecolors=('k'), headaxislength=0.1 )
	s = scatter(pty,ptx)
	plt.axis('equal')
	plt.title('Simplest default with labels')
	plt.draw()




show()

for x in xrange(500):
	ptx = np.random.uniform(0, 30, 25)
	pty = np.random.uniform(0, 20, 25) #np.random.uniform(150, 220, 20), 
	update_lines(ptx, pty)
	time.sleep(0.001)
	print "update"








