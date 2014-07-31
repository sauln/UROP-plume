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

import cPickle as pickle
import plumeClass
reload(plumeClass)
import auxiliary
reload(auxiliary)

####
###	We need to save the data off.  the data file shouldn't be too big. 
### 
###
###



def main(argv): 


	param = auxiliary.Parameters()
	plum = plumeClass.plume(param,  argv[1], False)


	#fileName = "npSaveExperiment.bin"
	np.random.seed(1)	
	plum = generateData(plum)
	
	print "Finished generating data."


def generateData(plum):
	print "Begin generating data"
	print "Will generate data for 0-%s seconds.\nTimesteps of size %s.\n%s puffs per timestep"%(plum.param.T, plum.param.dt, plum.param.den)
	timer = arange(plum.param.T+1)
	cT = 0
	#for x in linspace(0, plum.param.T, plum.param.steps, endpoint=True):

	for x in xrange(int(plum.param.T/plum.param.dt)):
		plum.tickSoA(x)
		


	plum.plumeEtAl.saveData(plum.param.T)
	
	return plum



if __name__ == '__main__':
    main(sys.argv) 






	
