import cPickle
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

import time

import plumeClass
reload(plumeClass)


def main(argv):
	print "begin"


	print "load?"
	plum = plumeClass.plumeEtAl(None, True, argv[1] )



	fig = plt.figure(1)
	plt.clf()
	ax = fig.add_subplot(111, aspect='equal')
	plt.axis([0,20,0,30])
	fig.sc = None
	show()

	print "make movie"
	plum.makeMovie(fig)

if __name__ == '__main__':
    main(sys.argv) 


