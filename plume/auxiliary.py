
"""

Nathaniel Saul 2014
UROP plume simulation
Field Robotics Laboratory
University of Hawaii

This module holds the auxiliary functions that we use in all of our simulations and experiments






"""


import sys
import os
import time

from numpy import *
from pylab import *

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



class Parameters():
	def __init__(self):
		self.T = 12.0
		self.dt = 0.002
		self.den = 5000.0
		self.flow = 'simple'#'mit'
		self.std = 1.0
		self.adjust = 3.0/self.den
		self.xi = 10.0
		self.yi = 26.0

		self.steps = int( ceil( self.T/self.dt ) )
		ticks = 1/self.dt
		self.ratio = self.den/ticks


def findSTD(points):
	total = 0.0			
	quant = size(points)
	for each in points:
		total += each
	mean = total/quant

	for each in points:
		total = (each - mean)**2
	std = sqrt(total/quant)
	return std









