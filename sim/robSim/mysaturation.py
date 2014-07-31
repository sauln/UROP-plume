import sys
from math import *
from numpy import *

eps = sys.float_info.epsilon

def mysaturation( x, max_val ):
	newx      = x.H
	x         = sqrt(  power(newx[:,0], 2) + power(newx[:,1], 2)  )
	myy       = (x >= max_val ) * max_val    +  (x<max_val) * x;
	newxangle = diag((x+eps)**(-1))*newx;
	y 		  = diag(myy)*newxangle;
	return y
