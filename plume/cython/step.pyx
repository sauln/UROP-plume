import numpy as np
cimport numpy as np
#from numpy import *
cimport cython

DTYPE = np.float64
#FTYPE = np.float

ctypedef np.float_t DTYPE_t
#ctypedef np.float_t


 
#as of 7/8/14 the step.kinzelbach1990SoA takes 0.084 seconds per call
#25.193 or 33.607 second simulation is spent in kinzelbach.
#lets try to optomize

@cython.boundscheck(False)
def concentration(np.ndarray[DTYPE_t, ndim=1] ys, \
	np.ndarray[DTYPE_t, ndim=1] xs, float y, float x, float r, int sparse = 1):

	cdef np.ndarray xT, yT, xU, yU, yI, xI, both
	cdef int tot
	cdef float tx, ty, i, j

	xT = np.where(xs[::sparse] < x + r)[0]
	xU = np.where(x - r < xs[::sparse])[0]
	yT = np.where(ys[::sparse] < y + r)[0]
	yU = np.where(y - r < ys[::sparse])[0]

	yI = np.intersect1d( yU, yT )
	xI = np.intersect1d( xU, xT )
	both = np.intersect1d(yI, xI)
	return both.size
	




@cython.boundscheck(False)
def kinzelbach1990(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] vx, np.ndarray[DTYPE_t, ndim=1] vy):
	#for some reason x and y got switched - 
	''' 
		Equation described in Kinzelback1990
		adapted to incorporate a structure of arrays instead 
		of an array of structures
	'''

	cdef float D = 0.2 #diffusion coefficient
	cdef float dt = 0.002
	cdef float each, alf, v, i, ad, di
	cdef np.ndarray h = np.zeros([x.shape[0], 1], dtype=DTYPE)


	#print x[0]
	#assert type(x[0]) == DTYPE and type(y[0]) == DTYPE

	if (x.shape[0] != y.shape[0]):
		raise ValueError("arrays not the same size")

	cdef np.ndarray[DTYPE_t, ndim=1] alphaX = np.asarray([np.random.normal(0,1) for each in x])
	cdef np.ndarray[DTYPE_t, ndim=1] alphaY = np.asarray([np.random.normal(0,1) for each in x])

	cdef np.ndarray[DTYPE_t, ndim=1] vxFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] vyFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vy])

	cdef np.ndarray[DTYPE_t, ndim=1] advX = np.asarray([dt*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] advY = np.asarray([dt*each for each in vy])


	cdef np.ndarray[DTYPE_t, ndim=1] difX = np.asarray([alf * ((abs(2 * D * v * dt)) ** .5  ) \
		for alf, v in zip(alphaX, vxFix)])

	cdef np.ndarray[DTYPE_t, ndim=1] difY = np.asarray([alf * ((abs(2 * D * v * dt)) ** .5  ) 
		for alf, v in zip(alphaY, vyFix)])

	cdef np.ndarray[DTYPE_t, ndim=1] xs = np.asarray([i + ad + di 	for i, ad, di in zip(x, advX, difX)])
	cdef np.ndarray[DTYPE_t, ndim=1] ys = np.asarray([i + ad + di 	for i, ad, di in zip(y, advY, difY)])
	
	

	newArr = np.where(xs >= 30)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs >= 30)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(xs <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(ys >= 20)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys >= 20)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]
		

	newArr = np.where(ys <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]



	cdef np.ndarray[DTYPE_t, ndim=1] finX = np.asarray(xs)
	cdef np.ndarray[DTYPE_t, ndim=1] finY = np.asarray(ys)

	return finX, finY




@cython.boundscheck(False)
def Negheeby2010(		np.ndarray[DTYPE_t, ndim=1] x,
						np.ndarray[DTYPE_t, ndim=1] y,  	
						np.ndarray[DTYPE_t, ndim=1] vx, 
						np.ndarray[DTYPE_t, ndim=1] vy,
						float dispIn):
		#for some reason x and y got switched - 
	''' 
		Equation described in Kinzelback1990
		adapted to incorporate a structure of arrays instead 
		of an array of structures
	'''

	#cdef float D = 0.01 #diffusion coefficient
	#cdef np.ndarray x = np.zeros([10, 1], dtype=DTYPE)
	#cdef np.ndarray y = np.zeros([10, 1], dtype=DTYPE)
	#cdef np.ndarray vx = np.zeros([10, 1], dtype=DTYPE)
	#cdef np.ndarray vy = np.zeros([10, 1], dtype=DTYPE)

	cdef float D = dispIn
	cdef float dt = 0.002
	cdef float each, alf, v, i, ad, di, R, T
	cdef np.ndarray h = np.zeros([x.shape[0], 1], dtype=DTYPE)


	#print x[0]
	#assert type(x[0]) == DTYPE and type(y[0]) == DTYPE

	if (x.shape[0] != y.shape[0]):
		raise ValueError("arrays not the same size")

	""" Random variables in the x and y direction """
	"""
	cdef np.ndarray[DTYPE_t, ndim=1] alphaX = np.asarray([np.random.normal(0,1) for each in x])
	cdef np.ndarray[DTYPE_t, ndim=1] alphaY = np.asarray([np.random.normal(0,1) for each in x])
	"""
	cdef np.ndarray[DTYPE_t, ndim=1] alphaX = np.random.uniform(0,1, x.size)
	cdef np.ndarray[DTYPE_t, ndim=1] alphaY = np.random.uniform(0,1, x.size)


	cdef np.ndarray[DTYPE_t, ndim=1] thetaX = np.random.uniform(0,1, x.size)
	cdef np.ndarray[DTYPE_t, ndim=1] thetaY = np.random.uniform(0,1, x.size)


	""" velocity vector """ 
	cdef np.ndarray[DTYPE_t, ndim=1] vxFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] vyFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vy])



	""" Advection component"""
	cdef np.ndarray[DTYPE_t, ndim=1] advX = np.asarray([dt*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] advY = np.asarray([dt*each for each in vy])


	""" Old diffusion bits
	cdef np.ndarray[DTYPE_t, ndim=1] difX = np.asarray(\
		[alf * ((abs(2 * D * v * dt)) ** .5  ) \
		for alf, v in zip(alphaX, vxFix)])
	cdef np.ndarray[DTYPE_t, ndim=1] difY = np.asarray(\
		[alf * ((abs(2 * D * v * dt)) ** .5  ) 
		for alf, v in zip(alphaY, vyFix)])
	"""


	""" New diffusion bit """
	cdef np.ndarray[DTYPE_t, ndim=1] difX = np.asarray(\
		[(R * ( (12 * D * dt) ** .5  ) ) * np.cos(2 * np.pi * T) \
		for R, T in zip(alphaX, thetaX)])

	cdef np.ndarray[DTYPE_t, ndim=1] difY = np.asarray(\
		[(R * ( (12 * D * dt) ** .5  ) ) * np.sin(2 * np.pi *T) \
		for R, T in zip(alphaY, thetaY)])


	#print advX[-1], difX[-1]

	cdef np.ndarray[DTYPE_t, ndim=1] xs = np.asarray([i + ad + di 	for i, ad, di in zip(x, advX, difX)])
	cdef np.ndarray[DTYPE_t, ndim=1] ys = np.asarray([i + ad + di 	for i, ad, di in zip(y, advY, difY)])
	
	

	newArr = np.where(xs >= 30)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs >= 30)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(xs <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(ys >= 20)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys >= 20)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]
		

	newArr = np.where(ys <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]



	cdef np.ndarray[DTYPE_t, ndim=1] finX = np.asarray(xs)
	cdef np.ndarray[DTYPE_t, ndim=1] finY = np.asarray(ys)

	return finX, finY


""" ########################################################### """

@cython.boundscheck(False)
def NegheebyWOran(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, np.ndarray[DTYPE_t, ndim=1] vx, np.ndarray[DTYPE_t, ndim=1] vy, float D):
	#for some reason x and y got switched - 
	''' 
		Equation described in Kinzelback1990
		adapted to incorporate a structure of arrays instead 
		of an array of structures
	'''

	#cdef float D = 0.1 #diffusion coefficient
	cdef float dt = 0.02
	cdef float each, alf, v, i, ad, di, R, T


	if (x.shape[0] != y.shape[0]):
		raise ValueError("arrays not the same size")


	cdef np.ndarray h = np.zeros([x.shape[0], 1], dtype=DTYPE)







	""" velocity vector """ 
	cdef np.ndarray[DTYPE_t, ndim=1] vxFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] vyFix = np.asarray([(each == 0)*1.0 + (each != 0)*each for each in vy])


	""" Advection component"""
	cdef np.ndarray[DTYPE_t, ndim=1] advX = np.asarray([dt*each for each in vx])
	cdef np.ndarray[DTYPE_t, ndim=1] advY = np.asarray([dt*each for each in vy])


	cdef float difX = (12*D*dt)**0.5 #* np.cos(2*np.pi)
	cdef float difY = (12*D*dt)**0.5 #* np.sin(2*np.pi)

	difX = 0
	difY = 0

	#print advX[-1], difX[-1]

	cdef np.ndarray[DTYPE_t, ndim=1] xs = np.asarray([i + ad + difX 	for i, ad in zip(x, advX)])
	cdef np.ndarray[DTYPE_t, ndim=1] ys = np.asarray([i + ad + difY 	for i, ad in zip(y, advY)])
	
	

	newArr = np.where(xs >= 30)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs >= 30)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(xs <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(xs <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]

	newArr = np.where(ys >= 20)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys >= 20)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]
		

	newArr = np.where(ys <= 0)[0]
	if newArr.any():
		print "delete"
		xs = np.delete(xs, newArr)
		ys = np.delete(ys, newArr)

		newArr = np.where(ys <= 0)[0]
		if newArr.any():
			print "did not delete correctly"
			print xs[newArr], ys[newArr]



	cdef np.ndarray[DTYPE_t, ndim=1] finX = np.asarray(xs)
	cdef np.ndarray[DTYPE_t, ndim=1] finY = np.asarray(ys)

	return finX, finY



	

	


