#we need to figure out what is happening wrong with the step.so

#so here is a test



import numpy as np


import step
reload(step)

import plumeClass
reload(plumeClass)

import auxiliary
reload(auxiliary)

import flowField
reload(flowField)

np.random.seed(1)

param = auxiliary.Parameters()
plum = plumeClass.plume(param)
plum.puffSoA.addPuffs(plum.param.yi, plum.param.xi, 1)
plum.kinzelbach1990SoA()


x = np.array([12.0])
y = np.array([26.0])

flow = flowField.flowField('simple')

	
x, y = step.kinzelbach1990SoA(flow, y, x)



print "New kinzelbach"
print x, y

print "old kinzelbach"
print plum.puffSoA.xs
print plum.puffSoA.ys

