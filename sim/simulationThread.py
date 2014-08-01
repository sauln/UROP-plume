#************************************************************#
#
#  This is the framework finally written in python for the
#  simulation.  This is part of 3 piece program.  This 
#  simulationThread.py, the environmentThread.py, and the
#  controlThread.py.  Please start the env and cont first.
#  Next steps are to actually send data back and forth and
#  figure out how to process the input and outputs from the
#  respective threads.
#
#
#************************************************************#

#

import lcm
from senlcm import *
import scipy.io as sio
import sys
import time
#from robotClass import *
import robotClass
reload(robotClass)

import matplotlib.pyplot as plt
from numpy import *
eps = sys.float_info.epsilon

from constants import Dt, T_thresh, ts

#setup robot object
rob = robotClass.robot()
T_thresh = T_thresh*(1/Dt)

dummyMsg = positionSim_t()


print "Time steps of: %s\nBegin robot at %s\nUntil %s"%(Dt, T_thresh, ts/Dt)






def envHandler(channel, data):#handle messages recieved over channel envReturn - contains all of the (sense environment) data
	msg = positionSim_t.decode(data)
	rob.extractENVMsg(msg)
	#print 'Environment has been sensed'
	
def conHandler(channel, data):#handle messages from the control- conReturn - contains all of the
	msg = positionSim_t.decode(data)
	rob.extractCONMsg(msg)
	#print 'Robots have moved'


lcm = lcm.LCM()
subEnv = lcm.subscribe("dataReturn", envHandler)
subCon = lcm.subscribe("conReturn", conHandler) 





for T in xrange(int(ts/Dt)):

	
	
	if T >= T_thresh:
		print T
		dummyMsg.T = T
		dummyMsg.X0 = rob.X0
		lcm.publish("envRetrieve", dummyMsg.encode())
		lcm.handle() #envReturn
		

		conMsg = rob.buildCONMsg()
		lcm.publish("conUpdate", conMsg.encode())
		lcm.handle()





msg 		= finishSim_t()
msg.length  = len(rob.dataStore.xRobotx)
msg.T       = rob.dataStore.T
msg.xRobotx = rob.dataStore.xRobotx
msg.xRoboty = rob.dataStore.xRoboty


dummyMsg.T = -1
lcm.publish( "envRetrieve", dummyMsg.encode())
lcm.publish( "finishSim", msg.encode() )

time.sleep(1)


