#this is an lcm tester for the enviroSim.py using the new plume


import auxiliary
reload(auxiliary)


import lcm
from senlcm import *
from numpy import *



def envHandler(channel, data):
	msg = positionSim_t.decode(data)
	print "DU: %s DU_p: %s\nV0: %s D2U0: %s U0: %s" \
		%(msg.DU, msg.DU_p, msg.V0, msg.D2U0, msg.U0)
	print msg.DU[1]



def envReturnHandler(channel, data):
	pass


lcm = lcm.LCM()
subEnv = lcm.subscribe("dataReturn", envHandler)
subEnvRet = lcm.subscribe("envUpdateConfirm", envReturnHandler) 


param = auxiliary.Parameters()
ts = param.T; Dt = param.dt
T_thresh = 2*(1/Dt)


dummyMsg = positionSim_t()
dummyMsg.X0[0] = 12
dummyMsg.X0[1] = 25
#for T in linspace(0, ts, (1/Dt) *ts): #
print ts/Dt

for T in xrange(int(ts/Dt)):
	print T

	if T >= T_thresh:
		dummyMsg.T = T
		lcm.publish("envRetrieve", dummyMsg.encode())
		lcm.handle() #envReturn

		
	
	


