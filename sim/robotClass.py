#robot class
import lcm
from senlcm import *
from numpy import *
import matplotlib.pyplot as plt

from robot_model_par import theta



class robot():
	def __init__(self):
		self.dataStore = dataStore()
		self.t    = 0
		self.V0   = [-1.5082,3.1416]
		self.DU   = [0.2968, 0.0271]
		self.DU_p = [-0.0909, 0.9959]
		self.U0   = 0.3375
		self.D2U0 = 0.2109 
		self.Xhatdot = [0.2, 0.2]
		self.X0 = [11.0, 24.0]
		self.Xhat = [11.0, 24.0]
		self.theta = theta
		self.msg = positionSim_t()

	def plot(self):
		len(self.dataStore.xRoboty)
		#print self.dataStore.xRoboty

		plt.plot(self.dataStore.xRoboty,self.dataStore.xRobotx)
		v = [0, 20, 0, 30]
		plt.axis(v)
		plt.axis('equal')
		plt.show()

	def animate(self):
		#here we will need to obtain the back log of the concentration matrix
		#so that we can plot the image just like Shaui made his video.		

		pass

	def store(self, T):
		self.dataStore.add(T, self.X0[0], self.X0[1],  \
				self.Xhat[0], self.Xhat[1])

	def buildCONMsg(self):#create the message that gets sent to the control algorithm
		msg = positionSim_t()
		msg.V0   = self.V0
		msg.DU   = self.DU
		msg.DU_p = self.DU_p
		msg.U0   = self.U0
		msg.D2U0 = self.D2U0
		msg.Xhatdot = self.Xhatdot
		msg.X0 = self.X0
		msg.Xhat = self.Xhat
		msg.theta = self.theta
		return msg

		"""

		self.msg.V0   = self.V0
		self.msg.DU   = self.DU
		self.msg.DU_p = self.DU_p
		self.msg.U0   = self.U0
		self.msg.D2U0 = self.D2U0
		#self.msg.Xhatdot = self.Xhatdot
		self.msg.X0 = self.X0
		self.msg.theta = self.theta
		#self.msg.Xhat = self.Xhat
		return self.msg

		"""


	def buildENVMsg(self):#create the message that gets sent to the environment
		msg = positionSim_t()
		msg.X0 = self.X0
		return msg

	def extractENVMsg(self, msg):#save off return data from the environment (sense environment)
		self.V0   = msg.V0
		self.DU   = msg.DU
		self.DU_p = msg.DU_p
		self.U0   = msg.U0
		self.D2U0 = msg.D2U0

	def extractCONMsg(self, msg):#save off return data from the control
		
		self.theta = msg.theta
		#self.Xhatdot = matrix(msg.Xhatdot).H
		self.X0      = matrix(msg.X0).H
		#self.Xhat    = matrix(msg.Xhat).H


class dataStore():
	def __init__(self):
		#store data here:
		self.T			= []
		self.xRobotx 	= []
		self.xRoboty 	= []
		self.xHatx 		= []
		self.xHaty		= []

	def add(self, T, X0x, X0y, hatx, haty):
		self.T.append(T)
		self.xRobotx.append(X0x)
		self.xRoboty.append(X0y)
		self.xHatx.append(hatx)
		self.xHaty.append(haty)


