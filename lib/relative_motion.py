from copy import deepcopy
import numpy as np
from constants import EARTH_MU, GEO_RADIUS, METERS_IN_KM, SECONDS_IN_DAY
from math import sqrt, sin, cos
import matplotlib.pyplot as plt
import CustomExceptions as e
import SysConfigs as cfgs
import time

class GeoRelativeModel():
	"""
	This class performs calculations of relative satellite positions using Clohessy-Wiltshire
	equations defined on page X of 'Fundamentals of Astrodynamics and Applications' by David Vallado.
	"""
	
	def __init__(self, t0_state, a):
		"""
		Defines starting state for two spacecraft given the state of chase relative to target at time == 0
		
		@type t0_state:  	numpy array of dimensions [1, 6]
		@param t0_state: 	[x, y, z, x_dot, y_dot, z_dot] in meters and meters per second respectively.  
							The values x, y, z represent radial, in-track, and cross-track.
		@type a:			number
		@param a:			Average altitude of target spacecraft                       
		@rtype:         	None
		@return:        	N/A
		"""
		
		assert t0_state.shape == cfgs.REL_STATE_DIM, e.unexpectedArrayDim(t0_state, cfgs.REL_STATE_DIM)
		
		self.init_state = deepcopy(t0_state)
		self.n = sqrt((EARTH_MU/a**3))
	
	def solveNextState(self, t):
		"""
		Returns the relative state of the system at time t
		
		@type t:			number
		@param t:			Number of seconds from the initial state epoch
		@rtype:				numpy array
		@return:			Future [x, y, z, x_dot, y_dot, z_dot] in meters and meters per second
		"""
		
		#copy argument multiple 
		n = self.n
		
		#define system matrix of x, y, z, x_dot, y_dot, z_dot equations
		sys_mat = np.array([
					[4-3*cos(n*t), 0, 0, sin(n*t)/n, 2*(1-cos(n*t))/n, 0],
					[6*(sin(n*t) - n*t), 1, 0, -2*(1-cos(n*t))/n, (4*sin(n*t) - 3*n*t)/n, 0],
					[0, 0, cos(n*t), 0, 0, sin(n*t)/n],
					[3*n*sin(n*t), 0, 0, cos(n*t), 2*sin(n*t), 0],
					[-6*n*(1-cos(n*t)), 0, 0, -2*sin(n*t), 4*cos(n*t)-3, 0],
					[0, 0, -n*sin(n*t), 0, 0, cos(n*t)]])
					
		return np.dot(sys_mat, self.init_state)
		
def test():
	x = []
	y = []
	TestModel = GeoRelativeModel(cfgs.REL_TEST_STATE, GEO_RADIUS*METERS_IN_KM)
	for i in range(-SECONDS_IN_DAY, SECONDS_IN_DAY, cfgs.DEFAULT_STEP):
		state = TestModel.solveNextState(i)
		x.append(state[1][0])
		y.append(state[0][0])
	plt.plot(x, y)
	plt.show()
	
if __name__=="__main__":
	test()
