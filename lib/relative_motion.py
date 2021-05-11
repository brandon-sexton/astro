from copy import deepcopy
import numpy as np
from lib.constants import EARTH_MU
from math import sqrt, sin, cos, atan, pi, isnan
import matplotlib.pyplot as plt
import lib.CustomExceptions as e
import lib.SysConfigs as cfgs
import time

class RelativeModel():
	"""
	This class performs calculations of relative satellite positions using Clohessy-Wiltshire
	equations defined on page X of 'Fundamentals of Astrodynamics and Applications' by David Vallado.
	"""
	
	def __init__(self, t0_state, a):
		"""
		Defines starting state for two spacecraft given the state of chase relative to target at time == 0
		
		@type t0_state:  	numpy array of dimensions (6, 1)
		@param t0_state: 	[x, y, z, x_dot, y_dot, z_dot] in meters and meters per second respectively.  
							The values x, y, z represent radial, in-track, and cross-track.
		@type a:			number
		@param a:			Average altitude of target spacecraft                       
		@rtype:         	None
		@return:        	N/A
		"""
		
		#Verify input numpy array is appropriate dimensions
		assert t0_state.shape == cfgs.REL_STATE_DIM, e.unexpectedArrayDim(t0_state, cfgs.REL_STATE_DIM)
		
		#Copy input array into internal state
		self.init_state = deepcopy(t0_state)
		
		#Save orbital rate value internally to prevent future repetitive calculations
		self.a = a
		self.n = sqrt((EARTH_MU/a**3))
	
	def getSystemMatrix(self, t):
		"""
		Returns the system of CW equations in matrix form.
		
		@type t:			number
		@param t:			Number of seconds from the initial state epoch
		@rtype:				numpy array
		@return:			Matrix representation of CW equations for time t
		"""
		
		n = self.n
		
		#define system matrix of x, y, z, x_dot, y_dot, z_dot equations
		sys_mat = np.array([
					[4-3*cos(n*t), 0, 0, sin(n*t)/n, 2*(1-cos(n*t))/n, 0],
					[6*(sin(n*t) - n*t), 1, 0, -2*(1-cos(n*t))/n, (4*sin(n*t) - 3*n*t)/n, 0],
					[0, 0, cos(n*t), 0, 0, sin(n*t)/n],
					[3*n*sin(n*t), 0, 0, cos(n*t), 2*sin(n*t), 0],
					[-6*n*(1-cos(n*t)), 0, 0, -2*sin(n*t), 4*cos(n*t)-3, 0],
					[0, 0, -n*sin(n*t), 0, 0, cos(n*t)]])
					
		return sys_mat
		
	def solveNextState(self, t):
		"""
		Returns the relative state of the system at time t
		
		@type t:			number
		@param t:			Number of seconds from the initial state epoch
		@rtype:				numpy array
		@return:			Future x, y, z, x_dot, y_dot, z_dot in meters and meters per second
		"""
		
		#Get matrix of CW equations for time t
		sys_mat = self.getSystemMatrix(t)
					
		return np.dot(sys_mat, self.init_state).reshape(cfgs.REL_STATE_DIM)
		
	def solveTimedBurnToWP(self, t, wp, applyResult=True):
		"""
		Returns velocity change required to hit a waypoint in a restricted amount of time
		
		@type t:			number
		@param t:			Number of seconds from the initial state epoch
		@type wp:			numpy array
		@param wp:			Desired positional offset to achieve
		@rtype:				numpy array
		@return:			Velocity difference required to achieve position waypoint
		"""
		
		#Verify input numpy array is appropriate dimensions
		assert wp.shape == cfgs.XYZ_DIM, e.unexpectedArrayDim(wp, cfgs.XYZ_DIM)
		
		#Get matrix of CW equations for time t
		sys_mat = self.getSystemMatrix(t)
		
		#Solve constant side of linear system using first 3 equations in CW matrix
		#This is possible because starting and stopping positions are known over given interval
		#We are left with 3 unknown velocities and 3 equations 
		constants = wp - np.dot(sys_mat[0:3, 0:3], self.init_state[0:3, 0]).reshape(cfgs.XYZ_DIM)
		
		#Isolate velocity portions of previously mentioned CW equations
		velocityMatrix = sys_mat[0:3, 3:6]
		
		#Save difference of desired velocity and current velocity to represent required burn magnitude
		result = np.linalg.solve(velocityMatrix, constants) - self.init_state[3:6, 0].reshape(cfgs.XYZ_DIM)
		
		#Update the model if no user override for solution only
		if applyResult:
			self.applyVelocityChange(result)
			
		return result
	
	def applyVelocityChange(self, velocityArray):
		"""
		Updates the current relative state by applying a velocity change

		@type velocityArray:	numpy array
		@param velocityArray:	Desired positional offset to achieve
		@rtype:					None
		@return:				N/A
		"""
		
		#Verify input numpy array is appropriate dimensions
		assert velocityArray.shape == cfgs.XYZ_DIM, e.unexpectedArrayDim(velocityArray, cfgs.XYZ_DIM)
		
		#Pad the input velocity array with zeros to represent a 6-dimensional state change vector
		StateChangeArray = np.pad(velocityArray, [(3, 0), (0, 0)], mode='constant')
		
		#Update current relative state with velocity change incorporated
		self.init_state = self.init_state + StateChangeArray
		
	def getPositionsOverInterval(self, t_i, t_f, dt=cfgs.DEFAULT_STEP):
		"""
		Returns x, y, and z positions at given timestep over desired interval

		@type t_i:				int
		@param t_i:				Starting time offset of propagation interval
		@type t_f:				int
		@param t_f:				Ending time offset of propagation interval
		@type dt:				int
		@param dt:				Ending time offset of propagation interval
		@rtype:					tuple of lists
		@return:				x, y, and z positions in respective lists
		"""
		#Verify arguments are all integers
		assert type(t_i) == int, e.unexpectedType(t_i, int)
		assert type(t_f) == int, e.unexpectedType(t_f, int)
		assert type(dt) == int, e.unexpectedType(dt, int)
		
		#Initialize empty lists to hold x, y, and z values of positions
		x, y, z = [], [], []
		
		#Loop through user-input timeframe and append position components to appropriate lists
		for t in range(t_i, t_f, dt):
			state = self.solveNextState(t)
			x.append(state[0][0])
			y.append(state[1][0])
			z.append(state[2][0])
			
		return x, y, z

	def getPeriod(self):
		"""
		Return the number of seconds required to complete a full period

		@rtype:		number
		@return:	number of seconds for full revolution of model
		"""

		return 2*pi/self.n

	def stepToNextTangent(self):
		"""
		Step the model forward to the next radial tangent

		@rtype:					number
		@return:				number of seconds the model was propagated
		"""
		
		#Save key components from current initial state
		x = self.init_state[0][0]
		x_dot = self.init_state[3][0]
		y_dot = self.init_state[4][0]
		
		#Solve for the time of nearest x_dot==0 occurence
		t = atan(-x_dot/(3*self.n*x + 2*y_dot))/self.n
		
		#Correct t by half period for any negative or 0 solutions
		if t <= cfgs.ZERO_TOL: 
			t += self.getPeriod()/2
		elif isnan(t):
			t = self.getPeriod()/2
		
		#Update t0_state to reflect new tangential position
		self.init_state = self.solveNextState(t)
		
		return t
		
	def solveBurnToRadialWP(self, radialTarget, applyResult=True):
		"""
		Solve the y_dot magnitude required to efficiently achieve a radial target

		@type radialTarget:		number
		@param radialTarget:	the desired radius relative to the target .5 period later in meters
		@rtype:					numpy array
		@return:				Velocity difference required to achieve position waypoint
		"""
		
		#Verify the model is already at a tangential position
		y_dot_mag = abs(self.init_state[3][0])
		assert y_dot_mag < cfgs.ZERO_TOL, e.outOfTolerance(y_dot_mag, cfgs.ZERO_TOL)
		
		#Save key components from current initial state
		x = self.init_state[0][0]
		x_dot = self.init_state[3][0]
		y_dot = self.init_state[4][0]
		n = self.n
		t = self.getPeriod()/2
		
		#Store the numerator and denomenator separately to solve for required y_dot difference
		#This is an algebraic solution of equation 1 in the CW matrix
		num = radialTarget - (4-3*cos(n*t))*x - (sin(n*t)/n)*x_dot
		den = 2*(1-cos(n*t))/n
		
		#Save velocity difference of current vs desired
		result = np.array([0, (num/den)-y_dot, 0]).reshape(cfgs.XYZ_DIM)
		
		#Update the model if no user override for solution only
		if applyResult:
			self.applyVelocityChange(result)
			
		return result
		
		
	def solveCircularDriftProfileToMatchState(self, wpTimeFromNow):
		"""
		Solve the four-burn sequence that creates a circular phasing altitude (two burns) relative to the desired orbit state, ingresses to a 0 waypoint (1 burn) then
		matches velocity at the input waypoint time (1 burn).  This method should only be used when starting at an altitude tangent.  Otherwise, the assumptions used
		to simplify the algebra will be invalid.

		@type wpTimeFromNow:	number
		@param wpTimeFromNow:	The time of the last velocity-targeting burn in seconds from now
		@rtype:					tuple
		@return:				The in-track burn magnitudes of each burn in the order (phasing burn 1, recirc burn, ingress burn, velocity match burn)
		"""

		#Store key constants to reduce clutter in system matrix
		n = self.n
		t_hp = self.getPeriod()/2
		t_Tp = wpTimeFromNow - self.getPeriod()
		xi = self.init_state[0][0]
		yi = self.init_state[1][0]

		#Save system matrix and solution vectors.  These are formed by using relationships of x(t) and y(t) CW equations
		soln_vector = np.array([[7*xi, 0, 0, -6*xi*n*t_hp + yi, 0, 0, 0]]).reshape((7,1))
		sys_mat = np.array([
					[0, 0, 0, 1, -4/n, 0, 0],
					[0, 0, 0, -6, 0, -4/n, 0],
					[0, 0, 0, 1, 0, 0, -4/n],
					[1, 0, 0, 0, 3*t_hp, 0, 0],
					[-1, 1, 0, -6*(sin(n*t_Tp)-n*t_Tp), 0, -(4*sin(n*t_Tp) - 3*n*t_Tp)/n, 0],
					[0, 0, 1, 0, 0, 0, -3*t_hp],
					[0, 1, -1, 0, 0, 0, 0]
		])
		
		#Find the results for [y1, y2, y3, x_phasing, y_dot1, y_dot2, y_dot3]
		result = np.dot(np.linalg.inv(sys_mat), soln_vector)

		#Burn #1 is the difference of the desired y_dot1 value and the current model y_dot value
		burn1 = RelativeBurn(np.array([0, (result[4][0] - self.init_state[4][0]), 0]).reshape(cfgs.XYZ_DIM))

		#Copy the current model to perform maneuvers and solve the remaining burns
		phaseModel = deepcopy(self)

		#Maneuver the model into the pre-circular phasing orbit
		phaseModel.applyVelocityChange(burn1.getBurnArray())

		#Step to the next tangent to recircularize at the desired altitude
		phaseModel.stepToNextTangent()

		#Burn #2 is the difference of the desired y_dot2 value and the current phasing model y_dot value
		burn2 = RelativeBurn(np.array([0, (result[5][0] - phaseModel.init_state[4][0]), 0]).reshape(cfgs.XYZ_DIM))

		#Maneuver the model to achieve a circular phasing orbit
		phaseModel.applyVelocityChange(burn2.getBurnArray())

		#Create an ingress model formed at a 0 waypoint
		ingressModel = RelativeModel(np.asarray([[0,0,0,0,result[6][0], 0]]).reshape(cfgs.REL_STATE_DIM), self.a)

		#Step backwards half a period to the ingress point
		state = ingressModel.solveNextState(-ingressModel.getPeriod()/2)

		#Step the phasing model forward to achieve the same position as the ingress model
		state2 = phaseModel.solveNextState(t_Tp - t_hp)

		#Burn #3 is the difference of the ingress model y_dot and the phasing model y_dot value
		burn3 = RelativeBurn(np.array([0, state[4][0] - state2[4][0], 0]).reshape(cfgs.XYZ_DIM))

		#Burn #4 is the anti-vector of the solved y_dot3 value
		burn4 = RelativeBurn(np.array([0, -result[6][0], 0]).reshape(cfgs.XYZ_DIM))

		return burn1, burn2, burn3, burn4, t_Tp
		
	def solveEccentricDriftProfileToMatchState(self, wpTimeFromNow):
		"""
		Solve the three-burn sequence that phases (1 burn), ingresses (1 burn), then matches velocity BEFORE the input waypoint time (1 burn).  This method should only 
		be used when starting at an altitude tangent.  Otherwise, the assumptions used to simplify the algebra will be invalid.  The final burn time will be chose based on 
		the number of full periods between now and the input waypoint time

		@type wpTimeFromNow:	number
		@param wpTimeFromNow:	The latest time of the last velocity-targeting burn in seconds from now
		@rtype:					tuple
		@return:				The in-track burn magnitudes of each burn in the order (phasing burn 1, recirc burn, ingress burn, velocity match burn)
		"""

		#Store key constants to reduce clutter in system matrix
		n = self.n
		t_hp = self.getPeriod()/2
		
		t_Tp = (wpTimeFromNow - (wpTimeFromNow % self.getPeriod())) - t_hp
		xi = self.init_state[0][0]
		yi = self.init_state[1][0]

		#Save system matrix and solution vectors.  These are formed by using relationships of x(t) and y(t) CW equations
		soln_vector = np.array([[7*xi, 0, -6*xi*n*t_hp + yi, 0, -6*xi*n*t_Tp + yi, 0]]).reshape((6,1))
		sys_mat = np.array([
					[0, 0, 0, 1, -4/n, 0],
					[0, 0, 0, 1, 0, -4/n], 
					[1, 0, 0, 0, 3*t_hp, 0],
					[0, 1, 0, 0, 0, -3*t_hp],
					[0, 0, 1, 0, 3*t_Tp, 0],
					[0, -1, 1, 0, 0, 0]
		])
		
		#Find the results for [y1, y2, y3, x_phasing, y_dot1, y_dot2, y_dot3]
		result = np.dot(np.linalg.inv(sys_mat), soln_vector)

		#Burn #1 is the difference of the desired y_dot1 value and the current model y_dot value
		burn1 = RelativeBurn(np.array([0, (result[4][0] - self.init_state[4][0]), 0]).reshape(cfgs.XYZ_DIM))

		#Copy the current model to perform maneuvers and solve the remaining burns
		phaseModel = deepcopy(self)

		#Maneuver the model into the phasing orbit
		phaseModel.applyVelocityChange(burn1.getBurnArray())

		#Create an ingress model formed at a 0 waypoint
		ingressModel = RelativeModel(np.asarray([[0,0,0,0,result[5][0], 0]]).reshape(cfgs.REL_STATE_DIM), self.a)

		#Step backwards half a period to the ingress point
		state = ingressModel.solveNextState(-ingressModel.getPeriod()/2)

		#Step the phasing model forward to achieve the same position as the ingress model
		state2 = phaseModel.solveNextState(t_Tp)

		#Burn #2 is the difference of the ingress model y_dot and the phasing model y_dot value
		burn2 = RelativeBurn(np.array([0, state[4][0] - state2[4][0], 0]).reshape(cfgs.XYZ_DIM))

		#Burn #3 is the anti-vector of the solved y_dot3 value
		burn3 = RelativeBurn(np.array([0, -result[5][0], 0]).reshape(cfgs.XYZ_DIM))

		return burn1, burn2, burn3, t_Tp	

class RelativeBurn:
	"""
	This class holds the x, y, z components of a velocity change relative to a spacecraft.
	"""
	def __init__(self, burnArray):
		"""
		Stores a burn array for convenient calling of magnitudes
		
		@type burnArray:  	numpy array of dimensions (3, 1)
		@param burnArray: 	[x_dot, y_dot, z_dot] in meters per second respectively.  
                    
		@rtype:         	None
		@return:        	N/A
		"""
		self.burnArray = deepcopy(burnArray)

	def getRadialVelocity(self):
		"""
		Gets the radial (x) component of the burn object
                    
		@rtype:         	number
		@return:        	x component of the burn array
		"""
		return self.burnArray[0][0]

	def getInTrackVelocity(self):
		"""
		Gets the in-track (y) component of the burn object
                    
		@rtype:         	number
		@return:        	y component of the burn array
		"""
		return self.burnArray[1][0]

	def getCrossTrackVelocity(self):
		"""
		Gets the cross-track (z) component of the burn object
                    
		@rtype:         	number
		@return:        	z component of the burn array
		"""
		return self.burnArray[2][0]

	def getBurnArray(self):
		"""
		Gets the array holding all burn compenents
                    
		@rtype:         	numpy array of dimensions (3, 1)
		@return:        	burn array of x,y,z components
		"""
		return self.burnArray