from copy import deepcopy

class GeoRelativeModel():
    """
    This class performs calculations of relative satellite positions using Clohessy-Wiltshire
    equations defined on page X of 'Fundamentals of Astrodynamics and Applications' by David Vallado.
    """
    
    def __init__(self, t0_state):
        """
        Defines starting state for two spacecraft given the state of chase relative to target at time == 0
        
        @type t0_state: numpy array of dimensions [1, 6] as [x, y, z, x_dot, y_dot, z_dot] in meters
                        and meters per second respectively.  The values x, y, z represent radial, in-track,
                        and cross-track.
                        
        @rtype:         None
        @return:        N/A
        """
        self.init_state = deepcopy(t0_state)
    
    def solveNextState(self, t):
		a = GEO_RADIUS*M_IN_KM
		n = sqrt((EARTH_MU/a**3))
		sys_mat = [
					[4-3*cos(n*t), 0, 0, sin(n*t)/n, 2*(1-cos(n*t))/n, 0],
					[6*(sin(n*t) - n*t), 1, 0, -2*(1-cos(n*t))/n, (4*sin(n*t) - 3*n*t)/n, 0],
					[0, 0, cos(n*t), 0, 0, sin(n*t)/n],
					[3*n*sin(n*t), 0, 0, cos(n*t), 2*sin(n*t), 0],
					[-6*n*(1-cos(n*t)), 0, 0, -2*sin(n*t), 4*cos(n*t)-3, 0],
					[0, 0, -n*sin(n*t), 0, 0, cos(n*t)]]
					
		new_state = []
		for row in sys_mat:
			sum = 0
			for i in range(len(self.init_state)):
				sum+=row[i]*self.init_state[i]
				
			new_state.append(sum)
			
		return new_state
