import numpy as np
REL_STATE_DIM = (6, 1) #The expected shape of the numpy array used in the relative motion model
REL_TEST_STATE = np.array([[-100000,-7000000,0,1,11,0]]).reshape(REL_STATE_DIM) #Test state in m and m/s for relative motion
DEFAULT_STEP = 600 #Default number of seconds to propagate spacecraft
XYZ_DIM = (3, 1) #Used to validate shape of 3-dimensional numpy arrays
ZERO_TOL = 1e-6 #The allowable tolerance to consider a number zero