import numpy as np
REL_STATE_DIM = (6, 1) #The expected shape of the numpy array used in the relative motion model
REL_TEST_STATE = np.array([[-5000,0,0,0,0,0]]).reshape(REL_STATE_DIM) #Test state in m and m/s for relative motion
DEFAULT_STEP = 600 #Default number of seconds to propagate spacecraft