import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
sys.path.append(os.getcwd())

from lib.relative_motion import RelativeModel
import lib.SysConfigs as cfgs
from lib.constants import GEO_RADIUS, METERS_IN_KM, SECONDS_IN_DAY

def test():
	relmod = RelativeModel(cfgs.REL_TEST_STATE, GEO_RADIUS*METERS_IN_KM)
	r, i, c = relmod.getPositionsOverInterval(0, SECONDS_IN_DAY)
	plt.plot(np.array(i)/1000, np.array(r)/1000, "b-")
	
	relmod.stepToNextTangent()
	burn = relmod.solveBurnToRadialWP(-45000)
	r, i, c = relmod.getPositionsOverInterval(0, SECONDS_IN_DAY)
	plt.plot(np.array(i)/1000, np.array(r)/1000, "r-")
	
	relmod.stepToNextTangent()
	burn = relmod.solveBurnToRadialWP(-45000)
	r, i, c = relmod.getPositionsOverInterval(0, SECONDS_IN_DAY)
	plt.plot(np.array(i)/1000, np.array(r)/1000, "g-")

	plt.show()
	
if __name__=="__main__":
	test()