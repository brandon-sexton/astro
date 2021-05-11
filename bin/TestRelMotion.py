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
	relmod.stepToNextTangent()
	b1, b2, b3, t = relmod.solveEccentricDriftProfileToMatchState(SECONDS_IN_DAY*10)

	relmod.applyVelocityChange(b1.getBurnArray())
	r, i, c = relmod.getPositionsOverInterval(0, int(t)+SECONDS_IN_DAY)
	plt.plot(np.array(i)/1000, np.array(r)/1000, "r-")
	
	relmod = RelativeModel(relmod.solveNextState(t), GEO_RADIUS*METERS_IN_KM)
	relmod.applyVelocityChange(b2.getBurnArray())
	r, i, c = relmod.getPositionsOverInterval(0, SECONDS_IN_DAY)
	plt.plot(np.array(i)/1000, np.array(r)/1000, "g-")

	plt.show()
	
if __name__=="__main__":
	test()