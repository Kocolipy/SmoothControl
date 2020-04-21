import numpy as np
# Constants used for indexing.
X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
TURN_RADIUS = 0.2
COST_FACTOR = TURN_RADIUS/np.sin(TURN_RADIUS)

CURVATURE = 1

NUM_POINTS = 100