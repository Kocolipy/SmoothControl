from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import os
import pickle
import sys

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
    import Spline
except ImportError as e:
    raise ImportError('Unable to import Spline.py. Make sure this file is in "{}"'.format(directory))


if __name__ == '__main__':
  leadlag_p = np.genfromtxt('/tmp/leadlag_path.txt', delimiter=',')
  leadlag_e = np.genfromtxt('/tmp/leadlag.txt', delimiter=',')
  pid_p = np.genfromtxt('/tmp/pid_path.txt', delimiter=',')
  pid_e = np.genfromtxt('/tmp/pid.txt', delimiter=',')
  lead_p = np.genfromtxt('/tmp/lead_path.txt', delimiter=',')
  lead_e = np.genfromtxt('/tmp/lead.txt', delimiter=',')
  splines = pickle.load(open('/tmp/planned_path.txt', "rb"))
  path = splines[:]
  
  full_path = []
  for spline in path:
    full_path += spline.discretise()
  full_path = np.array(full_path)  

  plt.figure()
  plt.plot(leadlag_p[:, 0], leadlag_p[:, 1], 'g', lw=1, label='Lead Lag')
  plt.plot(lead_p[:, 0], lead_p[:, 1], 'r',  lw=1, label='Lead')
  plt.plot(pid_p[:, 0], pid_p[:, 1], 'b',  lw=1, label='PID')
  plt.plot(full_path[:, 0], full_path[:, 1],   'black', lw=1, label='Planned')
  # Cylinder.
  a = np.linspace(0., 2 * np.pi, 20)
  x = np.cos(a) * .3 + .3
  y = np.sin(a) * .3 + .2
  plt.plot(x, y, 'k')
  # Walls.
  plt.plot([-2, 2], [-2, -2], 'k')
  plt.plot([-2, 2], [2, 2], 'k')
  plt.plot([-2, -2], [-2, 2], 'k')
  plt.plot([2, 2], [-2, 2], 'k')
  plt.axis('equal')
  plt.legend()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-2.5, 2.5])
  plt.ylim([-2.5, 2.5])

  plt.figure()
  plt.plot([0]*450, c='black', lw=1)
  plt.plot(leadlag_e, c='g', lw=1, label="Lead-Lag")
  plt.plot(pid_e, c='b', lw=1, label = "PID")
  plt.plot(lead_e, c='r', lw=1, label = "Lead")
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=3)
  plt.ylabel('Error [m]')
  plt.xlabel('Timestep')
plt.show()

    
   