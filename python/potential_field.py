from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np
import random


WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([.0, .5], dtype=np.float32)
CYLINDER_RADIUS = .3
CYLINDER_POSITION_1 = np.array([.5, .0], dtype=np.float32)
CYLINDER_RADIUS_1 = .3
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([-1.5, -1.5], dtype=np.float32)
MAX_SPEED = .5


def get_velocity_to_reach_goal(position, goal_position):
  # Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.
  v = goal_position - position
  magnitude = np.linalg.norm(v)
  v = cap(v, MAX_SPEED)

# (d) random noise to offset 
#  return v + np.random.uniform(-0.01, 0.01, 2)
  return v


def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  # Compute the velocity field needed to avoid the obstacles
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). Make sure to not create
  # speeds that are larger than max_speed for each obstacle. Both obstacle_positions
  # and obstacle_radii are lists.
  v = np.zeros(2, dtype=np.float32)

  for obs, rad in zip(obstacle_positions, obstacle_radii):
    vel = obs - position

    if (rad == 0):
    # virtual obstacle (reduce effect) 
      speed = min(0.1/np.linalg.norm(vel), 0.5*MAX_SPEED)
    else:
      if np.linalg.norm(vel) <= rad:
        speed = MAX_SPEED
      elif np.linalg.norm(vel) >= 4*rad: 
        speed = 0.0   
      else:
        speed = min(0.25/np.linalg.norm(vel), MAX_SPEED)
    v -= normalize(vel) * speed

  return v


def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
    #  [CYLINDER_POSITION] + extra,
    #  [CYLINDER_RADIUS]+ extra_rad)
      [CYLINDER_POSITION, CYLINDER_POSITION_1] + extra,
      [CYLINDER_RADIUS, CYLINDER_RADIUS_1] + extra_rad)
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid

  return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()
  eps = 0.01
  extra = []
  extra_rad = []

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
  ax.add_artist(plt.Circle(CYLINDER_POSITION_1, CYLINDER_RADIUS_1, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.025
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    if np.linalg.norm(v) < eps and np.linalg.norm(x - GOAL_POSITION) > 5*eps:
      # Get the direction the robot is travelling in
      d = x - positions[max(0, len(positions)-10)] 
      direction = d/np.linalg.norm(d) + np.random.uniform(-1, 1, 2)
      print(x, x + direction*0.05)

      # Generate virtual obstacle in the direction of travel
      extra.append(x + direction*0.05)
      extra_rad.append(0)

      # recalculate velocity
      print(v)
      v = get_velocity(x, args.mode)
      print(v)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()