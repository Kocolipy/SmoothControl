from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import os
import re
import scipy.signal
import yaml

import Constants
import OccupancyGrid

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2

ROBOT_RADIUS = 0.105 / 2.
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)  # Any orientation is good.
START_POSE = np.array([-1.5, -1.5, 0.], dtype=np.float32)
MAX_ITERATIONS = 10000


def sample_random_position(occupancy_grid):
  # Sample a valid random position (do not sample the yaw).
  # The corresponding cell must be free in the occupancy grid.
  maxx = occupancy_grid.origin[0] + occupancy_grid.resolution * occupancy_grid.values.shape[0]
  maxy = occupancy_grid.origin[1] + occupancy_grid.resolution * occupancy_grid.values.shape[1]

  position = np.zeros(2, dtype=np.float32)
  position[0] = np.random.uniform(occupancy_grid.origin[0], maxx)
  position[1] = np.random.uniform(occupancy_grid.origin[1], maxy)
 
  while not occupancy_grid.is_free(position):
    position[0] = np.random.uniform(occupancy_grid.origin[0], maxx)
    position[1] = np.random.uniform(occupancy_grid.origin[1], maxy)

  return position


def adjust_pose(node, final_position, occupancy_grid): 
  # Check whether there exists a simple path that links node.pose
  # to final_position. This function needs to return a new node that has
  # the same position as final_position and a valid yaw. The yaw is such that
  # there exists an arc of a circle that passes through node.pose and the
  # adjusted final pose. If no such arc exists (e.g., collision) return None.
  # Assume that the robot always goes forward.
  # Feel free to use the find_circle() function below.

  if np.array_equal(node.position, final_position):
    return node

  # check if final_position is free
  if not occupancy_grid.is_free(final_position):
    return None

  # Find the resulting yaw
  dir = final_position - node.position
  rightvec = np.array((-node.direction[Y], node.direction[X]))
  alpha = np.arccos(node.direction.dot(dir) / np.linalg.norm(dir))
  sign = 1 if dir.dot(rightvec) >= 0 else -1
  yaw = node.yaw + 2*sign*alpha

  final_pose = node.pose.copy()
  final_pose[:2] = final_position
  final_pose[2] = yaw
  final_node = Node(final_pose)

  c, rad = find_circle(node, final_node)
  # Get angles in circle 
  dir = node.position - c
  sign = 1 if dir.dot(np.array((0, 1))) >= 0 else -1
  beta_0 = sign*np.arccos(dir.dot(np.array((1, 0))) / np.linalg.norm(dir))

  dir = final_node.position - c
  sign = 1 if dir.dot(np.array((0, 1))) >= 0 else -1
  beta_1 = sign*np.arccos(dir.dot(np.array((1, 0))) / np.linalg.norm(dir))

  steps = 10
  for angle in np.linspace(beta_0, beta_1, steps):
    point = c + np.array([np.cos(angle), np.sin(angle)])*rad
    if not occupancy_grid.is_free(point):
      return None

  return final_node

# Defines an occupancy grid.
# class OccupancyGrid(object):
#   def __init__(self, values, origin, resolution):
#     self._original_values = values.copy()
#     self._values = values.copy()
#     # Inflate obstacles (using a convolution).
#     inflated_grid = np.zeros_like(values)
#     inflated_grid[values == OCCUPIED] = 1.
#     w = 2 * int(ROBOT_RADIUS / resolution) + 1
#     inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
#     self._values[inflated_grid > 0.] = OCCUPIED
#     self._origin = np.array(origin[:2], dtype=np.float32)
#     self._origin -= resolution / 2.
#     assert origin[YAW] == 0.
#     self._resolution = resolution
#
#   @property
#   def values(self):
#     return self._values
#
#   @property
#   def resolution(self):
#     return self._resolution
#
#   @property
#   def origin(self):
#     return self._origin
#
#   def draw(self):
#     plt.imshow(self._original_values.T, interpolation='none', origin='lower',
#                extent=[self._origin[X],
#                        self._origin[X] + self._values.shape[0] * self._resolution,
#                        self._origin[Y],
#                        self._origin[Y] + self._values.shape[1] * self._resolution])
#     plt.set_cmap('gray_r')
#
#   def get_index(self, position):
#     idx = ((position - self._origin) / self._resolution).astype(np.int32)
#     if len(idx.shape) == 2:
#       idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
#       idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
#       return (idx[:, 0], idx[:, 1])
#     idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
#     idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
#     return tuple(idx)
#
#   def get_position(self, i, j):
#     return np.array([i, j], dtype=np.float32) * self._resolution + self._origin
#
#   def is_occupied(self, position):
#     return self._values[self.get_index(position)] == OCCUPIED
#
#   def is_free(self, position):
#     return self._values[self.get_index(position)] == FREE


# Defines a node of the graph.
class Node(object):
  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._cost = 0.

  @property
  def pose(self):
    return self._pose

  def add_neighbor(self, node):
    self._neighbors.append(node)

  @property
  def parent(self):
    return self._parent

  @parent.setter
  def parent(self, node):
    self._parent = node

  @property
  def neighbors(self):
    return self._neighbors

  @property
  def position(self):
    return self._pose[:2]

  @property
  def yaw(self):
    return self._pose[YAW]
  
  @property
  def direction(self):
    return np.array([np.cos(self._pose[YAW]), np.sin(self._pose[YAW])], dtype=np.float32)

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, c):
    self._cost = c


def rrt(start_pose, goal_position, occupancy_grid):
  # RRT builds a graph one node at a time.
  graph = []
  start_node = Node(start_pose)
  final_node = None
  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return start_node, final_node
  graph.append(start_node)
  while True:
    position = sample_random_position(occupancy_grid)
    # With a random chance, draw the goal position.
    if np.random.rand() < .05:
      position = goal_position
    # Find closest node in graph.
    # In practice, one uses an efficient spatial structure (e.g., quadtree).
    potential_parent = sorted(((n, np.linalg.norm(position - n.position)) for n in graph), key=lambda x: x[1])
    # Pick a node at least some distance away but not too far.
    # We also verify that the angles are aligned (within pi / 4).
    u = None
    for n, d in potential_parent:
      if d > occupancy_grid.resolution*2 and d < occupancy_grid.resolution*5 and n.direction.dot(position - n.position) / d > 0.70710678118:
        u = n
        break
    else:
      continue
    v = adjust_pose(u, position, occupancy_grid)
    if v is None:
      continue
    u.add_neighbor(v)
    v.parent = u
    graph.append(v)
    if np.linalg.norm(v.position - goal_position) < occupancy_grid.resolution:
      final_node = v
      break
  return start_node, final_node


def find_circle(node_a, node_b):
  def perpendicular(v):
    w = np.empty_like(v)
    w[X] = -v[Y]
    w[Y] = v[X]
    return w
  db = perpendicular(node_b.direction)
  dp = node_a.position - node_b.position
  t = np.dot(node_a.direction, db)
  if np.abs(t) < 1e-3:
    # By construction node_a and node_b should be far enough apart,
    # so they must be on opposite end of the circle.
    center = (node_b.position + node_a.position) / 2.
    radius = np.linalg.norm(center - node_b.position)
  else:
    radius = np.dot(node_a.direction, dp) / t
    center = radius * db + node_b.position
  return center, np.abs(radius)


def draw_solution(start_node, final_node=None):
  ax = plt.gca()

  def draw_path(u, v, arrow_length=.01, color=(.8, .8, .8), lw=1):
    du = u.direction
    plt.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.005, head_length=.01, fc=color, ec=color)
    dv = v.direction
    plt.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.005, head_length=.01, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    ax.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))
    return abs((theta2 - theta1) * radius)

  # points = []
  # s = [(start_node, None)]  # (node, parent).
  # while s:
  #   v, u = s.pop()
  #   if hasattr(v, 'visited'):
  #     continue
  #   v.visited = True
  #   # Draw path from u to v.
  #   if u is not None:
  #     draw_path(u, v)
  #   points.append(v.pose[:2])
  #   for w in v.neighbors:
  #     s.append((w, v))
  #
  # points = np.array(points)
  # plt.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  length = 0
  if final_node is not None:
    plt.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      d = draw_path(v.parent, v, color='k', lw=2)
      v = v.parent
      length += d
  return length


if __name__ == '__main__':
  # maps = ["standard_map", "narrow_map", "arc_obs", "bubbles", "cage", "maze", "circle"]
  maps = ["arc_obs"]

  fig, ax = plt.subplots(1, len(maps), squeeze=False)
  for i, map in enumerate(maps):
    axis = ax[0, i]
    occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(map)

    # Run RRT.
    start_node, final_node = rrt(start_pose, goal_pose, occupancy_grid)

    # Plot environment.
    occupancy_grid.draw(axis, path=True, grid=True)
    axis.scatter(start_pose[0], start_pose[1], s=10, marker='o', color='green', zorder=1000)
    axis.scatter(goal_pose[0], goal_pose[1], s=10, marker='o', color='red', zorder=1000)
    dist = draw_solution(start_node, final_node)
    axis.axis('equal')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_xlim([-1.25, 1.25])
    axis.set_ylim([-1.25, 1.25])
  mng = plt.get_current_fig_manager()
  mng.window.state("zoomed")
  plt.show()