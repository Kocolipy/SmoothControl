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
X = Constants.X
Y = Constants.Y
YAW = Constants.YAW

ROBOT_RADIUS = Constants.ROBOT_RADIUS
MAX_ITERATIONS = Constants.MAX_ITERATIONS

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
    return node, 0.0

  # check if final_position is free
  if not occupancy_grid.is_free(final_position):
    return None, 0.0

  # Find the resulting yaw
  dir = final_position - node.position
  rightvec = np.array((-node.direction[Y], node.direction[X]))
  alpha = np.arccos(np.clip(node.direction.dot(dir) / np.linalg.norm(dir), -1, 1))
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

  steps = 100
  for angle in np.linspace(beta_0, beta_1, steps):
    point = c + np.array([np.cos(angle), np.sin(angle)])*rad
    if not occupancy_grid.is_free(point):
      return None, 0.0

  # Calculate arc length
  length = rad * abs(beta_0 - beta_1)

  return final_node, length


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
  for iter in range(MAX_ITERATIONS): 
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
    lowest_cost = None
    v = None
    for n, d in potential_parent:
      if d > .2 and d < 1.5 and n.direction.dot(position - n.position) / d > 0.70710678118:
        next, l = adjust_pose(n, position, occupancy_grid)
        if next is None:
          continue
        
        if lowest_cost is None:
          lowest_cost = n.cost + l
          u = n
          v = next

        if n.cost + l < lowest_cost:
          lowest_cost = n.cost + l
          u = n
          v = next 
    if v is None:
      continue

    u.add_neighbor(v)
    v.parent = u
    v.cost = lowest_cost
    graph.append(v)

    # Rewiring the tree
    # Search for nodes which are close to v
    potential_parent = sorted(((n, np.linalg.norm(v.position - n.position)) for n in graph), key=lambda x: x[1])
    for n, d in potential_parent:
      if d > .2 and d < 1.5 and v.direction.dot(n.position - v.position) / d > 0.70710678118:
        nn, arc = adjust_pose(v, n.position, occupancy_grid)
        
        if nn is None:
          continue
        
        if v.cost + arc < n.cost:
          v.add_neighbor(nn)
          nn.cost = v.cost + arc
          nn.parent = v
          n.parent.neighbors.remove(n)
          graph.remove(n)
          graph.append(nn)

    if np.linalg.norm(v.position - goal_position) < .2:
      if final_node is None:
        final_node = v
      elif v.cost < final_node.cost: 
          final_node = v

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

def draw_solution(axis, start_node, final_node=None):
  def draw_path(u, v, arrow_length=.1, color=(.8, .8, .8), lw=1):
    du = u.direction
    axis.arrow(u.pose[X], u.pose[Y], du[0] * arrow_length, du[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    dv = v.direction
    axis.arrow(v.pose[X], v.pose[Y], dv[0] * arrow_length, dv[1] * arrow_length,
              head_width=.05, head_length=.1, fc=color, ec=color)
    center, radius = find_circle(u, v)
    du = u.position - center
    theta1 = np.arctan2(du[1], du[0])
    dv = v.position - center
    theta2 = np.arctan2(dv[1], dv[0])
    # Check if the arc goes clockwise.
    if np.cross(u.direction, du).item() > 0.:
      theta1, theta2 = theta2, theta1
    axis.add_patch(patches.Arc(center, radius * 2., radius * 2.,
                             theta1=theta1 / np.pi * 180., theta2=theta2 / np.pi * 180.,
                             color=color, lw=lw))

  points = []
  s = [(start_node, None)]  # (node, parent).
  while s:
    v, u = s.pop()
    if hasattr(v, 'visited'):
      continue
    v.visited = True
    # Draw path from u to v.
    if u is not None:
      draw_path(u, v)
    points.append(v.pose[:2])
    for w in v.neighbors:
      s.append((w, v))

  points = np.array(points)
  axis.scatter(points[:, 0], points[:, 1], s=10, marker='o', color=(.8, .8, .8))
  if final_node is not None:
    axis.scatter(final_node.position[0], final_node.position[1], s=10, marker='o', color='k')
    # Draw final path.
    v = final_node
    while v.parent is not None:
      draw_path(v.parent, v, color='k', lw=2)
      v = v.parent


if __name__ == '__main__':
  maps = ["standard_map", "narrow_map", "arc_obs", "bubbles", "cage", "maze", "circle"]

  fig, ax = plt.subplots(1, len(maps))
  for i, map in enumerate(maps):
    occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(map)

    # Run RRT.
    start_node, final_node = rrt(start_pose, goal_pose, occupancy_grid)

    # Plot environment.
    occupancy_grid.draw(ax[i])
    draw_solution(ax[i], start_node, final_node)
    ax[i].scatter(start_pose[0], start_pose[1], s=10, marker='o', color='green', zorder=1000)
    ax[i].scatter(goal_pose[0], goal_pose[1], s=10, marker='o', color='red', zorder=1000)

    ax[i].axis('equal')
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    ax[i].set_xlim([-2.5, 2.5])
    ax[i].set_ylim([-2.5, 2.5])
  mng = plt.get_current_fig_manager()
  mng.window.state("zoomed")
  plt.show()

