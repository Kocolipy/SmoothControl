from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import matplotlib.pylab as plt
import matplotlib.patches as patches
from sortedcontainers import SortedKeyList
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

# Defines a node of the graph.
class Node(object):
  occupancy_grid = None
  goal = None;

  def __init__(self, pose):
    self._pose = pose.copy()
    self._neighbors = []
    self._parent = None
    self._pathcost = 0.

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
    return self._pathcost + np.sqrt(sum((Node.goal - self.position)**2))

  @property
  def pathcost(self):
    return self._pathcost

  @pathcost.setter
  def pathcost(self, c):
    self._pathcost = c

  @property
  def indices(self):
    return occupancy_grid.get_index(self.position)

  def __eq__(self, other):
    return self.indices == other.indices

def astar(start_pose, goal_position, occupancy_grid):
  # Open list contain nodes to be checked while closed list are inspected nodes
  open_list = SortedKeyList(key=lambda x: x.cost)

  # Define goal position and occupancy grid for all Nodes
  Node.occupancy_grid = occupancy_grid
  Node.goal = occupancy_grid.get_position(*occupancy_grid.get_index(goal_position))

  start_node = Node(start_pose)
  final_node = Node(goal_position)

  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return None
  explored = 0
  open_list.add(start_node)
  while open_list:
    current_node = open_list.pop(0)

    if occupancy_grid.is_explored(occupancy_grid.get_index(current_node.position)):
      continue

    occupancy_grid.set_explored(current_node.indices)
    explored += 1
    # if explored % 100 == 0:
    #   print(explored)
    #   occupancy_grid.sketch()

    # Found the goal
    if current_node == final_node:
      path = []
      current = current_node
      while current is not None:
        path.append(current.position)
        occupancy_grid.set_path(current.indices)
        current = current.parent
      return path

    # Generate children
    children = []
    for step in [(0, -1), (0, 1), (-1, 0), (1, 0), (1,1), (-1, 1), (1, -1), (-1, -1)]:  # Adjacent paths
      node_position = (current_node.indices[0] + step[0], current_node.indices[1] + step[1])

      if not occupancy_grid.is_free_index(node_position):
        continue

      new_node = Node(occupancy_grid.get_position(*node_position))
      new_node.parent = current_node

      new_node.pathcost = current_node.pathcost + np.sqrt(sum([abs(x) for x in step]))*occupancy_grid.resolution
      children.append(new_node)

    for child in children:
      # Check if new node is already in the open list
      to_add = True
      for i, open_node in enumerate(open_list):
        if child == open_node:
          to_add = False
          if child.pathcost < open_node.pathcost:
            del open_list[i]
            open_list.add(child)
          break

      # Add new node to the open list
      if to_add:
        open_list.add(child)

  return None


if __name__ == '__main__':
  # maps = ["standard_map", "narrow_map", "arc_obs", "bubbles", "cage", "maze", "circle"]
  maps = ["arc_obs"]

  fig, ax = plt.subplots(1, len(maps), squeeze=False)
  for i, map in enumerate(maps):
    axis = ax[0, i]

    import timeit
    timings = []
    for i in range(100):
      if (i + 1) % 10 == 0:
        print((i + 1), "executions timed.")
      occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(map)

      def testfunc():
        return astar(start_pose, goal_pose, occupancy_grid)

      a = timeit.timeit(testfunc, number=1)
      timings.append(a)
    timings = np.array(timings)
    print(map)
    print("Mean:", timings.mean())
    print("Std:", timings.std())

    continue
    occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(map)

    # # Run RRT.
    path = astar(start_pose, goal_pose, occupancy_grid)

    # Plot environment.
    occupancy_grid.draw(axis, path=True, grid=True)
    axis.scatter(start_pose[0], start_pose[1], s=10, marker='o', color='green', zorder=1000)
    axis.scatter(goal_pose[0], goal_pose[1], s=10, marker='o', color='red', zorder=1000)

    axis.axis('equal')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_xlim([-1.25, 1.25])
    axis.set_ylim([-1.25, 1.25])
  mng = plt.get_current_fig_manager()
  mng.window.state("zoomed")
  plt.show()

