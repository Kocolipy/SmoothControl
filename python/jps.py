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
from scipy.interpolate import interp1d

import Constants
import OccupancyGrid
import Spline

# Constants used for indexing.
X = Constants.X
Y = Constants.Y
YAW = Constants.YAW

ROBOT_RADIUS = Constants.ROBOT_RADIUS

# Defines a node of the graph.
class Node(object):
  occupancy_grid = None
  goal = None

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
    return self._pathcost + np.sqrt(sum((Node.goal.position - self.position)**2))

  @property
  def pathcost(self):
    return self._pathcost

  @pathcost.setter
  def pathcost(self, c):
    self._pathcost = c

  @property
  def indices(self):
    return self.occupancy_grid.get_index(self.position)

  def __eq__(self, other):
    return self.indices == other.indices

def normalize(v):
  norm = np.linalg.norm(v)
  if norm == 0:
    return v
  return v / norm

def astar(start_pose, goal_position, occupancy_grid):
  # Open list contain nodes to be checked
  open_list = SortedKeyList(key=lambda x: x.cost)

  # Define goal position and occupancy grid for all Nodes
  Node.occupancy_grid = occupancy_grid
  Node.goal = Node(goal_position)


  if not occupancy_grid.is_free(start_pose[:2]):
    print('Start position is not in the free space.')
    return None

  if not occupancy_grid.is_free(goal_position):
    print('Goal position is not in the free space.')
    return None

  start_node = Node(start_pose)
  explored = 0
  open_list.add(start_node)
  while open_list:
    current_node = open_list.pop(0)

    if occupancy_grid.is_explored(current_node.indices):
      continue

    occupancy_grid.set_explored(current_node.indices)
    explored += 1
    # if explored % 10 == 0:
    #   occupancy_grid.sketch()

    # Found the goal
    if current_node == Node.goal:
      path = []
      current = current_node
      while current is not None:
        path.append(current.position)
        occupancy_grid.set_path(current.indices)
        current = current.parent
      return path[::-1]

    # Generate new nodes to check
    children = []
    for cardinal_step in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
      node = explore_cardinal(occupancy_grid, current_node, cardinal_step[0], cardinal_step[1])
      if node:
        children.append(node)

    for diag_step in [(1,1), (-1, 1), (1, -1), (-1, -1)]:
      nodes = explore_diagonal(occupancy_grid, current_node, diag_step[0], diag_step[1])

      if nodes:
        children += nodes

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

def explore_diagonal(occupancy_grid, current_node, directionX, directionY):
  cur_x, cur_y = current_node.indices
  curCost = current_node.pathcost

  toExplore = []
  while (True):
    cur_x += directionX
    cur_y += directionY
    curCost += np.sqrt(2) * occupancy_grid.resolution

    if not occupancy_grid.is_free_index((cur_x, cur_y)):
      return toExplore

    new_node = Node(occupancy_grid.get_position(cur_x, cur_y))
    new_node.parent = current_node

    if new_node == Node.goal:
      toExplore.append(new_node)
      return toExplore

    # If a jump point is found,
    if (not occupancy_grid.is_free_index((cur_x + directionX, cur_y))) and occupancy_grid.is_free_index((cur_x + directionX, cur_y + directionY)):
      toExplore.append(new_node)
      return toExplore
    else:  # extend a horizontal search to look for potential jump points
      horizontal_node = explore_cardinal(occupancy_grid, new_node, directionX, 0)
      if horizontal_node:
        toExplore.append(horizontal_node)

    if (not occupancy_grid.is_free_index((cur_x, cur_y + directionY))) and occupancy_grid.is_free_index((cur_x + directionX, cur_y + directionY)):
      toExplore.append(new_node)
      return toExplore
    else:  # extend a vertical search to look for potential jump points
      vertical_node = explore_cardinal(occupancy_grid, new_node, 0, directionY)
      if vertical_node:
        toExplore.append(vertical_node)

def explore_cardinal(occupancy_grid, current_node, directionX, directionY):
  cur_x, cur_y = current_node.indices
  curCost = current_node.pathcost

  while (True):
    cur_x += directionX
    cur_y += directionY
    curCost += occupancy_grid.resolution

    if not occupancy_grid.is_free_index((cur_x, cur_y)):
      return None

    new_node = Node(occupancy_grid.get_position(cur_x, cur_y))
    new_node.parent = current_node

    if new_node == Node.goal:
      return new_node

    # check neighbouring cells
    if directionX == 0:
      if (not occupancy_grid.is_free_index((cur_x + 1, cur_y))) and occupancy_grid.is_free_index((cur_x + 1, cur_y + directionY)):
        return new_node
      if (not occupancy_grid.is_free_index((cur_x - 1, cur_y))) and occupancy_grid.is_free_index((cur_x - 1, cur_y + directionY)):
        return new_node

    else: # directionY == 0
      if (not occupancy_grid.is_free_index((cur_x, cur_y + 1))) and occupancy_grid.is_free_index((cur_x + directionX, cur_y + 1)):
        return new_node
      if (not occupancy_grid.is_free_index((cur_x, cur_y - 1))) and occupancy_grid.is_free_index((cur_x + directionX, cur_y - 1)):
        return new_node


def los_trimming(occupancy_grid, path):
  def los(occupancy_grid, x, y):
    direction = (y - x)[:2]
    for mag in np.linspace(0, 1, 101):
      if occupancy_grid.is_occupied(x + mag * direction):
        return False
    return True

  new_path = []

  # Line-of-sight filtering
  while path:
    current = path.pop(0)
    for i in range(min(6, len(path)) - 1, 0, -1):
      if los(occupancy_grid, current, path[i]):
        path = path[i:]
        break
    new_path.append(current)
  return new_path

def smoothing(occupancy_grid, path, start_pose):
  # Shift points to away from obstacles
  offsets = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
  for p in path:
    loc_index = np.array(occupancy_grid.get_index(p))
    step = [0, 0]
    for offset in offsets:
      if not occupancy_grid.is_free_index(tuple(loc_index + offset)):
        step -= normalize(offset)*occupancy_grid._resolution*0.2
    p += step

  # Add additional start and end points
  m = np.array([np.cos(start_pose[YAW]), np.sin(start_pose[YAW])], dtype=np.float32)

  # Scan for the best place to put the node to account for orientation
  nPoints = 10
  curvature = Constants.CURVATURE*occupancy_grid._resolution

  def circle(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

  dir = normalize(path[1] - path[0])
  sigma = np.pi/4

  # Position = 1 means the next point is left of the yaw, -1 is right
  position = -1 if m[0] * dir[1] - m[1] * dir[0] < 0 else 1
  if np.dot(dir, m) < -0.75:
    # Robot is opposite of the next point, avoid searching in direction of robot
    if position == 1:
      # print("opp, left")
      s = np.linspace(start_pose[YAW] + sigma, start_pose[YAW] + 2*sigma, nPoints)
    else:
      # print("opp, right")
      s = np.linspace(start_pose[YAW] - sigma, start_pose[YAW] - 2*sigma, nPoints)
  else:
    if position == 1:
      # print("left")
      s = np.linspace(start_pose[YAW], start_pose[YAW]+sigma, nPoints)
    else:
      # print("right")
      s = np.linspace(start_pose[YAW]-sigma, start_pose[YAW], nPoints)
  t = np.linspace(start_pose[YAW]-np.pi-sigma, start_pose[YAW]-np.pi+sigma, nPoints)
  sp = np.array([[[path[0] + circle(curvature, i),
                   path[0] + circle(0.5*occupancy_grid._resolution, j)] for j in t] for i in s])

  t1 = (0.5*occupancy_grid._resolution)**0.5
  t2 = curvature**0.5 + t1
  t21 = t2 - t1

  sp = sp.reshape(-1, 2, 2)
  Cp = [t21 / t2 / t1 * (path[0] - p0) + t1 / t2 / t21 * (p2 - path[0]) for (p2, p0) in sp]
  mag = np.dot(np.array([normalize(c) for c in Cp]), m)
  index = mag.argmax()
  path.insert(0, sp[index][1])
  path.insert(2, sp[index][0])

  # # End point
  vec = normalize(path[-2] - path[-1])*occupancy_grid._resolution*0.25
  path.append(path[-1]-vec)
  return path

def CatmullRomChain(occupancy_grid, P):
  """
  Calculate Catmull–Rom for a chain of points and return the combined curve.
  """
  sz = len(P)
  # C contain splines
  C = []
  i = 0
  while i < sz - 3:
    spline = Spline.Spline(P[i], P[i + 1], P[i + 2], P[i + 3])

    # Collect the point with the shortest distance to an obstacle
    points = spline.discretise()
    dist = None
    for j in range(len(points)):
      for d in np.array([[0,1], [1,0], [0, -1], [-1, 0]]):
        direc = d*occupancy_grid._resolution
        if occupancy_grid.is_occupied(points[j] + direc):
          center = occupancy_grid.get_position(*occupancy_grid.get_index(points[j] + direc))
          dis = np.linalg.norm(points[j] - center)

          if dist is None or dis < dist[0]:
            dist = [dis, j]

    # If point is too close to an obstacle, add point as support node and rebuild splines
    if dist and dist[0] < occupancy_grid._resolution:
        if i < 2:
          print("There is no path which allow for the starting orientation and curvature constraints.")
          return []
        else:
          orig = normalize(P[i + 2] - P[i + 1])
          new = points[dist[1]] - P[i + 1]
          point = P[i+1] + orig*np.dot(new, orig)
          P.insert(i+2, point)
          sz += 1
          continue

    C.append(spline)
    i += 1

  return C


def jps(start_pose, goal_pose, occupancy_grid):
  # JPS algorithm
  path = astar(start_pose, goal_pose, occupancy_grid)
  path.append(goal_pose)

  # LOS trimming
  filtered_path = los_trimming(occupancy_grid, path)

  # Perturb nodes if they are close to obstacles
  # Adding nodes to front and end of path to generate complete splines
  edited_path = smoothing(occupancy_grid, filtered_path, start_pose)

  # fit splines
  chain = CatmullRomChain(occupancy_grid, edited_path)
  if not chain:
    return None

  return chain

if __name__ == '__main__':
  # maps = ["standard_map", "narrow_map", "arc_obs", "bubbles", "cage", "maze", "circle"]
  maps = ["maze"]

  fig, ax = plt.subplots(1, len(maps), squeeze=False)
  for i, grid in enumerate(maps):
    axis = ax[0, i]

    # # Timing
    # from timeit import timeit
    #
    # def timing():
    #   astar(start_pose, goal_pose, occupancy_grid)
    #
    # times = []
    # for i in range(100):
    #   occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(grid)
    #   times.append(timeit(timing, number=1))
    # times = np.array(times)
    # print(np.mean(times))
    # print(np.std(times))

    occupancy_grid, start_pose, goal_pose = OccupancyGrid.loadMap(grid)

    path = astar(start_pose, goal_pose, occupancy_grid)
    path.append(goal_pose)

    filtered_path = los_trimming(occupancy_grid, path)
    linx, liny = zip(*filtered_path)

    edited_path = smoothing(occupancy_grid, filtered_path, start_pose)
    px, py = zip(*edited_path)

    chain = CatmullRomChain(occupancy_grid, edited_path)
    ax, ay = zip(*edited_path)

    # Combine discretised splines to form a path
    if chain:
      full_path = []
      for spline in chain:
        full_path += spline.discretise()
      splinex, spliney = zip(*full_path)
    else:
      exit()

    # # Measure distane of path
    # dist = 0
    # for index in range(len(c) - 1):
    #   dist += np.linalg.norm(c[index + 1] - c[index])
    # print(dist)

    # Plot environment.
    # axis.plot([], [], color="black", label="RRT")
    occupancy_grid.draw(axis, path=True, grid=True)
    axis.scatter(start_pose[0], start_pose[1], s=10, marker='o', color='green', zorder=1000)
    axis.scatter(goal_pose[0], goal_pose[1], s=10, marker='o', color='red', zorder=1000)
    axis.scatter(ax, ay, s=10, marker='o', color="red", zorder=800)
    axis.scatter(px, py, s=10, marker='o', color="black", zorder=1000)
    axis.plot(linx, liny, color="red", label="Linear")
    axis.plot(splinex, spliney, color="blue", label="Cubic Spline")

    axis.axis('equal')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_xlim([-1.5, 1.5])
    axis.set_ylim([-1.5, 1.5])
    axis.legend(loc="right")
  mng = plt.get_current_fig_manager()
  mng.window.state("zoomed")
  plt.show()

