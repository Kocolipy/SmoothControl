import re
import numpy as np
import scipy
import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import os
import yaml
import cv2

import Constants

# Constants used for indexing.
X = 0
Y = 1
YAW = 2

# Constants for occupancy grid.
FREE = 0
UNKNOWN = 1
OCCUPIED = 2
PATH = 3
EXPLORED = 4

ROBOT_RADIUS = Constants.ROBOT_RADIUS

# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy().astype(np.int_)
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def set_path(self, index):
    self._values[index] = (self._values[index]//4)*4 + PATH

  def set_explored(self, index):
    self._values[index] = EXPLORED

  def set_visited(self, index, yaw_int):
    self._values[index] += int(np.power(2, yaw_int+2))

  def sketch(self):
    fig = plt.figure()
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution],
              cmap="gray_r")
    plt.xlim([-2.5, 2.5])
    plt.ylim([-2.5, 2.5])
    axis = fig.add_subplot(1, 1, 1)

    minorLocator = MultipleLocator(self.resolution)
    # Set minor tick locations.
    axis.yaxis.set_minor_locator(minorLocator)
    axis.xaxis.set_minor_locator(minorLocator)
    axis.grid(which="minor")
    axis.grid(which="major")

    for i, row in enumerate(self._values):
      for j, v in enumerate(row):
        if v == EXPLORED:
          position = self.get_position(i, j) - (self.resolution / 2, self.resolution / 2)
          axis.add_patch(patches.Rectangle(position, self.resolution, self.resolution, color="blue"))

        indicator = v % 4
        if indicator == PATH:
          position = self.get_position(i, j) - (self.resolution/2, self.resolution/2)
          axis.add_patch(patches.Rectangle(position, self.resolution, self.resolution, color="yellow"))

    plt.show()


  def draw(self, axis, path=False, grid=False):
    axis.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution],
              cmap="gray_r")

    if grid:
      minorLocator = MultipleLocator(self.resolution)
      # Set minor tick locations.
      axis.yaxis.set_minor_locator(minorLocator)
      axis.xaxis.set_minor_locator(minorLocator)
      axis.grid(which="minor")
      axis.grid(which="major")

    if path:
      for i, row in enumerate(self._values):
        for j, v in enumerate(row):
          if v == EXPLORED:
            position = self.get_position(i, j) - (self.resolution/2, self.resolution/2)
            axis.add_patch(patches.Rectangle(position, self.resolution, self.resolution, color="blue"))

          indicator = v % 4
          if indicator == PATH:
            position = self.get_position(i, j) - (self.resolution/2, self.resolution/2)
            # axis.add_patch(patches.Rectangle(position, self.resolution, self.resolution, color="yellow"))

          if indicator == OCCUPIED:
            position = self.get_position(i, j) - (self.resolution/2, self.resolution/2)
            axis.add_patch(patches.Rectangle(position, self.resolution, self.resolution, color="grey"))

  def get_index(self, position):
    idx = np.rint((position - self._origin) / self._resolution).astype(int)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)]%4 == OCCUPIED or self._values[self.get_index(position)]%4 == UNKNOWN

  def is_occupied_index(self, index):
    return self._values[index]%4 == OCCUPIED or self._values[index]%4 == UNKNOWN

  def is_free(self, position):
    return self._values[self.get_index(position)]%4 == FREE

  def is_free_index(self, index):
    return self._values[index]%4 == FREE

  def is_explored(self, index):
    return not self.is_free_index(index) or self._values[index]//4 == 1

  def is_visited(self, index, yaw_int):
    return not self.is_free_index(index) or self._values[index] & int(np.power(2, yaw_int+2)) != 0


def read_pgm(filename, byteorder='>'):
  """Read PGM file."""
  with open(filename, 'rb') as fp:
    buf = fp.read()
  try:
    header, width, height, maxval = re.search(
        b'(^P5\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n])*'
        b'(\d+)\s(?:\s*#.*[\r\n]\s)*)', buf).groups()
  except AttributeError:
    raise ValueError('Invalid PGM file: "{}"'.format(filename))
  maxval = int(maxval)
  height = int(height)
  width = int(width)
  img = np.frombuffer(buf,
                      dtype='u1' if maxval < 256 else byteorder + 'u2',
                      count=width * height,
                      offset=len(header)).reshape((height, width))
  res = cv2.resize(img, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
  return res.astype(np.float32) / 255.


def loadMap(mapname):
  with open(mapname + '.yaml') as fp:
    data = yaml.load(fp, Loader=yaml.FullLoader)

  start = np.array(data["start"], dtype=np.float32)
  goal = np.array(data["goal"], dtype=np.float32)

  img = read_pgm(os.path.join(os.path.dirname(mapname), data['image']))
  occupancy_grid = np.empty_like(img, dtype=np.int8)
  occupancy_grid[:] = UNKNOWN
  occupancy_grid[img < .1] = OCCUPIED
  occupancy_grid[img > .9] = FREE
  # Transpose (undo ROS processing).
  occupancy_grid = occupancy_grid.T
  # Invert Y-axis.
  occupancy_grid = occupancy_grid[:, ::-1]
  occupancy_grid = OccupancyGrid(occupancy_grid, data['origin'], data['resolution'])

  return occupancy_grid, start, goal