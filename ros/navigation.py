#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rospy
import sys
import pickle

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
from tf import TransformListener
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from tf.transformations import euler_from_quaternion

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../python')
sys.path.insert(0, directory)
try:
    import path
except ImportError as e:
    raise ImportError('Unable to import path.py. Make sure this file is in "{}"'.format(directory))

SPEED = .2
EPSILON = .1

X = 0
Y = 1
YAW = 2


def get_velocity(position, path_points):
    v = np.zeros_like(position)
    if len(path_points) == 0:
        return v

    while len(path_points) > 1 and np.linalg.norm(position - path_points[0]) < .2:
        path_points.pop(0)

    dir = path_points[0] - position
    if np.linalg.norm(dir) > 0.5:
        dir = dir / np.linalg.norm(dir) * 0.5
    return dir

def closestPoint(position, splines):
    error = splines[0].VecToSpline(position[:2])
    if len(splines) > 1:
        error_next = splines[1].VecToSpline(position[:2])
        if np.linalg.norm(error_next) < np.linalg.norm(error):
            # Robot is closer to next spline
            splines.pop(0)
            error = error_next

    t = splines[0].pointOnSpline(position[:2])
    if t > 1:
        if len(splines) > 1:
            splines.pop(0)
            t = t - 1 
        else:
            t = 1
    g = splines[0].gradient(t)
    return splines[0].poly(t), g/np.linalg.norm(g) 

def turn(vecA, vecB):
    # 1 means vecB is left of the vecA, -1 is right, 0 is same/opposite direction
    det = vecA[0] * vecB[1] - vecA[1] * vecB[0]
    if det < 0:
        return -1
    elif det > 0:
        return 1
    else:
        return 0

def angle_between(vecA, vecB):
    vecA = vecA / np.linalg.norm(vecA)
    vecB = vecB / np.linalg.norm(vecB)
    return np.arccos(np.clip(np.dot(vecA, vecB), -1.0, 1.0))

def calculateError(position, splines):
    target, grad = closestPoint(position, splines)
    offset = target - position[:2]

    yaw = np.array([np.cos(position[2]), np.sin(position[2])], dtype=np.float32)

    theta = turn(grad, yaw)*angle_between(yaw, grad)
    r = turn(grad, -1*offset)*np.linalg.norm(offset)
    return r, theta

def leadController(position, splines, record):
    r, _ = calculateError(position, splines)
    with open('/tmp/led_e.txt', 'a') as fp:
        fp.write(str(r) + '\n')

    # Record[0] stores the error, record[1] store the output
    r_m1 = record[0]
    w_m1 = record[1]
    w = 168*r - 168*r_m1 + 0.8521*w_m1
    record[0] = r
    record[1] = w

    return 0.2, -1*w, record

def leadlagController(position, splines, record):
    r, _ = calculateError(position, splines)
    with open('/tmp/leadlag_e.txt', 'a') as fp:
        fp.write(str(r) + '\n')

    # Record[0] and [1] stores the error and prev error
    # Record[2] and [3] store the output and prev output
    r_m1 = record[0]
    r_m2 = record[1]
    w_m1 = record[2]
    w_m2 = record[3]
    w = 168*r - 331.2*r_m1 + 163.2*r_m2 + 1.852*w_m1 - 0.852*w_m2
    #w = 168*r - 168*r_m1 + 0.8521*w_m1
    record[1] = record[0]
    record[0] = r
    record[3] = record[2]
    record[2] = w

    return 0.2, -1*w, record

def PIDController(position, splines, record):
    Kp = 19.1818
    Ki = 1.295888696
    Kd = 49.961277
    
    r, theta = calculateError(position, splines)
    with open('/tmp/pid_e.txt', 'a') as fp:
        fp.write(str(r) + '\n')

    # Record[0] stores the integral, record[1] store the previous error
    record[0] += r*0.02
    r_deriv = 0.2*np.sin(theta)
    w = Kp*r + Ki*record[0] + r_deriv*Kd
    record[1] = r

    return 0.2, -1*w, record


def BBController(position, splines):
    target, _ = closestPoint(position, splines)
    offset = target - position[:2]

    yaw = np.array([np.cos(position[2]), np.sin(position[2])], dtype=np.float32)
    return 0.1, turn(yaw, offset)*0.5

class SLAM(object):
    def __init__(self):
        rospy.Subscriber('/map', OccupancyGrid, self.callback)
        self._tf = TransformListener()
        self._occupancy_grid = None
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    def callback(self, msg):
        values = np.array(msg.data, dtype=np.int8).reshape((msg.info.width, msg.info.height))
        processed = np.empty_like(values)
        processed[:] = path.FREE
        processed[values < 0] = path.UNKNOWN
        processed[values > 50] = path.OCCUPIED
        processed = processed.T
        origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
        resolution = msg.info.resolution
        self._occupancy_grid = path.OccupancyGrid(processed, origin, resolution)

    def update(self):
        # Get pose w.r.t. map.
        a = 'occupancy_grid'
        b = 'base_link'
        if self._tf.frameExists(a) and self._tf.frameExists(b):
            try:
                t = rospy.Time(0)
                position, orientation = self._tf.lookupTransform('/' + a, '/' + b, t)
                self._pose[X] = position[X]
                self._pose[Y] = position[Y]
                _, _, self._pose[YAW] = euler_from_quaternion(orientation)
            except Exception as e:
                print(e)
        else:
            print('Unable to find:', self._tf.frameExists(a), self._tf.frameExists(b))
        pass

    @property
    def ready(self):
        return self._occupancy_grid is not None and not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose

    @property
    def occupancy_grid(self):
        return self._occupancy_grid


class GoalPose(object):
    def __init__(self):
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback)
        self._position = np.array([np.nan, np.nan], dtype=np.float32)

    def callback(self, msg):
        # The pose from RViz is with respect to the "map".
        self._position[X] = msg.pose.position.x
        self._position[Y] = msg.pose.position.y
        print('Received new goal position:', self._position)

    @property
    def ready(self):
        return not np.isnan(self._position[0])

    def reached(self):
        self._position = np.array([np.nan, np.nan], dtype=np.float32)

    @property
    def position(self):
        return self._position


def run(args):
    rospy.init_node('rrt_navigation')

    # Update control every 100 ms.
    rate_limiter = rospy.Rate(100)
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
    path_publisher = rospy.Publisher('/path', Path, queue_size=1)
    slam = SLAM()
    goal = GoalPose()
    frame_id = 0
    travelling = False
    previous_time = rospy.Time.now().to_sec()
    pose_history = []
    with open('/tmp/robot_path.txt', 'w'):
        pass
    with open('/tmp/planned_path.txt', 'w'):
        pass
    with open('/tmp/pid_e.txt', 'w'):
        pass
    with open('/tmp/leadlag_e.txt', 'w'):
        pass
    with open('/tmp/lead_e.txt', 'w'):
        pass

    # Stop moving message.
    stop_msg = Twist()
    stop_msg.linear.x = 0.
    stop_msg.angular.z = 0.

    # Make sure the robot is stopped.
    i = 0
    while i < 10 and not rospy.is_shutdown():
        publisher.publish(stop_msg)
        rate_limiter.sleep()
        i += 1

    while not rospy.is_shutdown():
        slam.update()
        current_time = rospy.Time.now().to_sec()

        # Make sure all measurements are ready.
        # Get map and current position through SLAM:
        # > roslaunch exercises slam.launch
        if not goal.ready or not slam.ready:
            rate_limiter.sleep()
            continue
        if goal.ready and not travelling:
            splines = path.jps(slam.pose, goal.position, slam.occupancy_grid)

            record = [0, 0, 0, 0]
            travelling = True
             # Save plotted path
            pickle.dump(splines, open('/tmp/planned_path.txt', 'a')) 
            if not splines:
                print('Unable to reach goal position:', goal.position)
                exit()

        goal_reached = np.linalg.norm(slam.pose[:2] - goal.position) < .2
        if goal_reached:
            print("Goal reached")
            publisher.publish(stop_msg)
            rate_limiter.sleep()
            goal.reached()
            travelling = False
            continue

        #u, w, record = PIDController(slam.pose, splines, record)
        u, w, record = leadlagController(slam.pose, splines, record)
        #u, w, record = leadController(slam.pose, splines, record)
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        publisher.publish(vel_msg)

        # Save robot position
        pose_history.append(slam.pose)
        if len(pose_history) >= 10:
            with open('/tmp/robot_path.txt', 'a') as fp:
                fp.write('\n'.join(','.join(str(v) for v in p) for p in pose_history) + '\n')
                pose_history = []

        # Update plan every 1s.
        #time_since = current_time - previous_time
        #if travelling and time_since < 2.:
        #    rate_limiter.sleep()
         #   continue
        #previous_time = current_time

        # Publish path to RViz.
        path_msg = Path()
        path_msg.header.seq = frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'map'
	full_path = []
        for spline in splines:
	  full_path += spline.discretise()
        for u in full_path:
            pose_msg = PoseStamped()
            pose_msg.header.seq = frame_id
            pose_msg.header.stamp = path_msg.header.stamp
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = u[X]
            pose_msg.pose.position.y = u[Y]
            path_msg.poses.append(pose_msg)
        path_publisher.publish(path_msg)

        rate_limiter.sleep()
        frame_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs RRT navigation')
    args, unknown = parser.parse_known_args()
    try:
        run(args)
    except rospy.ROSInterruptException:
        pass
