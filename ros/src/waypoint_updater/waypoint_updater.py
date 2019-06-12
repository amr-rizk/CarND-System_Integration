#!/usr/bin/env python

# 2018
# Gaurav Garg, Amr Rizk, Roland Schirmer, Nobuyuki Tomatsu

import math
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 150 # Number of waypoints to publish. Set down to 150, was 200
MAX_DECC = .5  # To define the maximum decceleration of the car


class WaypointUpdater(object):	# define class
	def __init__(self):
		rospy.init_node('waypoint_updater')	# initialize rospy node
		# Subscribers
		rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)	# current position
		rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)	# all waypoints, once
		rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)	# traffic light waypoint
		# rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)    # obstacles, not used
		# Publishers
		self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=10)	# 3 enough?
		# Declare private __fields
		self.__base_waypoints = None	# all waypoints
		self.__base_waypoints_x_y = None	# 2D, only x, y
		self.__current_pose = None	# current position
		self.__waypoints_tree = None	# waypoints sorted by distance by KDTree
		self.__stopline_wp_idx = -1	# Index of stopline waypoint

		self.loop()

	def loop(self):	# cycle node at 50Hz
		rate = rospy.Rate(50)
		while not rospy.is_shutdown():	# while running
			if self.__current_pose and self.__waypoints_tree:	# don't calculate if empty'
				idx = self.get_nearest_waypoint_id(self.__current_pose)
				self.update_waypoints(idx)
			rate.sleep()    

	def pose_cb(self, pose):	# current position
		self.__current_pose = pose.pose

	def waypoints_cb(self, lane):
		if not self.__waypoints_tree:
			self.__base_waypoints = lane.waypoints
			self.__base_waypoints_x_y = [[w.pose.pose.position.x, w.pose.pose.position.y] for w in self.__base_waypoints]
			self.__waypoints_tree = KDTree(self.__base_waypoints_x_y)

	def get_nearest_waypoint_id(self, pose):	# return index of nearest waypoint

		idx = self.__waypoints_tree.query([pose.position.x, pose.position.y])[1] # calculate with KDTree

		closest_point = self.__base_waypoints_x_y[idx]	# nearest point
		previous_point = self.__base_waypoints_x_y[idx - 1]

		closest_vector = np.array(closest_point)	# vector to point
		previous_vector = np.array(previous_point)
		current_pos_vector =  np.array([self.__current_pose.position.x, self.__current_pose.position.y])	# vector to car position

		val = np.dot(closest_vector - previous_vector, current_pos_vector - closest_vector)	 # Skalarprodukt / dot product
		if val > 0:	# point lies behind car
			return (idx + 1) % len(self.__base_waypoints_x_y)

		return idx	# point in front of car

	def update_waypoints(self, idx):
		# Creating header and setting timestamp
		header = Header()
		header.stamp = rospy.Time.now()
		msg = Lane()
		msg.header = header
		next_waypoints = self.__base_waypoints[idx: idx + LOOKAHEAD_WPS]	# next waypoints to publish
		msg.waypoints = next_waypoints
		if self.__stopline_wp_idx != -1 and self.__stopline_wp_idx < (idx + LOOKAHEAD_WPS):	# if stopline closer than LOOKAHEAD_WPS then decelerate
			msg.waypoints = self.__decelerate(next_waypoints, idx)

		self.final_waypoints_pub.publish(msg)	# publish waypoints

	def __decelerate(self, waypoints, idx):
		temp = []
		for i, wp in enumerate(waypoints):  # add numbers
			p = Waypoint()
			p.pose = wp.pose
			stop_idx = max(self.__stopline_wp_idx - idx - 2, 0)	# stop before stop line
			dist = self.distance(waypoints, i, stop_idx)	# calculate distance to decrease velocity proportional to distance
			#vel = min(dist, wp.twist.twist.linear.x)	# velocity <= distance, <= current velocity
			vel=math.sqrt(2*MAX_DECC*dist)
			if vel < 1.:
				vel = 0.
			p.twist.twist.linear.x=min(vel,wp.twist.twist.linear.x) 
			temp.append(p)	# add current value to temp
		return temp

	def traffic_cb(self, msg):
		self.__stopline_wp_idx = msg.data

	#def obstacle_cb(self, msg):	# Callback for /obstacle_waypoint message. not used
		# pass

	def get_waypoint_velocity(self, waypoint):
		return waypoint.twist.twist.linear.x

	def set_waypoint_velocity(self, waypoints, waypoint, velocity):
		waypoints[waypoint].twist.twist.linear.x = velocity

	def distance(self, waypoints, wp1, wp2):
		dist = 0
		dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
		for i in range(wp1, wp2+1):
			dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
			wp1 = i

		return dist



if __name__ == '__main__':
	try:
		WaypointUpdater()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start waypoint updater node.')
