#! /usr/bin/env python

import rospy
import math
from nav_msgs.msg import Odometry, Path
import tf
import numpy as np
from geometry_msgs.msg import Twist
#from learning_based_robot.msg import gt_pose
import matplotlib.pyplot as plt

look_ahead = 0.4
x_list = []
y_list = [] 
phi_list = []
speed = Twist() 

class list_manipulation:

	def __init__(self):
		self.x = [1.0]
		self.y = [1.0]
		self.phi = [0.0]
	def list_creation(self, current_position):
		global x_list
		global y_list
		global phi_list 
		x_list = self.x.append(current_position.x)
		y_list = self.y.append(current_position.y) 
		phi_list = self.phi.append(current_position.phi)
		return x_list, y_list, phi_list 

class current_position:

	def __init__(self, list_manipulation):
		rospy.Subscriber("/odometry/filtered",Odometry, self.callback)
		rospy.sleep(2)

	def callback(self, data):
		self.current_pose = data
		self.x = self.current_pose.pose.pose.position.x
		self.y = self.current_pose.pose.pose.position.y
		q = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
		(r, p, phi) = tf.transformations.euler_from_quaternion(q)
		self.phi = phi

	def distance(self, pose_x, pose_y):
		dx = self.x - pose_x
		dy = self.y - pose_y 
		return(math.hypot(dx,dy))

class Goal_tracker:

	def __init__(self, waypoint_x, waypoint_y):
		self.waypoint_x = waypoint_x
		self.waypoint_y = waypoint_y
		self.prev_waypoint_index = None

	def mini_goal_index(self, current_position):
		d = []
		if self.prev_waypoint_index == None:
			dx = [current_position.x - i for i in self.waypoint_x]
			dy = [current_position.y - j for j in self.waypoint_y]
			d.append(math.hypot(dx,dy))
			index = np.argmin(d)
			self.prev_waypoint_index = index 
		
		else:
			index = self.prev_waypoint_index
			#dist = current_position.distance(self.waypoint_x[index], self.waypoint_y[index])

		while look_ahead > current_position.distance(self.waypoint_x[index], self.waypoint_y[index]):
			if((index+1)>=len(self.waypoint_x)):
				break
			else:
				index=index+1

		return index

class Controller:

	def __init__(self):
		pass
	def control(self, waypoints, way_index, mini_goal_index, current_position):
		#index = waypoints.mini_goal_index(current_position)
		if(way_index <len(waypoints)):
			self.xn = mini_goal_index.waypoint_x[way_index]
			self.yn = mini_goal_index.waypoint_y[way_index]

		self.del_phi = math.atan2(self.yn, self.xn) - current_position.phi
	
	def stop_near_goal(self,waypoint_x,waypoint_y):
		goal_x = waypoint_x[len(waypoint_x)-1]
		goal_y = waypoint_y[len(waypoint_y)-1]
		x_diff = goal_x - self.xn
		y_diff = goal_y - self.yn
		overall_diff = math.hypot(x_diff, y_diff)
		if(overall_diff < 0.3):
			#speed = Twist()
			speed.linear.x = 0.3
		elif(overall_diff < 0.1):
			speed.linear.x = 0.0
		else:
			speed.linear.x = 0.8
		
		if(self.del_phi < 0.3):
			speed.angular.z = 0.2
		elif(self.del_phi < 0.1):
			speed.angular.z = 0.0
		else:
			speed.angular.z = 0.7
		
		velocity_pub.publish(speed)

def main():
    # reference waypoints input
    waypoint_x = [1,2,3,4,5]
    waypoint_y = [1,1,1,1,1]
    plt.scatter(waypoint_x, waypoint_y, label="reference_trajectory", s=10, color="blue", alpha=0.5)
    plt.xlabel('x in metres')
    plt.ylabel('y in metres')
    plt.title('Reference trajectory')
    plt.legend()
    plt.show()
    rospy.init_node("low_level_controller")
    global velocity_pub
#     global velocity_pub
#     velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
#     list_manipulation()
#     Goal_tracker(waypoint_x, waypoint_y)
#     way_index = Goal_tracker.mini_goal_index(current_position)
# 
if __name__=="__main__":
    main()