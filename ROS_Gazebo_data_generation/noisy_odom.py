#! /usr/bin/env python

import rospy
import numpy as np
from math import *
import tf
from nav_msgs.msg import Odometry

a1 = 0.0001
a2 = 0.0001
a3 = 0.0001
a4 = 0.0001
prev_odom = None
base_frame = "base_link"
pose = [0.0,0.0,0.0]

def noisy(data):
	global a1
	global a2
	global a3
	global a4
	global prev_odom
	global pose
	global base_frame

	if(prev_odom == None):
		prev_odom = data
		pose[0] = data.pose.pose.position.x
		pose[1] = data.pose.pose.position.y
		q = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
		(r, p, y) = tf.transformations.euler_from_quaternion(q)
		pose[2] = y
	else:
		dx = data.pose.pose.position.x - prev_odom.pose.pose.position.x
		dy = data.pose.pose.position.y - prev_odom.pose.pose.position.y
		transl = sqrt(dx*dx + dy*dy)
		q = [ prev_odom.pose.pose.orientation.x, prev_odom.pose.pose.orientation.y, prev_odom.pose.pose.orientation.z, prev_odom.pose.pose.orientation.w ]
		(r,p, theta1) = tf.transformations.euler_from_quaternion(q)
		q = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
		(r,p, theta2) = tf.transformations.euler_from_quaternion(q)
		rot1 = atan2(dy, dx) - theta1
		rot2 = theta2-theta1-rot1
		
		std_dev_rot1 = (a1*abs(rot1))+(a2*transl)
		std_dev_rot2 = (a1*abs(rot2))+(a2*transl)
		std_dev_transl = (a3*transl)+(a4*(abs(rot1)+abs(rot2)))

		rot1 = rot1 + np.random.normal(0,std_dev_rot1*std_dev_rot1)
		rot2 = rot2 + np.random.normal(0,std_dev_rot2*std_dev_rot2)
		transl = transl + np.random.normal(0,std_dev_transl*std_dev_transl)
	
		pose[0] = pose[0]+(transl*cos(theta1+rot1))
		pose[1] = pose[1]+(transl*sin(theta1+rot1))
		pose[2] = pose[2]+rot1+rot2
		prev_odom = data

	# publish to tf

	#br = tf.TransformBroadcaster()
	#br.sendTransform((pose[0],pose[1],0), tf.transformations.quaternion_from_euler(0, 0, pose[2]), data.header.stamp, base_frame, "odom")

	# publish to noisy_odom
 
	#odom = Odometry()
	'''noisy_odom = noise_odom()
	noisy_odom.header.stamp = odom.header.stamp
	noisy_odom.header.frame_id = odom.header.frame_id
	noisy_odom.child_frame_id = "/base_link"
	noisy_odom.x = pose[0]
	noisy_odom.y = pose[1]
	noisy_odom.phi = pose[2]'''
	noisy_odom = Odometry()
	noisy_odom.header.stamp = data.header.stamp
	noisy_odom.header.frame_id = "odom"
	noisy_odom.child_frame_id = "base_link"
	noisy_odom.pose.pose.position.x = pose[0]
	noisy_odom.pose.pose.position.y = pose[1]
	noisy_odom.pose.pose.position.z = data.pose.pose.position.z
	q = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
	(r, p, y) = tf.transformations.euler_from_quaternion(q)
	noisy_odom_quat = tf.transformations.quaternion_from_euler(r,p,pose[2])
	noisy_odom.pose.pose.orientation.x = noisy_odom_quat[0]
	noisy_odom.pose.pose.orientation.y = noisy_odom_quat[1]
	noisy_odom.pose.pose.orientation.z = noisy_odom_quat[2]
	noisy_odom.pose.pose.orientation.w = noisy_odom_quat[3]

	noisy_odom.pose.covariance = data.pose.covariance
	noisy_odom.twist.covariance = data.twist.covariance
	
	pub.publish(noisy_odom)

if __name__ == '__main__':
	
	rospy.init_node('noisy_odom',anonymous=True)
	rospy.Subscriber("/odom",Odometry,noisy)
	pub = rospy.Publisher("/noisy_odom",Odometry,queue_size=10)
	rospy.spin()
