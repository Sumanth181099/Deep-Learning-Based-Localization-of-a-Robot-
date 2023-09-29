#! /usr/bin/env python

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rosbag
from sensor_msgs.msg import CompressedImage, LaserScan
from learning_based_robot.msg import gt_pose
from nav_msgs.msg import Odometry
import tf

bag0 = rosbag.Bag('Image_seq7.bag','w')
bag1 = rosbag.Bag('gt_pose_seq7.bag','w')
bag2 = rosbag.Bag('laser_seq7.bag', 'w')

def ground_truth_pose(data):
	if(data!=None):
		ground_truth = gt_pose()
		ground_truth.header.stamp = data.header.stamp
		ground_truth.child_frame_id = data.child_frame_id
		ground_truth.x = data.pose.pose.position.x
		ground_truth.y = data.pose.pose.position.y
		q = [ data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w ]
		(r, p, y) = tf.transformations.euler_from_quaternion(q)
		ground_truth.phi = y

		pose_pub.publish(ground_truth)

def sync_cb(image_data, laser_data, ekf_odom):

	try:
		bag0.write('/lbrobot/camera1/image_raw/compressed', image_data)
		#bag1.write('/odometry/filtered',Odometry)
		bag1.write('/odom/pose', ekf_odom)
		bag2.write('/scan', laser_data)
		print("in try")
	
	finally:
		'''bag0.close()
		bag1.close()
		bag2.close()'''
		print("Recording!")

def starter():
	global pose_pub
	rospy.init_node('image_dataset', anonymous=True)
	#rate = rospy.Rate(10)
        rospy.sleep(2)             # 2 seconds sleep
	rospy.Subscriber("/odometry/filtered", Odometry, ground_truth_pose)
	image_sub = Subscriber("/lbrobot/camera1/image_raw/compressed", CompressedImage)
	laser_sub = Subscriber("/scan", LaserScan)
	odom_sub  = Subscriber("/odom/pose", gt_pose)
	#odom_sub  = Subscriber("/odometry/filtered", Odometry)
	pose_pub = rospy.Publisher("/odom/pose", gt_pose, queue_size=10)
	#approx_time_sync = ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=10, slop=0.1))
	#approx_time_sync.registerCallback(sync_cb)

	'''rospy.init_node('image_dataset', anonymous=True)
	rospy.Subscriber("/odometry/filtered", Odometry, ground_truth_pose)
	pose_pub = rospy.Publisher("/odom/pose", gt_pose, queue_size=10)'''
	
	approx_time_sync = ApproximateTimeSynchronizer([image_sub, laser_sub, odom_sub], queue_size=10, slop=0.1, allow_headerless=True)
	approx_time_sync.registerCallback(sync_cb)
	rospy.spin()

	bag0.close()
	bag1.close()
	bag2.close()

if __name__=="__main__":
	try:
		starter()
	except rospy.ROSInterruptException:
		pass
