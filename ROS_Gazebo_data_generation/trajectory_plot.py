#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry, Path 
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

path = Path()
markerArray = MarkerArray()
count = 0
MARKERS_MAX = 1000
x_list = []
y_list = []
t_x = 0
t_y = 0
#fig = plt.figure()

def callback(data):
    global path
    global x_list
    global y_list
    global r_x
    global r_y
    try:
	    path.header = data.header
	    pose = PoseStamped()
	    pose.header = data.header
	    #pose.header.frame_id = "base_link"
	    pose.pose = data.pose.pose
	    path.poses.append(pose)
            x = data.pose.pose.position.x
	    y = data.pose.pose.position.y
            z = data.pose.pose.position.z
	    #print("original x", x)
	    path_pub.publish(path)
	    marker = Marker()
	    marker.header = path.header
	    #marker.header.frame_id = "odom"
	    marker.type = marker.POINTS
	    marker.action = marker.ADD
	    marker.scale.x = 0.1
	    marker.scale.y = 0.01
	    marker.scale.z = 0.01
	    marker.color.a = 1.0
	    marker.color.r = 1.0
	    marker.color.g = 1.0
	    marker.color.b = 0.0
	    marker.pose.orientation.w = 1.0
	    t_x = round(x,2)
	    t_y = round(y,2)
	    t_z = round(z,2)
	    #print("t_x", t_x)
	    x_list.append(t_x)
	    y_list.append(t_y)
	    #print("x_list", x_list)
    	    #plot_traj(x_list, y_list)
	    plt.plot(x_list,y_list)
	    
    except KeyboardInterrupt:
	    print("yes")
	    plt.savefig('traj.png')
    	    
	 
'''def plot_traj(x, y):
	    plt.plot(x,y)
	    print("x is", x)
	    #print("y is", y)
	    #plt.xlabel('t_x')
	    #plt.ylabel('t_y')
	    plt.title('training sequence 1')
	    plt.legend()
	    plt.show()'''
def starter():
	global path_pub
	rospy.init_node('path_waypoints',anonymous=True, disable_signals=True)
	#odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, callback)
	odom_sub = rospy.Subscriber('/odom', Odometry, callback)
	path_pub = rospy.Publisher('/path_waypoints', Path, queue_size=10)
	#ani = animation.FuncAnimation(fig, callback, interval=1000)
	rospy.sleep(1)
	rospy.spin()
	plt.savefig('traj.png')
	'''plt.plot(x_list,y_list)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('training sequence 1')
	plt.legend()
	plt.show()
	def show_plot():
		plt.show()'''
	#visual_pub = rospy.Publisher('/visualization_maker', Marker, queue_size=10)
	#visualarr_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)

if __name__ == '__main__':
	starter()
	
