#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from copy import deepcopy
import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
	self.base_waypoints = None
	self.final_waypoints = None
	self.current_pose = None 

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)

	# TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        #rospy.Subscriber('/traffic_waypoint', Lane, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

	# Set rate
	rate = rospy.Rate(5)  # 5

        # Check that sim is active and get waypoints if so
	while not rospy.is_shutdown():
	  self.get_final_waypoints()
	  rate.sleep()

        #rospy.spin()


    def pose_cb(self, msg):
        self.current_pose = msg.pose

    def waypoints_cb(self, waypoints):
	self.base_waypoints = waypoints.waypoints
    
    def get_final_waypoints(self):
	# Check for initialization
        # Get current position and yaw angle

	if self.base_waypoints and self.current_pose:
          #rospy.logwarn('base_waypoints and current_pose defined')

	  # Initialize waypoint index and flag
	  i_wp = 0	
	  flag_wp = False

	  # Go through waypoints until waypoint is past current state
	  while flag_wp == False:

            # Get base waypoint position 
      	    x_wp = self.base_waypoints[i_wp].pose.pose.position.x
	    y_wp = self.base_waypoints[i_wp].pose.pose.position.y 

            # Get current position and yaw angle
      	    x = self.current_pose.position.x
	    y = self.current_pose.position.y
	    quaternion = (self.current_pose.orientation.x,
                       self.current_pose.orientation.y,
                       self.current_pose.orientation.z,
                       self.current_pose.orientation.w)
	    euler = tf.transformations.euler_from_quaternion(quaternion) 
            yaw = euler[2]

            # Find x position difference between waypoint and current state
            x_diff = (x_wp - x) * math.cos(yaw) + (y_wp - y) * math.sin(yaw)

	    # If x_diff is less than zero, then waypoint is behind
	    # TO DO: Improve method by solving for min distance! Ron
	    if x_diff < 0.0:
	      i_wp += 1   # increment waypoint index
	
            # If x_diff is greater than or equal to zero, use this as current closest waypoint
	    if x_diff >= 0.0:
	      flag_wp = True
	      rospy.logwarn('Current closest waypoint is (ind, x, y): %s, %f, %f ', i_wp, x_wp, y_wp)	
  	      # Set the final waypoints
	      self.set_final_waypoints(i_wp)
              
	      # Set the velocity for the waypoints
	      for wp in self.final_waypoints:
    	        self.set_final_waypoint_velocity(wp)
	      
	      # Publish waypoint list
	      final_waypoints_msg = Lane()      
	      final_waypoints_msg.header.stamp = rospy.Time.now()
	      final_waypoints_msg.waypoints = self.final_waypoints
	      self.final_waypoints_pub.publish(final_waypoints_msg)

    def set_final_waypoints(self, i_wp):
	# Set final_waypoints to include the current waypoint plus the next LOOKAHEAD waypoints
	self.final_waypoints = deepcopy(self.base_waypoints[i_wp : i_wp+LOOKAHEAD_WPS])

    def set_final_waypoint_velocity(self, waypoint):
	# Set the waypoint velocity    
	wp_velocity = self.get_waypoint_velocity(waypoint)
	self.set_waypoint_velocity2(waypoint, wp_velocity)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity2(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity
        #rospy.logwarn('Waypoint velocity set to: %f', velocity) 
    
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
