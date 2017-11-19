#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

	# initialize min distance
	min_dist = float("inf")

	pose1 = pose.position # Car state
	ind = 0

	# loop through waypoints to check for closest
	for wp in self.waypoints.waypoints:	
	    ind += 1
	    pose2 = wp.pose.pose.position 
	    dist = self.distance(pose1, pose2)
	    if dist < min_dist:
		min_dist = dist
		closest_wp = wp
		ind_closest = ind
	return ind_closest, closest_wp 

    def distance(self, pose1, pose2):
	""" Return the distance between two points
	"""
	xdiff = pose1.x - pose2.x
	ydiff = pose1.y - pose2.y
    	zdiff = pose1.z - pose2.z
        dist = math.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
	return dist

    def get_closest_light(self, pose):
        """Identifies the closest light waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.lights

        """
	# initialize min distance
	min_dist = float("inf")

	pose1 = pose.position # closest waypoint
	ind = 0

	# loop through light waypoints to check for closest
	for wp in self.lights:
	    ind += 1	
	    pose2 = wp.pose.pose.position 
	    dist = self.distance(pose1, pose2)
	    if dist < min_dist:
		min_dist = dist
		closest_wp = wp
		ind_closest = ind
	return ind_closest 


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

	# If self.waypoints is not defined, return the default no light message
	if (self.waypoints == None):
	    return -1, TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
	# Find closest waypoint to car
	#if(self.pose):
            #ind_wp_closest_to_car, wp_closest_to_car = self.get_closest_waypoint(self.pose.pose)

        # Find the closest visible traffic light (if one exists). 
	    #light_wp = self.get_closest_light(wp_closest_to_car.pose.pose)
	
	# TODO 
	# In init, go through stop_line_positions and find closest wp to each
	# At each time step, check current post vs the stop line waypoint set (8 pts)
	#   - transform to local frame
	#   - find closest wp in front
	#   - if wp within tolerance, return waypoint and light color

	# Check light state
        if light:
	    state = light.state # for testing, CHANGE! 
            #state = self.get_light_state(light)
	    rospy.logwarn("light waypoint %s, state %s", light_wp, state)
            #return light_wp, state
	    return wp_closest_to_car, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
