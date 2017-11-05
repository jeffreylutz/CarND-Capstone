from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import math
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        vehicle_mass   = kwargs['vehicle_mass']
	fuel_capacity  = kwargs['fuel_capacity']
	brake_deadband = kwargs['brake_deadband']
	decel_limit    = kwargs['decel_limit']
	accel_limit    = kwargs['accel_limit']
	wheel_radius   = kwargs['wheel_radius']
	wheel_base     = kwargs['wheel_base']
	steer_ratio    = kwargs['steer_ratio']
	max_lat_accel  = kwargs['max_lat_accel']
	max_steer_angle = kwargs['max_steer_angle']	
	min_speed       = kwargs['min_speed']	

	# Initialize variables
	self.previous_time = None

	# Yaw control setup
	yaw_params = [wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle]
	self.yaw_controller = YawController(*yaw_params)

	# PID setup for throttle and brake
	self.pid_throttle = PID(0.35, 0.0, 0.0, 0.0, 1.0)
	self.pid_brake = PID(0.3, 0.0, 0.0, 0.0, 1.0)

	# Low pass filter for steering
	self.lowpass = LowPassFilter(0.2, 1.0)

	pass

    def update_sample_step(self):
	current_time = rospy.get_time()
	time_step = current_time - self.previous_time if self.previous_time else 0.2
	self.previous_time = current_time
	return time_step

    def control(self, linear_velocity, angular_velocity, current_linear_vel, dbw_enabled):

	# If manual, reset controllers and return zeroes for throttle, brake, and steering        
	if not dbw_enabled:
	  # reset()
	  return 0.0, 0.0, 0.0

	# Find throttle, brake and steering commands
	# Update time_step
	time_step = self.update_sample_step()	

	# Calculate velocity error
	velocity_error = linear_velocity - current_linear_vel
	
	# Throttle and brake commands using PID control 
	throttle = self.pid_throttle.step(velocity_error, time_step)
	brake = self.pid_brake.step(-velocity_error, time_step)

	# Yaw controller for steering
	steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_linear_vel)
	rospy.logwarn('Angular velocity target: %s', angular_velocity)
	rospy.logwarn('Steering command: %s', steering)
	# Smooth steering control
	steering = self.lowpass.filt(steering)


	return throttle, brake, steering
