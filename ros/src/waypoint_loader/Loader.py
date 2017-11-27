#!/usr/bin/env python

import os
import csv
import math

class Loader(object):

    def __init__(self):
        self.velocity = 20.0


    def get_speed_limit(self):
        return self.velocity

if __name__ == '__main__':
    try:
        Loader()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint node.')
