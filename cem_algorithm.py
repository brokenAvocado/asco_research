#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import *
from geometry_msgs.msg import *
from rampage_msgs.msg import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
import time

#algorithm outputs the state and dt of the rampage, utilizes CEM
class CEMLearning:
    def __init__(self):
        rospy.init_node('rampage_algorithm', anonymous=True)
        self.odo_sub = rospy.Subscriber('/ ', , queue_size=1)
        self.imu_sub = rospy.Subscriber('/ ', , queue_size=1)
        self.rampage_pub = rospy.Publisher('/ ', , queue_size=1)
    def deriv(self, ):
        #convert the velocities to acclerations, 

    def hist(self, ):

    def calc(self, ):
        #do the CEM here
        rampage_pub.publish() #output the two accls and dt
