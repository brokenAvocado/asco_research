#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import *
from geometry_msgs.msg import *
from rampage_msgs.msg import *

import math

# utilizes state from algorithm giving program in order to convert to movement, steer and velocity output from algirthm
class controls:

    def __init__(self):
        rospy.init_node('rampage_test', anonymous=True)
        self.cmd_pub = rospy.Publisher('/uav_cmds', UavCmds, queue_size=1)
        self.traj_pub = rospy.Publisher('/trajectory', Path, queue_size=1)
        self.states_sub = rospy.Subscriber('/ ', , queue_size = 1)
        self.gps_sub = rospy.Subscriber('/ ', , queue_size = 1)

        #initialize motors
        self.UAV_CMD_ON = 0x01
        self.UAV_DEV_MOTORS = 0x10
        self.UAV_CMD_WRITE = 0x04
        msg_uav_cmds = UavCmds()
        msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_ON)
        self.cmd_pub.publish(msg_uav_cmds)

        #retrieve states and sensor data (note: will need dt in order to know how much displacement to travel)
        self.accl_x, self.accl_y, self.dt = self.states_sub #will give x_dots, most likely the velocities in both axis, accl for both axis, heading, and rotational velocity
        self.init_x, self.init_y = self.gps_sub
        self.dt = 
        self.pos = np.array([self.init_x, self.init_y])

    def currentPos(self, pos_x, pos_y, vel_x, vel_y):
        posNew_x = pos_x + vel_x * self.dt
        posNew_y = pos_y + vel_y * self.dt
        return posNew_x, posNew_y

    def vel(self, accl_x, accl_y):
        vel = ((accl_x*self.dt)^2 + (accl_y*self.dt)^2)^0.5
        return int(1500 + 31.32516*vel)

    def angVel(self, accl_x, accl_y):
        angvel = math.atan(accl_y, accl_x)/self.dt #does something give the angvel
        return int(1195.6*angvel + 1500)

    def stop(self):
        msg_uav_cmds = UavCmds()
        msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_WRITE)
        msg_uav_cmds.data.append(self.velo(0))
        msg_uav_cmds.data.append(self.angVelo(0))
        self.cmd_pub.publish(msg_uav_cmds)

    def move(self):
        i = 0
        j = self.accl_x.length[0]
        isActive = True
        while not rospy.is_shutdown():
            while(isActive):
                startMoveTime = rospy.Time.now().to_sec()
                x = init_x, y = init_y
                if ((rospy.Time.now().to_sec() - startMoveTime) == dt && i <= j)
                    xNew, yNew = self.currentPos(x, y, vel_x[i], vel_y[i])
                    self.pos = np.append([self.pos], [xNew, yNew], axis = 0)
                    msg_uav_cmds = UavCmds()
                    msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_WRITE)
                    msg_uav_cmds.data.append(self.velo(self.accl_x[i], self.accl_y[i]))
                    msg_uav_cmds.data.append(self.angVelo(self.accl_x[i], self.accl_y[i]))
                    self.cmd_pub.publish(msg_uav_cmds)
                    x = xNew, y = yNew
                    i++
                self.stop
        
        print(rospy.Time.now().to_sec() - startMoveTime)
            self.stop()
