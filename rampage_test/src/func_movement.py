#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import *
from geometry_msgs.msg import *
from rampage_msgs.msg import *
 
#purpose of this program is to move rampage based on a parametric function, so we can test the steer and velocity
class TestMove:

    def __init__(self):
        rospy.init_node('rampage_test', anonymous=True)
        self.cmd_pub = rospy.Publisher('/uav_cmds', UavCmds, queue_size=1)
        self.traj_pub = rospy.Publisher('/trajectory', Path, queue_size=1)
        
        #initialize motors
        self.UAV_CMD_ON = 0x01
        self.UAV_DEV_MOTORS = 0x10
        self.UAV_CMD_WRITE = 0x04
        msg_uav_cmds = UavCmds()
        msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_ON)
        self.cmd_pub.publish(msg_uav_cmds)

    def velo(self, velo):
        return int(1500 + 31.32516*velo)

    def angVelo(self, ang):
        return int(1195.6*ang + 1500)

#    def servo(self, servo):
#        return int((servo - 2200.) * 0.000714295499404)            

    def stop(self):
        msg_uav_cmds = UavCmds()
        msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_WRITE)
        msg_uav_cmds.data.append(self.velo(0))
        msg_uav_cmds.data.append(self.angVelo(0))
        self.cmd_pub.publish(msg_uav_cmds)

    def move(self): 
        velInquiry = 0
        angInquiry = 20
#       servoInquiry = 90

        choice = int(input("Enter 0-velocity, 1-angular, 2-servo: "))
        if (choice == 0):
            velInquiry = int(input("Enter a velocity: "))
        elif (choice == 1): 
            angInquiry = int(input("Enter an angular velocity: "))
 #       elif (choice == 2):
  #          servoInquiry = int(input("Enter a servo angle: "))
        
        while not rospy.is_shutdown():
            startMoveTime = rospy.Time.now().to_sec()
            print(type(self.velo(velInquiry)))
            #while(rospy.Time.now().to_sec()-startMoveTime < 1000):
            print(rospy.Time.now().to_sec()-startMoveTime)  
            msg_uav_cmds = UavCmds()
            msg_uav_cmds.signature = (self.UAV_DEV_MOTORS | self.UAV_CMD_WRITE)
            msg_uav_cmds.data.append(self.velo(velInquiry))
            msg_uav_cmds.data.append(self.angVelo(angInquiry))
            self.cmd_pub.publish(msg_uav_cmds)

if  __name__ == '__main__':
    try:
        x = TestMove()
        x.move()
#        x.stop()
    except rospy.ROSInterruptException: pass
