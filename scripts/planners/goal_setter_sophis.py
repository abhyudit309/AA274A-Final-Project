#!/usr/bin/env python3

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
import rospy
import tf

class GoalSetter(object):
    
    def __init__(self, pub, waypoints, counter=0):
        self.pub = pub
        self.waypoints = waypoints
        self.counter = counter
        self.curr = Pose2D()
        self.goal = Pose2D()
        self.tol = 0.1

    def get_yaw_from_quaternion(self, quat):
        return tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

    def callback(self,data):
        self.curr.x = data.pose.pose.position.x
        self.curr.y = data.pose.pose.position.y
        self.curr.theta = self.get_yaw_from_quaternion(data.pose.pose.orientation)
        self.goal.x = self.waypoints[self.counter, 0]
        self.goal.y = self.waypoints[self.counter, 1]
        self.goal.theta = np.pi/2
        if(np.abs(self.curr.x - self.goal.x) < self.tol and 
            np.abs(self.curr.y - self.goal.y) < self.tol):
            self.counter += 1
        
        if self.counter < len(self.waypoints):
            self.pub.publish(self.goal)
            rospy.loginfo('current goal is = {}'.format(self.goal))   
        else:
            rospy.signal_shutdown("Mapped the world")      
        
def main():
    rospy.init_node('goal_setter',anonymous=True)
    # waypoints = np.array([[3.365, 2.726 ,0],[2.517, 2.839, 0],[0.48, 2.63, 0], [0.17, 2.08, 0], [0.2014, 0.382, 0], [2.39, 0.35, 0], [1.35, 1.56, 0], [2.29, 1.6, 0], [3.44, 0.32, 0], [3.15, 1.6, 0]])
    waypoints = np.array([[3.15, 1.6], [3.45, 2.54], [3.11, 2.76], [1.41, 2.85], [0.23, 2.39], [0.165, 1], [0.15, 0.6],
    [0.42, 0.34], [2.26, 0.38], [2.45, 1.1], [2.06, 1.59], [1.15, 1.57], [2.06, 1.59], [2.45, 1.1], [2.26, 0.38], [3.08, 0.18], [3.39, 0.36], [3.15, 1.6]])
    goal_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
    goal_set = GoalSetter(goal_pub, waypoints)
    rospy.Subscriber("/odom", Odometry, goal_set.callback)
    rospy.spin()

if __name__== '__main__':
    main()