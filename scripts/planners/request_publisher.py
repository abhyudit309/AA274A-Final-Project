#!/usr/bin/env python3

from nav_msgs.msg import Odometry
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from geometry_msgs.msg import Pose2D
import numpy as np
import rospy

class RequestPublisher(object):
    
    def __init__(self, pub, animal_list):
        self.pub = pub
        self.animal_list = animal_list
        self.counter = 0
        self.waypoints = []        
        self.curr = Pose2D()
        self.goal = Pose2D()
        self.tol = 0.1

    def location_callback(self, data):
        self.curr.x = data.pose.pose.position.x
        self.curr.y = data.pose.pose.position.y
        rospy.loginfo('Counter = {}'.format(self.counter))
        rospy.loginfo('Shape of Waypoints = {}'.format(len(self.waypoints)))
        self.goal.x = self.waypoints[self.counter][0]
        self.goal.y = self.waypoints[self.counter][1]

        if(np.abs(self.curr.x - self.goal.x) < self.tol and 
            np.abs(self.curr.y - self.goal.y) < self.tol):
            self.counter += 1
        
        if self.counter < len(self.waypoints):
            self.pub.publish(self.goal)
            rospy.loginfo('current goal is = {}'.format(self.goal))   
        else:
            rospy.signal_shutdown("All animals rescued!!") 

    def animal_callback(self, data):
        rospy.loginfo(data)
        for animal in self.animal_list:
            if animal in data.ob_msgs.name: 
                idx = data.ob_msgs.name.index(animal)
                rospy.loginfo(type(data.ob_msgs.location[idx]))
                self.waypoints.append(data.ob_msgs.location[idx])
                rospy.loginfo(type(self.waypoints))
            else:
                rospy.loginfo("Typo!!")   
        
def main():
    rospy.init_node('request_publisher', anonymous=True)
    animal_input = input("write the names of 3 animals separated by commas: ")
    animal_list = animal_input.split(",")
    goal_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
    rescue = RequestPublisher(goal_pub, animal_list)
    rospy.Subscriber("/detector/objects", DetectedObjectList, rescue.animal_callback)
    rospy.Subscriber("/odom", Odometry, rescue.location_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass