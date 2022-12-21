#!/usr/bin/env python3

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from utils import wrapToPi
import rospy
import tf

global waypoints
waypoints = np.array([[3.27,0.26,-0.03], [1.92,0.23,-3.11], [0.23,0.39,-1.78],[0.23,2.00,-1.78], [0.84,1.28,0.03], [2.26,1.80,1.98],[2.50,2.67,0.08],[1.10, 2.65, -0.027] ,[3.40,2.45,1.41], [3.15, 1.6, 0]]) #3rd last [1.10, 2.65, -0.027]

class GoalSetter(object):
    
    def __init__(self, pub):
        self.pub = pub
        self.waypoints = waypoints
        self.counter = 0
        self.curr = Pose2D()
        self.goal = Pose2D()
        self.near_thresh = 0.4
        self.at_thresh = 0.1
        self.at_thresh_theta = 0.05
        self.animal_input = []
        self.animal_list = []
        self.stored_msg_name_list = []
        self.stored_msg_loc_list = []
        self.requestFlag=False

    def get_yaw_from_quaternion(self, quat):
        return tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

    def callback(self,data):
        self.curr.x = data.pose.pose.position.x
        self.curr.y = data.pose.pose.position.y
        self.curr.theta = self.get_yaw_from_quaternion(data.pose.pose.orientation)
        # self.goal.theta = waypoints[self.counter, 2]
          
        if self.counter < len(self.waypoints):

            self.goal.x = self.waypoints[self.counter, 0]
            self.goal.y = self.waypoints[self.counter, 1]

            if(np.linalg.norm(np.array([self.curr.x - self.goal.x, self.curr.y - self.goal.y]))
            < self.at_thresh):
                #if self.goal.x == 1.92 and self.goal.y == 0.23:
                    #rospy.sleep(5)
                self.counter = self.counter+1
                if self.requestFlag:
                    rospy.loginfo('--------------Robot to Command center! {} rescued, over!-----------------='.format(self.animal_list[self.counter-1]))
                    rospy.sleep(10)
                else:
                    rospy.loginfo('-------------- Reached checkpoint!-----------------={}'.format(self.goal)) 
                #rospy.sleep(20)

            self.pub.publish(self.goal)
            rospy.loginfo('current goal is = {}'.format(self.goal))
            rospy.loginfo('current counter is = {}'.format(self.counter))  
        else:
            # rospy.signal_shutdown("Mapped the world") 
            self.requestFlag=True
            self.counter = 0
            new_waypoint_list = []

            self.animal_input = input("Robot to Command center! Waiting for names of animals to be rescued (comma separated, please!): ")
            self.animal_list = self.animal_input.split(",") 

            # Compare animal_list with stored dict and edit waypoints
            # set self.counter to 0
            for animal in self.animal_list:
                if animal in self.stored_msg_name_list: 
                    idx = self.stored_msg_name_list.index(animal)
                    new_waypoint_list.append(self.stored_msg_loc_list[idx])

            self.waypoints = np.array(new_waypoint_list)
            
            rospy.loginfo('----------Reset counter to 0-----------')


    def request_callback(self, data):
        name_list = []
        loc_list = []

        for ob_msg in data.ob_msgs:
            name_list.append(ob_msg.name)
            loc_list.append(ob_msg.location)

        self.stored_msg_name_list = name_list
        self.stored_msg_loc_list = np.array(loc_list)
        
def main():
    rospy.init_node('goal_setter')
    goal_pub = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)
    goal_set = GoalSetter(goal_pub)
    rospy.Subscriber("/detector/objects", DetectedObjectList, goal_set.request_callback)
    rospy.Subscriber("/odom", Odometry, goal_set.callback)
    rospy.spin()

if __name__== '__main__':
    main()



    # 1. Subscribe to DetectedObjectList
    # 2. In callback function, set detected_obvj_list = last obj message
    # 3. Once all goal points are done -> prompt user for animal list
    # 4. Split string into animals
    # 5. Use last obj message to find each animal's location
    # 6. Loop through each animal