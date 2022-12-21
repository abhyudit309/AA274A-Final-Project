#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from asl_turtlebot.msg import DetectedObjectList

class animal_marker:
    def __init__(self):
        self.vis_pub = rospy.Publisher('animal_location', Marker, queue_size=10)
    
    def marker_callback(self, data):
        id = len(data.ob_msgs)
        ob_msg = data.ob_msgs[-1]
        marker = Marker()

        marker.header.frame_id = "map"

        # IMPORTANT: If you're creating multiple markers, 
        # each need to have a separate marker ID.
        marker.id = id
        marker.type = 2 # sphere

        marker.pose.position.x = ob_msg.location[0]
        marker.pose.position.y = ob_msg.location[1]
        marker.pose.position.z = 1

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        self.vis_pub.publish(marker)

def main():
    am = animal_marker()
    rospy.init_node('animal_location_marker', anonymous=True)
    rospy.Subscriber("/detector/objects", DetectedObjectList, am.marker_callback)
    rospy.spin() 

if __name__ == '__main__':
    try:
        main() 
    except rospy.ROSInterruptException:
        pass
