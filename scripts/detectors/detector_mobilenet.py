#!/usr/bin/env python3

import rospy
import os

# watch out on the order for the next two imports lol
# from tf import TransformListener
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import torch
import torchvision
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import numpy as np
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, LaserScan
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
import cv2
import math
import playsound

# path to the trained conv net
PATH_TO_LABELS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../../tfmodels/coco_labels.txt",
)

detected_objects = DetectedObjectList()

# set to True to use tensorflow and a conv net
# False will use a very simple color thresholding to detect stop signs only
USE_PYTORCH = True
# minimum score for positive detection
MIN_SCORE = 0.55


def load_object_labels(filename):
    """loads the coco object readable name"""

    fo = open(filename, "r")
    lines = fo.readlines()
    fo.close()
    object_labels = {}
    for l in lines:
        object_id = int(l.split(":")[0])
        label = (
            l.split(":")[1][1:]
            .replace("\n", "")
            .replace("-", "_")
            .replace(" ", "_")
        )
        object_labels[object_id] = label

    return object_labels


class Detector:
    def __init__(self):
        rospy.init_node("turtlebot_detector", anonymous=True)
        self.bridge = CvBridge()

        # create note for Marker publish
        self.vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
        self.sound_publisher = rospy.Publisher("/detector/sound", String, queue_size=10)
        #rospy.init_node('marker_node', anonymous=True)

        self.detected_objects_pub = rospy.Publisher(
            "/detector/objects", DetectedObjectList, queue_size=10
        )
        if USE_PYTORCH:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.5)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
            self.preprocess = weights.transforms()

        # camera and laser parameters that get updated
        self.cx = 0.0
        self.cy = 0.0
        self.fx = 1.0
        self.fy = 1.0
        self.laser_ranges = []
        self.laser_angle_increment = 0.01  # this gets updated

        self.object_publishers = {}
        self.object_labels = load_object_labels(PATH_TO_LABELS)
        self.odom_msg_temp = []
        rospy.Subscriber(
            "/camera/image_raw/compressed",
            CompressedImage,
            self.compressed_camera_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            "/camera/camera_info", CameraInfo, self.camera_info_callback
        )
        rospy.Subscriber("/scan", LaserScan, self.laser_callback) 

        rospy.Subscriber('/odom', Odometry, self.location_callback)

    def give_odom_to_location(self, object_msg):
        object_msg.location = self.odom_msg_temp
        
    def location_callback(self, odom_data):
        self.odom_msg_temp = [odom_data.pose.pose.position.x, odom_data.pose.pose.position.y] 
            
      
    def run_detection(self, img):
        """runs a detection method in a given image"""

        image_np = self.load_image_into_numpy_array(img)

        if USE_PYTORCH:
            # uses MobileNet to detect objects in images
            # this works well in the real world, but requires
            # good computational resources
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch = [self.preprocess(torch.from_numpy(image_np).permute(2,0,1).cuda())]
                else:
                    batch = [self.preprocess(torch.from_numpy(image_np).permute(2,0,1))]
                prediction = self.model(batch)[0]
                boxes = prediction["boxes"].cpu().numpy()
                scores = prediction["scores"].cpu().numpy()
                classes = prediction["labels"].cpu().numpy()
            return self.filter(boxes, scores, classes, classes.shape[0])

        else:
            # uses a simple color threshold to detect stop signs
            # this will not work in the real world, but works well in Gazebo
            # with only stop signs in the environment
            R = image_np[:, :, 0].astype(np.int) > image_np[:, :, 1].astype(
                np.int
            ) + image_np[:, :, 2].astype(np.int)
            (
                Ry,
                Rx,
            ) = np.where(R)
            if len(Ry) > 0 and len(Rx) > 0:
                xmin, xmax = Rx.min(), Rx.max()
                ymin, ymax = Ry.min(), Ry.max()
                boxes = [
                    [
                        float(ymin) / image_np.shape[1],
                        float(xmin) / image_np.shape[0],
                        float(ymax) / image_np.shape[1],
                        float(xmax) / image_np.shape[0],
                    ]
                ]
                scores = [0.99]
                classes = [13]
                num = 1
            else:
                boxes = []
                scores = 0
                classes = 0
                num = 0

            return boxes, scores, classes, num

    def filter(self, boxes, scores, classes, num):
        """removes any detected object below MIN_SCORE confidence"""

        f_scores, f_boxes, f_classes = [], [], []
        f_num = 0

        for i in range(int(num)):
            if scores[i] >= MIN_SCORE:
                f_scores.append(scores[i])
                f_boxes.append(boxes[i])
                f_classes.append(int(classes[i]))
                f_num += 1
            else:
                break

        return f_boxes, f_scores, f_classes, f_num

    def load_image_into_numpy_array(self, img):
        """converts opencv image into a numpy array"""

        (im_height, im_width, im_chan) = img.shape

        return (
            np.array(img.data)
            .reshape((im_height, im_width, 3))
            .astype(np.uint8)
        )

    def project_pixel_to_ray(self, u, v):
        """takes in a pixel coordinate (u,v) and returns a tuple (x,y,z)
        that is a unit vector in the direction of the pixel, in the camera frame.
        This function access self.fx, self.fy, self.cx and self.cy"""

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        norm = math.sqrt(x * x + y * y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm

        return (x, y, z)

    def estimate_distance(self, thetaleft, thetaright, ranges):
        """estimates the distance of an object in between two angles
        using lidar measurements"""

        leftray_indx = min(
            max(0, int(thetaleft / self.laser_angle_increment)), len(ranges)
        )
        rightray_indx = min(
            max(0, int(thetaright / self.laser_angle_increment)), len(ranges)
        )

        if leftray_indx < rightray_indx:
            meas = ranges[rightray_indx:] + ranges[:leftray_indx]
        else:
            meas = ranges[rightray_indx:leftray_indx]

        num_m, dist = 0, 0
        for m in meas:
            if m > 0 and m < float("Inf"):
                dist += m
                num_m += 1
        if num_m > 0:
            dist /= num_m

        return dist

    def camera_callback(self, msg):
        """callback for camera images"""

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.camera_common(img_laser_ranges, img, img_bgr8)

    def compressed_camera_callback(self, msg):
        """callback for camera images"""

        # save the corresponding laser scan
        img_laser_ranges = list(self.laser_ranges)

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.camera_common(img_laser_ranges, img, img_bgr8)

    def camera_common(self, img_laser_ranges, img, img_bgr8):
        (img_h, img_w, img_c) = img.shape
        # runs object detection in the image
        (boxes, scores, classes, num) = self.run_detection(img)

        if num > 0:
            # some objects were detected
            for (box, sc, cl) in zip(boxes, scores, classes):
                ymin = int(box[1])
                xmin = int(box[0])
                ymax = int(box[3])
                xmax = int(box[2])
                xcen = int(0.5 * (xmax - xmin) + xmin)
                ycen = int(0.5 * (ymax - ymin) + ymin)

                cv2.rectangle(
                    img_bgr8, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2
                )

                # computes the vectors in camera frame corresponding to each sides of the box
                rayleft = self.project_pixel_to_ray(xmin, ycen)
                rayright = self.project_pixel_to_ray(xmax, ycen)

                # convert the rays to angles (with 0 poiting forward for the robot)
                thetaleft = math.atan2(-rayleft[0], rayleft[2])
                thetaright = math.atan2(-rayright[0], rayright[2])
                if thetaleft < 0:
                    thetaleft += 2.0 * math.pi
                if thetaright < 0:
                    thetaright += 2.0 * math.pi

                # estimate the corresponding distance using the lidar
                dist = self.estimate_distance(
                    thetaleft, thetaright, img_laser_ranges
                )

                if not cl in self.object_publishers:
                    self.object_publishers[cl] = rospy.Publisher(
                        "/detector/" + self.object_labels[cl],
                        DetectedObject,
                        queue_size=10,
                    )

                # publishes the detected object and its location
                object_msg = DetectedObject()
                object_msg.id = cl
                object_msg.name = self.object_labels[cl]
                object_msg.confidence = sc
                object_msg.distance = dist
                object_msg.thetaleft = thetaleft
                object_msg.thetaright = thetaright
                object_msg.corners = [ymin, xmin, ymax, xmax]
                self.give_odom_to_location(object_msg)
                # Adding the location in the msg type
                ## Adding a new subscriber

                rospy.loginfo("Object Name is: {}".format(self.object_labels[cl]))

                animal_options_list = ["cat", "dog", "horse", "elephant", "zebra"]
                animal_sound_list = ["meow", "woof", "neigh", "trumpet", "bray"]
                animal_sound_path_list = ["../../sounds/cat-meow.mp3", "../../sounds/dog-woof.mp3", "../../sounds/horse-neigh.mp3", "../../sounds/sheep-baa.mp3", "../../sounds/zebra-bray.mp3"]

                if((object_msg.name in animal_options_list) and object_msg.confidence >= MIN_SCORE):
                    self.object_publishers[cl].publish(object_msg)                  

                    # add detected object to detected objects list
                    if(object_msg.name not in detected_objects.objects):
                        detected_objects.objects.append(self.object_labels[cl])
                        detected_objects.ob_msgs.append(object_msg)

                        # publish animal sound based on detected animal
                        sound_idx = animal_options_list.index(object_msg.name)
                        self.sound_publisher.publish(animal_sound_list[sound_idx])

                        # play animal sound based on detected animal
                        #playsound(animal_sound_path_list[sound_idx])
                elif(object_msg.name == 'stop_sign'):                    
                    self.object_publishers[cl].publish(object_msg)
                    
            self.detected_objects_pub.publish(detected_objects)

        # displays the camera image
        cv2.imshow("Camera", img_bgr8)
        cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """extracts relevant camera intrinsic parameters from the camera_info message.
        cx, cy are the center of the image in pixel (the principal point), fx and fy are
        the focal lengths. Stores the result in the class itself as self.cx, self.cy,
        self.fx and self.fy"""

        if any(msg.P):
            self.cx = msg.P[2]
            self.cy = msg.P[6]
            self.fx = msg.P[0]
            self.fy = msg.P[5]
        else:
            rospy.loginfo("`CameraInfo` message seems to be invalid; ignoring it.")

    def laser_callback(self, msg):
        """callback for thr laser rangefinder"""

        self.laser_ranges = msg.ranges
        self.laser_angle_increment = msg.angle_increment

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    d = Detector()
    d.run()
