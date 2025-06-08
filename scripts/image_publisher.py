#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publish_image():
    rospy.init_node('test_image_publisher')
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
    bridge = CvBridge()

    # Load test image (change the path to your test image)
    image_path = "/home/garrett/drone/workspace/src/tracker-control/lab2.jpg"
    cv_image = cv2.imread(image_path)

    if cv_image is None:
        rospy.logerr("Failed to load image!")
        return

    ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

    rate = rospy.Rate(1)  # 1 Hz (1 time per second)
    while not rospy.is_shutdown():
        pub.publish(ros_image)
        rospy.loginfo("Image published.")
        rate.sleep()

if __name__ == '__main__':
    publish_image()
