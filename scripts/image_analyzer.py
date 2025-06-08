#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pynput import keyboard
import threading
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

# Global variables to store the latest image and edges
latest_image = None
key_pressed = None
stop_loop = False

fy = 554.26
fx = 554.26

# Initialize old_position as a ROS Point at (0, 0, 0)
old_position = Point(0, 0, 0)

# Initialize old_width and old_height
old_width = 0
old_height = 0

# Global variable to store the latest odometry data
latest_odom = None

def odom_callback(msg):
    global latest_odom
    try:
        # Save the odometry message to the global variable
        latest_odom = msg
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing odometry: {e}")

# Subscribe to the odometry topic
rospy.Subscriber('/odom', Odometry, odom_callback)

image_pub = rospy.Publisher('/filtered_keypoints_image', Image, queue_size=10)


def on_press(key):
    global key_pressed, stop_loop
    try:
        if key.char == 'q':
            stop_loop = True
        elif key.char == 's':
            key_pressed = 's'
    except AttributeError:
        pass  # Handle special keys if needed



def image_callback(msg):
    global latest_image
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #rospy.loginfo("Received an image!")
        
        # Save the image to the global variable
        latest_image = cv_image
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":
    rospy.init_node('image_analyzer', anonymous=True)
    bridge = CvBridge()


    # Subscribe to the image topic
    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    rospy.loginfo("Image Analyzer Node Started")
    orb = cv2.ORB_create(nlevels=3, scaleFactor=1.2, fastThreshold=20, nfeatures=1000)
    try:
        while not rospy.is_shutdown() and not stop_loop:
            if latest_image is not None:
                keypoints, descriptors = orb.detectAndCompute(latest_image, None)
                #rospy.loginfo(f"ORB feature extraction applied! Found {len(keypoints)} keypoints.")
                image_with_keypoints = cv2.drawKeypoints(latest_image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                #cv2.imshow("Image Viewer", latest_image)

                #cv2.imshow("Image Viewer Keypoints", image_with_keypoints)
                #cv2.waitKey(0)
                margin = 5  # Define a margin in pixels
                filtered_keypoints = []
                for kp in keypoints:
                    is_unique = True
                    for other_kp in keypoints:
                        if kp != other_kp:
                            distance = ((kp.pt[0] - other_kp.pt[0]) ** 2 + (kp.pt[1] - other_kp.pt[1]) ** 2) ** 0.5
                            if distance <= margin:
                                filtered_keypoints.append(kp)
                                break
                        

                #rospy.loginfo(f"Filtered keypoints: {len(filtered_keypoints)} remaining.")
                

                # Get the bounding box of the filtered keypoints
                if filtered_keypoints and latest_odom is not None:

                    latest_position = latest_odom.pose.pose.position

                    x_coords = [kp.pt[0] for kp in filtered_keypoints]
                    y_coords = [kp.pt[1] for kp in filtered_keypoints]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    rospy.loginfo(f"Bounding box: Top-left=({x_min}, {y_min}), Bottom-right=({x_max}, {y_max})")

                    # Draw the bounding box on the image with filtered keypoints
                    image_with_filtered_keypoints = cv2.drawKeypoints(latest_image, filtered_keypoints, None, color=(255, 0, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                    cv2.rectangle(image_with_filtered_keypoints, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

                    # Publish the image with filtered keypoints to a ROS topic
                    
                    try:
                        ros_image = bridge.cv2_to_imgmsg(image_with_filtered_keypoints, encoding="bgr8")
                        image_pub.publish(ros_image)
                    except Exception as e:
                        rospy.logerr(f"Error publishing image: {e}")

                    #cv2.imshow("Filtered Keypoints Viewer", image_with_filtered_keypoints)

                    width = x_max - x_min
                    height = y_max - y_min

                    dz = latest_position.x - old_position.x
                    #print(f"pos {latest_position.x} {latest_position.y} {latest_position.z} from {old_position.x} {old_position.y} {old_position.z}")
                    if abs(dz) > 0.3 and abs(old_height-height) > 0 and abs(old_width-width) > 0:
                        
                        true_height = abs(dz*height*old_height/(fy*(old_height-height)))
                        true_width = abs(dz*height*old_width/(fx*(old_width-width)))

                        print(f"size {true_width} {true_height} from dz: {dz} old_h: {old_height} h: {height}")

                        
                        old_width = width
                        old_height = height
                        old_position = latest_position

                    else:
                        print(f"dz is {dz} {(old_height-height)} {(old_width-width)}")

                    # Use the camera parameters to get real width and height from object
                    

                #cv2.waitKey(0)

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Image Analyzer Node")
    finally:
        cv2.destroyAllWindows()
