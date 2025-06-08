#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist, Vector3
import numpy as np
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import JointState
import tf2_ros
import tf_conversions
import motor


class EKFHeightDepth:
    def __init__(self, f, H0, Z0, P0, Q, R):
        # camera focal length
        self.f = f
        # state vector
        self.x = np.array([H0, Z0], dtype=float)
        # state covariance
        self.P = P0.copy()
        # process covariance
        self.Q = Q.copy()
        # measurement covariance
        self.R = R.copy()

    def predict(self, delta_Z):
        u = np.array([0.0, delta_Z])
        self.x = self.x + u
        self.P = self.P + self.Q

    def update(self, h_meas, z_meas):
        H_pred, Z_pred = self.x

        # measured vector
        z_vector = np.array([h_meas, z_meas])

        # h prediction from state
        h_pred = self.f * H_pred / Z_pred
        # predicted vector
        z_pred_vec = np.array([h_pred, Z_pred])

        # measurement Jacobian
        H_jac = np.array([
            [ self.f / Z_pred,  -self.f * H_pred / (Z_pred**2)],  
            [ 0.0,              1.0                           ] 
        ])

        # innovation
        y = z_vector - z_pred_vec
        print(f"z vec: {z_vector} z pred vec: {z_pred_vec} test={h_meas*z_meas/self.f}")
        # innovation covariance
        S = H_jac @ self.P @ H_jac.T + self.R

        # kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        # state update
        self.x = self.x + K @ y

        # covariance update
        I = np.eye(2)
        self.P = (I - K @ H_jac) @ self.P

    def current_state(self):
        """Returns current estimates of (H, Z) and covariance P."""
        return self.x.copy(), self.P.copy()



latest_image = None


old_position = Point(0, 0, 0)
old_enc = 0
old_enc_wheel = 0
old_height = 0
old_width = 0

latest_enc = None
position_target = 4


def get_yaw_from_quaternion(q):
    """Extract yaw (rotation around Y-axis) from quaternion."""
    # Use tf_conversions to convert to Euler angles
    euler = tf_conversions.transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w]
    )
    # Usually wheels rotate around Y (pitch), but adjust based on your URDF
    return euler[1]  # euler[1] is pitch (rotation around Y)


# callbacks --------
def odom_callback(msg):
    global latest_odom
    try:
        # Save the odometry message to the global variable
        latest_odom = msg
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing odometry: {e}")

def encoder_callback(msg):
    global latest_enc
    try:
        # Save the odometry message to the global variable
        latest_enc = msg.data
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing encoder: {e}")


def setpoint_position_callback(msg):
    global position_target
    position_target = msg.data 
    


def image_callback(msg):
    global latest_image
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #rospy.loginfo("Received an image!")
        
        latest_image = cv_image
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":

    fl_motor = motor.DCMotor(gain=1.0, time_constant=0.5, rate_hz=100,
                             voltage_topic="/front_left_motor/command",
                             velocity_topic="/fl_motor_control/command")
    fr_motor = motor.DCMotor(gain=1.0, time_constant=0.5, rate_hz=100,
                             voltage_topic="/front_right_motor/command",
                             velocity_topic="/fr_motor_control/command")

    # subscribers
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/front_right_motor/encoder', Int32, encoder_callback)
    rospy.Subscriber('/setpoint_position', Float32, setpoint_position_callback)


    # publishers
    image_pub = rospy.Publisher('/filtered_keypoints_image', Image, queue_size=10)
    depth_pub = rospy.Publisher('/estimated_depth', Float32, queue_size=10)
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    fl_motor_cmd_pub = rospy.Publisher('/front_left_motor/command', Float32, queue_size=10)
    fr_motor_cmd_pub = rospy.Publisher('/front_right_motor/command', Float32, queue_size=10)




    rospy.init_node('image_analyzer', anonymous=True)
    rospy.set_param('/use_sim_time', True)
    bridge = CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    #detector = ObjectDetector()

    f = 1*554.26  # focal length 
    H0, Z0 = 0.8, 2.8  # initial guesses
    P0 = np.diag([0.5**2, 1**2])  # initial uncertainty
    Q = np.diag([1e-5, 8e-2]) # process noise 
    R = np.diag([3.0**2, 5.0**2]) # measurement of height and depth noise variance 

    ekf = EKFHeightDepth(f, H0, Z0, P0, Q, R)

    depth_msg = Float32()
    depth_msg.data = -3

    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    rospy.loginfo("Image Analyzer Node Started")
    orb = cv2.ORB_create(nlevels=3, scaleFactor=1.2, fastThreshold=20, nfeatures=1000)

    H_est, Z_est = H0, Z0

    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():

            rate.sleep()
            depth_pub.publish(depth_msg)
            if latest_image is not None:

                keypoints, descriptors = orb.detectAndCompute(latest_image, None)
                #rospy.loginfo(f"ORB feature extraction applied! Found {len(keypoints)} keypoints.")
                image_with_keypoints = cv2.drawKeypoints(latest_image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                #cv2.imshow("Image Viewer", latest_image)

                #cv2.imshow("Image Viewer Keypoints", image_with_keypoints)
                #cv2.waitKey(0)
                margin = 5 
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
                

                # bounding box
                if filtered_keypoints and latest_enc is not None:

                    x_coords = [kp.pt[0] for kp in filtered_keypoints]
                    y_coords = [kp.pt[1] for kp in filtered_keypoints]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))

                    #rospy.loginfo(f"Bounding box: Top-left=({x_min}, {y_min}), Bottom-right=({x_max}, {y_max})")

                    # Draw the bounding box on the image with filtered keypoints
                    image_with_filtered_keypoints = cv2.drawKeypoints(latest_image, filtered_keypoints, None, color=(255, 0, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
                    cv2.rectangle(image_with_filtered_keypoints, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

                    # Publish the image with filtered keypoints to a ROS topic
                    
                    try:
                        ros_image = bridge.cv2_to_imgmsg(image_with_filtered_keypoints, encoding="bgr8")
                        image_pub.publish(ros_image)
                    except Exception as e:
                        rospy.logerr(f"Error publishing image: {e}")


                    height = y_max - y_min

                    dz = -((2*3.141592*0.05)/(10.7*4096))*(latest_enc - old_enc)
                    dz_wheel = -((2*3.141592*0.05)/(10.7*4096))*(latest_enc - old_enc_wheel)
                    #print(f"dz: {dz} {latest_enc - old_enc} {latest_enc}")
                    #print(f"pos {latest_position.x} {latest_position.y} {latest_position.z} from {old_position.x} {old_position.y} {old_position.z}")
                    if abs(dz) > 0.1 and abs(old_height-height) > 2:

                        ekf.predict(dz)
                        ekf.update(height, height/(abs(old_height-height)/abs(dz)))
                        (H_est, Z_est), P_est = ekf.current_state()
                        print(f"current state: H_est={H_est:.2f}, Z_est={Z_est:.2f} (A)")

                        old_height = height
                        old_enc = latest_enc
                        old_enc_wheel = latest_enc
                    elif abs(dz_wheel) > 0.05:
                        ekf.predict(dz_wheel)
                        old_enc_wheel = latest_enc
                        (H_est, Z_est), P_est = ekf.current_state()

            depth_msg.data = -1*Z_est - 0.5 -0.4
            depth_pub.publish(depth_msg)

            fl_motor_cmd_pub.publish(Float32(-0.2*(position_target - Z_est)))
            fr_motor_cmd_pub.publish(Float32(-0.2*(position_target - Z_est)))


    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Image Analyzer Node")
    finally:
        cv2.destroyAllWindows()
