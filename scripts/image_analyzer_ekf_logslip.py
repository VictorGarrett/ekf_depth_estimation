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
from sensor_msgs.msg import Imu
import tf2_ros
import tf_conversions
import math


# callbacks --------
def odom_callback(msg):
    global latest_odom
    try:
        # Save the odometry message to the global variable
        latest_odom = msg
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing odometry: {e}")

def imu_callback(msg):
    global imu_accel_buffer
    try:
        imu_accel_buffer.append((msg.linear_acceleration.x, rospy.get_time()))
        imu_accel_buffer.pop(0)

    except Exception as e:
        rospy.logerr(f"Error processing odometry: {e}")

def encoder_callback(msg):
    global latest_enc
    try:
        # Save the odometry message to the global variable
        latest_enc = (msg.data, rospy.get_time())
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


class EKFHeightDepth:
    def __init__(self, f, H0, Z0, a0, P0, Q, R):
        self.f = f
        # State: log(fH), log(Z), log(a)
        self.x = np.log(np.array([f * H0, Z0, a0], dtype=float))
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        self.Q_multiplier = np.array([1.8, 1.3, 0.3], dtype=float)
        self.multiplier_decay_rate = 0.95

    def predict(self, delta_Z):
        Hf_log, Z_log, a_log = self.x
        a = np.exp(a_log)
        Z = np.exp(Z_log)

        # Update Z in log-space using the additive motion model
        Z_new = Z + delta_Z * a
        self.x[1] = np.log(Z_new)  # update log(Z)

        # Compute Jacobian of the motion model
        F_jac = np.eye(3)
        F_jac[1, 1] = Z / Z_new
        F_jac[1, 2] = delta_Z * a / Z_new

        # Update covariance
        Q_scaled = self.Q * (np.ones(3) + self.Q_multiplier)
        self.P = F_jac @ self.P @ F_jac.T + Q_scaled
        self.Q_multiplier *= self.multiplier_decay_rate

    def update(self, h_meas, z_meas):
        # Skip update if measurements are invalid
        if h_meas <= 0 or z_meas <= 0:
            rospy.logwarn("Non-positive measurement encountered; skipping update.")
            return

        # Convert measurements to log-space
        z_log = np.array([
            np.log(h_meas),
            np.log(z_meas)
        ])

        # Measurement prediction in log-space
        Hf_log, Z_log, a_log = self.x
        h_pred_log = Hf_log - Z_log           # log(h) = log(fH) - log(Z)
        z_pred_log = Z_log - a_log            # log(z) = log(Z) - log(a)
        z_pred = np.array([h_pred_log, z_pred_log])

        # Measurement Jacobian
        H_jac = np.array([
            [1.0, -1.0,  0.0],
            [0.0,  1.0, -1.0]
        ])

        # Innovation
        y = z_log - z_pred

        # Innovation covariance
        S = H_jac @ self.P @ H_jac.T + self.R[:2, :2]
        if np.linalg.cond(S) > 1e12:
            rospy.logwarn("Innovation covariance S is ill-conditioned.")
            return

        # Kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        print(K)
        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H_jac) @ self.P

    def current_state(self):
        """Returns current estimates of (H, Z, a) and covariance P."""
        Hf, Z, a = np.exp(self.x)
        H = Hf / self.f
        return np.array([H, Z, a]), self.P.copy()



latest_image = None

# imu parameters
last_imu_mesage_time = 0
last_imu_update_time = 0
int_vel_value = 0.0
int_pos_value = 0.0

old_position = Point(0, 0, 0)
old_enc = (0, 0)
old_enc_wheel = 0
old_height = 0
old_width = 0

latest_enc = (0, 0)
position_target = 4

imu_accel_buffer = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]


def get_yaw_from_quaternion(q):
    """Extract yaw (rotation around Y-axis) from quaternion."""
    # Use tf_conversions to convert to Euler angles
    euler = tf_conversions.transformations.euler_from_quaternion(
        [q.x, q.y, q.z, q.w]
    )
    # Usually wheels rotate around Y (pitch), but adjust based on your URDF
    return euler[1]  # euler[1] is pitch (rotation around Y)


if __name__ == "__main__":

    # subscribers
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/imu/data', Imu, imu_callback)
    rospy.Subscriber('/front_right_motor/encoder', Int32, encoder_callback)
    rospy.Subscriber('/setpoint_position', Float32, setpoint_position_callback)

    # publishers
    image_pub = rospy.Publisher('/filtered_keypoints_image', Image, queue_size=10)
    depth_pub = rospy.Publisher('/estimated_depth', Float32, queue_size=10)
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    fl_motor_cmd_pub = rospy.Publisher('/front_left_motor/command', Float32, queue_size=10)
    fr_motor_cmd_pub = rospy.Publisher('/front_right_motor/command', Float32, queue_size=10)
    inov_pub = rospy.Publisher('/inovation', Vector3, queue_size=10)




    rospy.init_node('image_analyzer', anonymous=True)
    bridge = CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    #detector = ObjectDetector()
    
    last_imu_update_time = rospy.get_time()

    f = 1*554.26  # focal length 
    #H0, Z0, a0 = 1.08, 3.2, 0.9  # initial guesses (these work well somehow)
    #0.38634978 0.27903039 0.06439163
    #these work REALLY well
    #H0, Z0, a0 = 0.92, 3.2, 0.7  # initial guesses
    #P0 = np.diag([0.7**2, 0.5**2, 0.3**2])  # initial uncertainty
    #Q = np.diag([4e-3*1.386, 8e-2*1.279, 2e-4*1.064]) # process noise 
    #R = np.diag([3.0**2, 10.0**2, 0.3**2]) # measurement of height and depth noise variance 

    H0, Z0, a0 = 1, 3.0, 0.89  # initial guesses
    P0 = np.diag([2.7**2, 0.5**2, 1.3**2])  # initial uncertainty
    Q = np.diag([8e-3*1.386, 6e-2*1.279, 9e-3*1.064]) # process noise 
    R = np.diag([2.0**2, 8.0**2, 0.3**2]) # measurement of height and depth noise variance 

    ekf = EKFHeightDepth(f, H0, Z0, a0, P0, Q, R)

    depth_msg = Float32()
    depth_msg.data = -3

    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    rospy.loginfo("Image Analyzer Node Started")
    orb = cv2.ORB_create(nlevels=3, scaleFactor=1.2, fastThreshold=20, nfeatures=1000)

    H_est, Z_est, a_est = H0, Z0, a0

    imu_wheel_dz = 0
    image_wheel_dz = 0

    wheel_velocity_buffer = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    last_mean_wheel_velocity = 0.0
    velocity = 0.0

    rate = rospy.Rate(30)

    try:
        while not rospy.is_shutdown():

            rate.sleep()
            depth_pub.publish(depth_msg)
            position_target = 3 + math.sin(0.6* rospy.get_time()) + 0.7*math.sin(0.2* rospy.get_time())


            dz = -((2*3.141592*0.05)/(4096))*(latest_enc[0] - old_enc[0])
            #print(dz)

            if (latest_enc[1] - old_enc[1]) > 0:
                new_velocity = dz / (latest_enc[1] - old_enc[1])

                velocity = 0.3*new_velocity + 0.7*velocity


                wheel_velocity_buffer.append((velocity, latest_enc[1]))
                wheel_velocity_buffer.pop(0)


                #print(f"{wheel_velocity_buffer[0]}  {wheel_velocity_buffer[1]}  {wheel_velocity_buffer[2]}")



            old_enc = latest_enc

            # displacement since last image update
            image_wheel_dz += dz


            # prediction step, should heppen almost every loop
            #if abs(dz) > 0:
            #    ekf.predict(dz)
            #    (H_est, Z_est, a_est), P_est = ekf.current_state()
                #print(f"current state: H_est={H_est:.2f}, Z_est={Z_est:.2f}, a_est={a_est:.2f}, p={P_est} (P)")
            
            # imu slip update
            if rospy.get_time() - last_imu_update_time > 1000000000.0:

                vf, tf = wheel_velocity_buffer[2]
                vi, ti = wheel_velocity_buffer[1]

                wheel_accel = (vf - vi) / (tf - ti)
                print(f"Wheel final velocity (vf): {vf:.4f}, Time (tf): {tf:.4f}")
                print(f"Wheel initial velocity (vi): {vi:.4f}, Time (ti): {ti:.4f}")
                print(f"Wheel acceleration (wheel_accel): {wheel_accel:.4f}")
                print(f"IMU acceleration buffer: {imu_accel_buffer}")


                print(f"Calculated wheel acceleration: {wheel_accel:.4f} on {tf}, Last IMU acceleration: {imu_accel_buffer[2][0]:.4f} on {imu_accel_buffer[2][1]}")

                
            # image height and depth updates
            if abs(image_wheel_dz) > 0.1 and latest_image is not None:

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

                    if abs(old_height-height) > 2:
                        ekf.predict(image_wheel_dz)
                        ekf.update(height, height/(abs(old_height-height)/abs(image_wheel_dz)))
                        
                        (H_est, Z_est, a_est), P_est = ekf.current_state()

                        old_height = height
                        image_wheel_dz = 0.0
                        print(f"current state: H_est={H_est:.2f}, Z_est={Z_est:.2f}, a_est={a_est:.2f}, Q_mult={ekf.Q_multiplier} (H)")
            
            

            depth_msg.data = -1*Z_est - 0.5 -0.4
            depth_pub.publish(depth_msg)

            fl_motor_cmd_pub.publish(Float32(-0.5*(position_target - Z_est)))
            fr_motor_cmd_pub.publish(Float32(-0.5*(position_target - Z_est)))

            #fl_motor_cmd_pub.publish(Float32(-0.4))
            #fr_motor_cmd_pub.publish(Float32(-0.4))


    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Image Analyzer Node")
    finally:
        cv2.destroyAllWindows()
