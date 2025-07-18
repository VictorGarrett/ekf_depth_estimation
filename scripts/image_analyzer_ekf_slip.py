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
        # camera focal length
        self.f = f
        # state vector
        self.x = np.array([H0, Z0, a0], dtype=float)
        # state covariance
        self.P = P0.copy()
        # process covariance
        self.Q = Q.copy()
        # measurement covariance
        self.R = R.copy()

        #self.Q_multiplier = np.array([8, 8, 5], dtype=float)
        self.Q_multiplier = np.array([1.8, 1.3, 0.3], dtype=float)
        self.multiplier_decay_rate = 0.95


    def predict(self, delta_Z):
        u = np.array([0.0, delta_Z, 0.0])
        _, _, a_pred = self.x

        self.x = self.x + u*a_pred

        F_jac = np.array([[1.0, 0.0   , 0.0    ],
                          [0.0, 1.0   , delta_Z],
                          [0.0, 0.0   , 1.0    ]
                          ])

        self.P = F_jac @ self.P @ F_jac.T + self.Q * (np.array([1, 1, 1], dtype=float)+self.Q_multiplier)
        self.Q_multiplier *= self.multiplier_decay_rate

    def update(self, h_meas, z_meas):
        H_pred, Z_pred, a_pred = self.x

        # measured vector
        z_vector = np.array([h_meas, z_meas])

        # h prediction from state
        h_pred = self.f * H_pred / Z_pred
        # predicted vector
        z_pred_vec = np.array([h_pred, Z_pred/a_pred])

        # measurement Jacobian
        H_jac = np.array([
                            [ self.f / Z_pred,  -self.f * H_pred / (Z_pred**2), 0.0                 ],  
                            [ 0.0            ,  1/a_pred                      , -Z_pred/(a_pred**2) ] 
                        ])

        # innovation
        y = z_vector - z_pred_vec
        rospy.loginfo(f"Innovation: {y}")
        
        print(f"z vec: {z_vector} z pred vec: {z_pred_vec} test={h_meas*z_meas/self.f}")
        try:
            inov_vector_msg = Vector3()
            inov_vector_msg.x = y[0]  # Innovation in height
            inov_vector_msg.y = y[1]  # Innovation in depth
            inov_vector_msg.z = 0.0   # Placeholder for unused dimension
            inov_pub.publish(inov_vector_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing innovation vector: {e}")
        # innovation covariance
        S = H_jac @ self.P @ H_jac.T + self.R[:2, :2]

        # kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        print(f"before update: {self.x}")
        # state update
        self.x = self.x + K @ y
        print(f"after update: {self.x}")

        # covariance update
        I = np.eye(3)
        self.P = (I - K @ H_jac) @ self.P

    def update_it(self, h_meas, z_meas, max_iter=40, epsilon=1e-5):
        z_vector = np.array([h_meas, z_meas])
        x_updated = self.x.copy()
        P_updated = self.P.copy()

        for i in range(max_iter):
            H_pred, Z_pred, a_pred = x_updated

            if Z_pred <= 0 or a_pred <= 0:
                rospy.logwarn("Invalid state encountered during IEKF iteration.")
                break

            # Measurement prediction
            h_pred = self.f * H_pred / Z_pred
            z_pred = Z_pred / a_pred
            z_pred_vec = np.array([h_pred, z_pred])

            # Innovation
            y = z_vector - z_pred_vec

            # Jacobian
            H_jac = np.array([
                [ self.f / Z_pred, -self.f * H_pred / (Z_pred**2), 0.0 ],
                [ 0.0,              1.0 / a_pred,                 -Z_pred / (a_pred**2) ]
            ])

            # Innovation covariance
            S = H_jac @ P_updated @ H_jac.T + self.R[:2, :2]
            if np.linalg.cond(S) > 1e12:
                rospy.logwarn("Innovation covariance S is ill-conditioned.")
                break

            # Kalman gain
            K = P_updated @ H_jac.T @ np.linalg.inv(S)

            # Compute update
            delta_x = K @ y
            new_x = x_updated + 0.5*delta_x

            # Check convergence
            if np.linalg.norm(delta_x) < epsilon:
                rospy.loginfo(f"IEKF converged in {i+1} iterations.")
                x_updated = new_x
                break

            x_updated = new_x

        # Finalize update
        self.x = x_updated

        # Final Jacobian and Kalman gain for covariance update
        H_pred, Z_pred, a_pred = self.x
        H_jac = np.array([
            [ self.f / Z_pred, -self.f * H_pred / (Z_pred**2), 0.0 ],
            [ 0.0,              1.0 / a_pred,                 -Z_pred / (a_pred**2) ]
        ])
        S = H_jac @ self.P @ H_jac.T + self.R[:2, :2]
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        # Optional: one last state correction (can be omitted if converged)
        final_y = z_vector - np.array([self.f * self.x[0] / self.x[1], self.x[1] / self.x[2]])
        self.x = self.x + K @ final_y

        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H_jac) @ self.P
    
    def update_a(self, a_meas):
        H_pred, Z_pred, a_pred = self.x

        # measured vector
        z_vector = np.array([a_meas])

        # predicted vector
        z_pred_vec = np.array([a_pred])

        # measurement Jacobian
        H_jac = np.array([
                            [ 0.0            ,  0.0                           ,  1.0                ] 
                        ])

        # innovation
        y = z_vector - z_pred_vec
        # innovation covariance
        S = H_jac @ self.P @ H_jac.T + self.R[2, 2]

        # kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        # state update
        self.x = self.x + K @ y

        # covariance update
        I = np.eye(3)
        self.P = (I - K @ H_jac) @ self.P

    def update_hz(self, h_meas, z_meas):
        H_pred, Z_pred, a_pred = self.x

        # measured vector
        z_vector = np.array([h_meas, z_meas])

        # h prediction from state
        h_pred = self.f * H_pred / Z_pred
        # predicted vector
        z_pred_vec = np.array([h_pred, Z_pred/a_pred])

        # measurement Jacobian
        H_jac = np.array([
                            [ self.f / Z_pred,  -self.f * H_pred / (Z_pred**2), 0.0                 ],  
                            [ 0.0            ,  1/a_pred                      , -Z_pred/(a_pred**2) ]
                        ])

        # innovation
        y = z_vector - z_pred_vec
        print(f"z vec: {z_vector} z pred vec: {z_pred_vec} test={h_meas*z_meas/self.f}")
        # innovation covariance
        S = H_jac @ self.P @ H_jac.T + self.R[:2, :2]

        # kalman gain
        K = self.P @ H_jac.T @ np.linalg.inv(S)

        # state update
        self.x = self.x + K @ y

        # covariance update
        I = np.eye(3)
        self.P = (I - K @ H_jac) @ self.P

    def current_state(self):
        """Returns current estimates of (H, Z) and covariance P."""
        return self.x.copy(), self.P.copy()
    
    def check_observability(self, delta_Z, tol=1e-8):
        """
        Computes the observability matrix at current EKF state and evaluates its rank,
        condition number, and singular values to assess numerical observability.

        Parameters:
        - delta_Z: control input (displacement in depth)
        - tol: tolerance for rank estimation
        - verbose: whether to print details

        Returns:
        - O: observability matrix
        - rank: estimated matrix rank
        - cond_number: condition number (L2)
        - singular_values: full list of singular values
        """

        # Current log-state
        H, Z, a = self.x

        # Linearized dynamics Jacobian (F)
        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1, delta_Z],
            [0.0, 0.0, 1.0]
        ])

        # Measurement Jacobian (H)
        H = np.array([
                            [ self.f / Z,  -self.f * H / (Z**2), 0.0                 ],  
                            [ 0.0            ,  1/Z                      , -Z/(a**2) ] 
                        ])

        # Observability matrix: [H; H*F; H*F^2]
        O = np.vstack([
            H,
            H @ F,
            H @ F @ F
        ])

        # Normalization for numerical robustness
        O_normalized = O / np.linalg.norm(O, ord='fro')

        # Singular values and condition number
        U, S, Vt = np.linalg.svd(O_normalized)
        cond_number = S[0] / S[-1] if S[-1] > 0 else np.inf

        # Numerical rank (with tolerance)
        rank = np.sum(S > tol)

        print("\n[EKF] Observability Matrix:\n", O)
        print("[EKF] Normalized Observability Matrix (Frobenius):\n", O_normalized)
        print("[EKF] Singular values:", S)
        print("[EKF] Condition number:", cond_number)
        print("[EKF] Estimated Rank:", rank)

        return O, rank, cond_number, S



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
    #Q = np.diag([4e-4*1.386, 8e-2*1.279, 2e-4*1.064]) # process noise 
    #R = np.diag([3.0**2, 10.0**2, 0.3**2]) # measurement of height and depth noise variance 

    H0, Z0, a0 = 0.92, 3.2, 0.7  # initial guesses
    P0 = np.diag([0.7**2, 0.5**2, 0.3**2])  # initial uncertainty
    Q = np.diag([8e-4*1.386, 6e-2*1.279, 2e-4*1.064]) # process noise 
    R = np.diag([3.0**2, 8.0**2, 0.3**2]) # measurement of height and depth noise variance 

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
                        ekf.check_observability(image_wheel_dz)
                        ekf.predict(image_wheel_dz)
                        ekf.update_it(height, height/(abs(old_height-height)/abs(image_wheel_dz)), 20)
                        
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
