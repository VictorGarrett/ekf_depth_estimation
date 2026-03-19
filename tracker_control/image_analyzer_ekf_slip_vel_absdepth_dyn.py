#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist, Vector3
import numpy as np
from std_msgs.msg import Float32, Float64, Int32
import tf2_ros
import math
from collections import deque

# Assuming motor_controller.py is ported to ROS 2 similarly
from tracker_control.motor_controller import PIController 

class EKFHeightDepth:
    def __init__(self, f, r, b, m, k, j, res, l, gr, H0, Z0, a0, v0, i0, w0, theta0, P0, Q, R):
        self.f, self.r, self.b, self.m, self.k = f, r, b, m, k
        self.j, self.res, self.l, self.gr = j, res, l, gr

        self.x = np.array([H0, Z0, a0, v0, i0, w0, theta0], dtype=float)
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.Q_multiplier = np.array([0.0014, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01], dtype=float)
        self.multiplier_decay_rate = 0.998

    def dynamics(self, x, u):
        H, Z, a, v, i, w, theta = x
        J_eq = self.j + self.m * self.r**2 / (2 * self.gr**2)
        b_eq = self.b 
        
        dx = np.zeros_like(x)
        dx[0] = 0.0  # H_dot
        dx[1] = v    # Z_dot
        dx[2] = 0.0  # a_dot
        dx[3] = - (a * self.r / self.gr) * (self.k * i - b_eq * w) / J_eq
        dx[4] = 0.0  # i_dot
        dx[5] = (self.k * i - b_eq * w) / J_eq
        dx[6] = w / self.gr
        return dx

    def predict(self, u, dt, current_time):


        dx = self.dynamics(self.x, u)
        self.x[4] = (u - self.k * self.x[5]) / self.res
        self.x = dx * dt + self.x
        
        # Jacobian and Covariance Prediction
        J_eq = self.j + self.m * self.r**2 / 2
        b_eq = self.b

        #w
        dwdot_di = (self.gr**2 * self.k) / J_eq
        dwdot_dw = -b_eq / J_eq
        
        #v
        w_dot_val = (self.gr**2 * self.k * self.x[4] - b_eq * self.x[5]) / J_eq
        dvdot_da = - (self.r / self.gr) * w_dot_val
        dvdot_di = - (self.x[2] * self.r / self.gr) * dwdot_di
        dvdot_dw = - (self.x[2] * self.r / self.gr) * dwdot_dw

        #i
        di_dw = -self.k / self.res
        
        F_jac = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, dt, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, dvdot_da*dt, 1.0, dvdot_di*dt, dvdot_dw*dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, di_dw, 0.0],
            [0.0, 0.0, 0.0, 0.0, dwdot_di*dt, 1.0 + dwdot_dw*dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, self.gr*dt, 1.0]
        ])
        
        Q_scaled = self.Q * dt * (np.ones(7) + self.Q_multiplier)
        self.P = F_jac @ self.P @ F_jac.T + Q_scaled
        self.P = np.clip(self.P, -1e3, 1e3)
        self.Q_multiplier *= self.multiplier_decay_rate

    def update(self, h_meas, H_meas, w_meas, theta_meas):

        print(f"Measurement: h={h_meas:.3f}, H={H_meas:.3f}, w={w_meas:.3f}, theta={theta_meas:.3f}")
        y_meas = np.array([h_meas,
                           H_meas,
                           w_meas,
                           theta_meas])

        H, Z, a, v, i, w, theta = self.x

        y_pred = np.array([H*self.f/Z,
                           H,
                           w/self.gr,
                           theta])

        H_jac = np.array([
            [self.f/Z, -H*self.f/(Z**2), 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1/self.gr, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        y = y_meas - y_pred


        S = H_jac @ self.P @ H_jac.T + self.R

        if np.linalg.cond(S) < 1e12:
            K = self.P @ H_jac.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(7) - K @ H_jac) @ self.P
            print(f"Updated State: H={self.x[0]:.3f}, Z={self.x[1]:.3f}, a={self.x[2]:.3f}, v={self.x[3]:.3f}, i={self.x[4]:.3f}, w={self.x[5]:.3f}, theta={self.x[6]:.3f}")
        else:
            print("S degenerated, skipping update")
        return y

class ImageAnalyzer(Node):
    def __init__(self):
        super().__init__('image_analyzer')
        
        # --- Parameters & State ---
        self.bridge = CvBridge()
        self.latest_odom = None
        self.latest_enc = (0, 0.0) # (data, time)
        self.latest_height_data = (0, 0.0, 0) # (h, time, dist_to_center)
        self.height_used = True
        self.position_target = 3.1
        self.old_enc = (0, 0.0)
        self.wheel_velocity_buffer = deque([(0.0, 0.0)] * 3, maxlen=3)
        self.velocity = 0.0

        # Physics Constants
        f = 554.26
        gr = 10
        r = 0.0525
        b = 0.00011
        m = 12.81
        k = 0.021
        j = 0.00011
        res = 1.05
        l = 0.001
        
        # EKF Initialization
        H0, Z0, a0, v0, i0, w0, theta0 = 1.0, 3.1, 0.95, 0.0, 0.0, 0.0, 0.0
        P0 = np.diag([0.2**2, 1.2**2, 0.6**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2])
        Q = np.diag([4e-6, 0.15**2, 9e-3, 0.55**2, 9e-2, 9e-1, 9e-2])
        R = np.diag([1.0**2, 0.4**2, 1.2**2, 2.0**2])
        
        self.ekf = EKFHeightDepth(f, r, b, m, k, j, res, l, gr, H0, Z0, a0, v0, i0, w0, theta0, P0, Q, R)
        self.motor_controller = PIController(kp=0.585, ki=1.95, dt=0.01, V_nom=12.0)
        
        # --- Subscribers ---
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Int32, '/front_right_motor/encoder', self.encoder_callback, 10)
        self.create_subscription(Float32, '/setpoint_position', self.setpoint_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # --- Publishers ---
        self.image_pub = self.create_publisher(Image, '/filtered_keypoints_image', 10)
        self.depth_pub = self.create_publisher(Float32, '/estimated_depth', 10)
        self.vel_targ_pub = self.create_publisher(Float32, '/motor/setpoint_vel', 10)
        self.vel_enc_pub = self.create_publisher(Float32, '/motor/rnc_vel', 10)
        self.fl_cmd_pub = self.create_publisher(Float64, '/front_left_motor/voltage', 10)
        self.fr_cmd_pub = self.create_publisher(Float64, '/front_right_motor/voltage', 10)
        self.inov_pub = self.create_publisher(Vector3, '/inovation', 10)
        
        print("twas never soft")
        # EKF State Publishers
        self.ekf_pubs = {
            'H': self.create_publisher(Float32, '/ekf/H', 10),
            'Z': self.create_publisher(Float32, '/ekf/Z', 10),
            'a': self.create_publisher(Float32, '/ekf/a', 10),
            'v': self.create_publisher(Float32, '/ekf/v', 10)
        }

        self.last_time = self.get_clock().now()
        self.start_time = self.get_clock().now()
        # --- Timer (100Hz) ---
        self.timer = self.create_timer(0.01, self.control_loop)

    def odom_callback(self, msg):
        self.latest_odom = msg

    def encoder_callback(self, msg):
        self.latest_enc = (msg.data, self.get_clock().now().nanoseconds / 1e9)

    def setpoint_callback(self, msg):
        self.position_target = msg.data

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                v_center = cv_image.shape[0] // 2
                dist_to_center = (y + h) - v_center + 1
                
                # Draw bounding box
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Publish image with bbox
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8'))
                
                self.latest_height_data = (h, self.get_clock().now().nanoseconds / 1e9, dist_to_center)
                self.height_used = False
            else:
                print("No contours found")


        except Exception as e:
            self.get_logger().error(f"Image error: {e}")

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = 0.01 # Fixed step as per original logic or calculate via: now - self.last_time
        
        # Path Generation
        t = now
        #self.position_target = 3.0 + 0.9*math.sin(0.7*t)# + 0.5*math.sin(0.2*t) + 0.3*math.sin(0.3*t)
        
        H_est, Z_est, a_est, v_est, i_est, w_est, theta_est = self.ekf.x
        velocity_target = 0.5 * (self.position_target - Z_est)#0.4*math.sin(0.7*t)

        #if (self.get_clock().now() - self.start_time) > rclpy.duration.Duration(seconds=10.0):
        #    velocity_target = 0.05*3.14/2
        #else:
        #    velocity_target = 0


        # Encoder Velocity Logic
        enc_val, enc_time = self.latest_enc
        old_val, old_time = self.old_enc
        
        if (enc_time - old_time) > 0:
            dz = (-0.05/10) * ((2 * math.pi) / 4096) * (enc_val - old_val)
            new_velocity = dz / (enc_time - old_time)
            
            # Smooth velocity
            #current_buffer_avg = sum(v[0] for v in self.wheel_velocity_buffer) / len(self.wheel_velocity_buffer)
            self.velocity = 0.3 * new_velocity + 0.7 * self.velocity
            self.wheel_velocity_buffer.append((self.velocity, enc_time))

        # PI Control
        self.vel_enc_pub.publish(Float32(data=self.velocity/0.05))
        self.vel_targ_pub.publish(Float32(data=velocity_target/0.05))

        voltage = self.motor_controller.compute_command(velocity_target/0.05, self.velocity/0.05)
        
        # EKF Predict
        self.ekf.predict(-voltage * 12, dt, now)

        # EKF Update
        if not self.height_used:
            h, _, d_center = self.latest_height_data
            y_innov = self.ekf.update(h, 0.28 * h / d_center, -self.velocity/0.05, (2*math.pi/4096)*enc_val)
            self.height_used = True
            self.inov_pub.publish(Vector3(x=float(y_innov[0]), y=float(y_innov[1]), z=float(np.linalg.norm(y_innov))))

        # Publish Results
        self.depth_pub.publish(Float32(data=-1.0 * self.ekf.x[1] - 0.9))
        self.fl_cmd_pub.publish(Float64(data=-voltage))
        self.fr_cmd_pub.publish(Float64(data=-voltage))
        
        self.ekf_pubs['H'].publish(Float32(data=float(self.ekf.x[0])))
        self.ekf_pubs['Z'].publish(Float32(data=float(self.ekf.x[1])))
        self.ekf_pubs['a'].publish(Float32(data=float(self.ekf.x[2])))
        self.ekf_pubs['v'].publish(Float32(data=float(self.ekf.x[3])))

        self.old_enc = self.latest_enc
        self.last_time = now

def main(args=None):
    rclpy.init(args=args)
    node = ImageAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()