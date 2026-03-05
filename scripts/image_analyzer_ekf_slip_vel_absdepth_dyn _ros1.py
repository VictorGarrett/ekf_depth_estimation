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
import tf2_ros
import tf_conversions
import math
from motor_controller import PIController
from collections import deque


# callbacks --------
def odom_callback(msg):
    global latest_odom
    try:
        latest_odom = msg
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing odometry: {e}")


def encoder_callback(msg):
    global latest_enc
    try:
        latest_enc = (int(msg.data), rospy.get_time())
        #rospy.loginfo(f"Odometry updated: Position=({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
    except Exception as e:
        rospy.logerr(f"Error processing encoder: {e}")

def setpoint_position_callback(msg):
    global position_target
    position_target = msg.data 

def image_callback(msg):
    global latest_height
    global height_used
    global height_slope
    global hl_slope
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #rospy.loginfo("Received an image!")

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # color brown in HSV
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])

        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            vertical_center = cv_image.shape[0] // 2
            distance_to_center = (y + h) - vertical_center+1

            #rospy.loginfo(f"Distance from bottom of rectangle to vertical center: {(0.28*h)/(distance_to_center)}")

            height_slope = (h - latest_height[0])/(rospy.get_time() - latest_height[1])
            hl_slope = (distance_to_center - latest_height[2])/(rospy.get_time() - latest_height[1])
            latest_height = (h, rospy.get_time(), distance_to_center)
            height_used = False


            #cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #image_pub.publish(bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))
    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")



class EKFHeightDepth:
    def __init__(self, f, r, b, m, k, j, res, l, gr,  H0, Z0, a0, v0,i0, w0,theta0, P0, Q, R):
        self.f = f
        self.r = r
        self.b = b
        self.m = m
        self.k = k
        self.j = j
        self.res = res
        self.l = l
        self.gr = gr

        self.x = np.array([
            H0,
            Z0,
            a0,
            v0,
            i0,
            w0,
            theta0
        ], dtype=float)

        #print(self.x)
        self.P = P0.copy()
        self.Q = Q.copy()
        self.R = R.copy()

        self.Q_multiplier = np.array([0.0014, 0.01, 0.03, 0.01, 0.01, 0.01, 0.01], dtype=float)
        self.multiplier_decay_rate = 0.998

    def dynamics(self, x, u):
        H, Z, a, v, i, w, theta = x

        J_eq = self.j + self.m * self.r**2 / (2*self.gr**2)
        #J_eq = 2*J_eq
        b_eq = self.b 
        
        #i = (u - self.k * w) / self.res
        
        dx = np.zeros_like(x)
        dx[0] = 0  # H_dot
        dx[1] = v  # Z_dot
        dx[2] = 0  # a_dot (slip factor constant)
        dx[3] = - (a * self.r / self.gr) * (self.k * i - b_eq * w) / J_eq  # v_dot
        dx[4] = 0  # i_dot
        #dx[5] = (self.gr**2 * self.k * i - b_eq * w) / J_eq  # w_dot
        dx[5] = (self.k * i - b_eq * w) / J_eq  # w_dot
        dx[6] = w/self.gr  # theta_dot
        
        return dx

    def rk4_step(self, x, u, dt):
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5*dt*k1, u)
        k3 = self.dynamics(x + 0.5*dt*k2, u)
        k4 = self.dynamics(x + dt*k3, u)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def predict(self, u, dt):
        H, Z, a, v, i, w, theta = self.x
        
        #self.x = self.rk4_step(self.x, u, dt)

        dx = self.dynamics(self.x, u)
        self.x[4] = (u - self.k * w) / self.res
        self.x = dx * dt + self.x
        print(f"i={i:.3f}, w={w:.3f}, dw={dx[5]:.3f}, v={v:.3f}, dv={dx[3]:.3f} dvdt={dx[3]/dt:.3f} t={rospy.get_time():.3f}")
        

        J_eq = self.j + self.m * self.r**2 / 2
        b_eq = self.b
        
        #w
        dwdot_di = (self.gr**2 * self.k) / J_eq
        dwdot_dw = -b_eq / J_eq
        
        #v
        w_dot_val = (self.gr**2 * self.k * i - b_eq * w) / J_eq
        dvdot_di = - (a * self.r / self.gr) * dwdot_di
        dvdot_dw = - (a * self.r / self.gr) * dwdot_dw
        dvdot_da = - (self.r / self.gr) * w_dot_val
        
        #i
        di_dw = -self.k / self.res
        
        # Jacobian matrix (F_jac)
        F_jac = np.array([
            # H    Z    a      v      i        w        theta
            [1.0, 0.0, 0.0,    0.0,   0.0,     0.0,     0.0],  # H
            [0.0, 1.0, 0.0,    dt,    0.0,     0.0,     0.0],  # Z = Z + v·dt
            [0.0, 0.0, 1.0,    0.0,   0.0,     0.0,     0.0],  # a (slip factor)
            [0.0, 0.0, dvdot_da*dt, 1.0, dvdot_di*dt, dvdot_dw*dt, 0.0],  # v
            [0.0, 0.0, 0.0,    0.0,   0.0,     di_dw,   0.0],  # i (algebraic update)
            [0.0, 0.0, 0.0,    0.0,   dwdot_di*dt, 1.0 + dwdot_dw*dt, 0.0],  # w
            [0.0, 0.0, 0.0,    0.0,   0.0,     self.gr*dt, 1.0]   # theta
        ])
        

        Q_scaled = self.Q * dt * (np.ones(7) + self.Q_multiplier)
        self.P = F_jac @ self.P @ F_jac.T + Q_scaled
        self.P = np.clip(self.P, -1e3, 1e3)  
        self.Q_multiplier *= self.multiplier_decay_rate

    def update(self, h_meas, H_meas, w_meas, theta_meas):


        y_meas = np.array([
            h_meas,
            H_meas,
            w_meas,
            theta_meas
        ])

        H, Z, a, v, i, w, theta = self.x

        #print(f"states before update: H={H:.2f}, Z={Z:.2f}, a={a:.2f}, v={v:.2f} i={i:.2f} w={w:.2f} theta={theta:.2f}")
        #print(f"Update args: h_meas={h_meas:.2f}, H_meas={H_meas:.2f}, w_meas={w_meas:.2f}, theta_meas={theta_meas:.2f}")

        y_pred = np.array([H*f/Z, 
                           H,
                           w/self.gr,
                           theta])

        H_jac = np.array([
            [f/Z, -H*f/(Z**2),  0.0,  0.0,  0.0,  0.0,  0.0],
            [1.0,         0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,         0.0,  0.0,  0.0,  0.0,  1/self.gr,  0.0],
            [0.0,         0.0,  0.0,  0.0,  0.0,  0.0,  1.0]
        ])
        #print(f"predicted y: {y_pred}")
        #print(f"measured y: {y_meas}")
        y = y_meas - y_pred
        #print(f"innovation={y}")
        inov_pub.publish(Vector3(y[0], y[1], np.linalg.norm(y)))

        #print(f"P={self.P}")
        S = H_jac @ self.P @ H_jac.T + self.R
        #print(H_jac)
        #print(self.P)
        #print(S)
        if np.linalg.cond(S) > 1e12:
            rospy.logwarn("Innovation covariance S is ill-conditioned.")
            return

        K = self.P @ H_jac.T @ np.linalg.inv(S)
        #print(f"K={K}")
        self.x = self.x + K @ y
        I = np.eye(7)
        self.P = (I - K @ H_jac) @ self.P

        H, Z, a, v, i, w, theta = self.x
        print(f"states after update: H={H:.2f}, Z={Z:.2f}, a={a:.2f}, v={v:.2f} i={i:.2f} w={w:.2f} theta={theta:.2f}")

    
    def current_state(self):
        H = self.x[0]
        Z = self.x[1]
        a = self.x[2]
        v = self.x[3]
        i = self.x[4]
        w = self.x[5]
        theta = self.x[6]

        return np.array([H, Z, a, v, i, w, theta]), self.P.copy()
    
    def check_observability(self, u, dt, tol=1e-8):

        H, Z, a, v, i, w, theta = self.x
        den = a * self.m * self.r**2 + 2 * self.j
        den2 = den ** 2

        N_v = 2 * self.k * self.r * a * i + 2 * self.b * v
        dN_da = 2 * self.k * self.r * i
        dD_da = self.m * self.r**2

        dv_da = dt * ((dN_da * den - N_v * dD_da) / den2)
        dv_dv = 1 + (2 * self.b * dt / den)
        dv_di = (2 * self.k * self.r * a * dt) / den

        dw_da = dt * (- (2 * self.k * i - 2 * self.b * w) * self.m * self.r**2 / den2)
        dw_di = (2 * self.k * dt) / den
        dw_dw = 1 - (2 * self.b * dt) / den

        F = np.array([
            [1.0,  0.0,   0.0,   0.0,   0.0,              0.0, 0.0],
            [0.0,  1.0,   0.0,    dt,   0.0,              0.0, 0.0],
            [0.0,  0.0,   1.0,   0.0,   0.0,              0.0, 0.0],
            [0.0,  0.0, dv_da, dv_dv, dv_di,              0.0, 0.0],
            [0.0,  0.0,   0.0,   0.0,   0.0, -self.k/self.res, 0.0],
            [0.0,  0.0, dw_da,   0.0, dw_di,            dw_dw, 0.0],
            [0.0,  0.0,   0.0,   0.0,   0.0,              1.0, 1.0]
        ])


        H = np.array([
            [f/Z, -H*f/(Z**2),  0.0,  0.0,  0.0,  0.0,  0.0],
            [1.0,         0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,         0.0,  0.0,  0.0,  0.0,  1.0,  0.0],
            [0.0,         0.0,  0.0,  0.0,  0.0,  0.0,  1.0]
        ])

        O = np.vstack([
            H,
            H @ F,
            H @ F @ F
        ])


        O_normalized = O / np.linalg.norm(O, ord='fro')

        U, S, Vt = np.linalg.svd(O_normalized)
        cond_number = S[0] / S[-1] if S[-1] > 0 else np.inf

        rank = np.sum(S > tol)

        print("\n[EKF] Observability Matrix:\n", O)
        print("[EKF] Normalized Observability Matrix (Frobenius):\n", O_normalized)
        print("[EKF] Singular values:", S)
        print("[EKF] Condition number:", cond_number)
        print("[EKF] Estimated Rank:", rank)

        return O, rank, cond_number, S



latest_height = (0, 0, 0)  # (height, timestamp, distance to center)
height_used = True
height_slope = 0.0
hl_slope = 0.0


old_position = Point(0, 0, 0)
old_enc = (0, 0)
old_enc_update = (0, 0)

old_enc_wheel = 0
old_height = 0
old_width = 0

latest_enc = (0, 0)
position_target = 4



if __name__ == "__main__":

    # subscribers
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/front_right_motor/encoder', Int32, encoder_callback)
    rospy.Subscriber('/setpoint_position', Float32, setpoint_position_callback)

    # publishers
    image_pub = rospy.Publisher('/filtered_keypoints_image', Image, queue_size=10)
    depth_pub = rospy.Publisher('/estimated_depth', Float32, queue_size=10)
    vel_targ_pub = rospy.Publisher('/motor/setpoint_vel', Float32, queue_size=10)
    vel_enc_pub = rospy.Publisher('/motor/rnc_vel', Float32, queue_size=10)
    fl_motor_cmd_pub = rospy.Publisher('/front_left_motor/command', Float32, queue_size=10)
    fr_motor_cmd_pub = rospy.Publisher('/front_right_motor/command', Float32, queue_size=10)
    inov_pub = rospy.Publisher('/inovation', Vector3, queue_size=10)
    interv_pub = rospy.Publisher('/interval', Float32, queue_size=10)
    hd_pub = rospy.Publisher('/height_dif', Float32, queue_size=10)
    zm_pub = rospy.Publisher('/zm', Float32, queue_size=10)




    # EKF state publishers
    ekf_H_pub = rospy.Publisher('/ekf/H', Float32, queue_size=10)
    ekf_Z_pub = rospy.Publisher('/ekf/Z', Float32, queue_size=10)
    ekf_a_pub = rospy.Publisher('/ekf/a', Float32, queue_size=10)
    ekf_v_pub = rospy.Publisher('/ekf/v', Float32, queue_size=10)





    rospy.init_node('image_analyzer', anonymous=True)
    bridge = CvBridge()
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    
    f = (1-0.00)*554.26  # focal length
    gr = 10 
    r = (1+0.05)*0.05
    b = (1+0.05)*0.0001 + (1+0.05)*0.01/gr**2
    m = (1+0.05)*12.2
    #m = m - 0.5*m
    k = (1+0.05)*0.02
    #k = k - 0.5*k
    j = (1+0.05)*0.0001 + (1+0.05)*0.00125/gr**2
    #j=2*j
    res = (1+0.05)*1
    l = 0.001

    # initial guesses
    H0 = 1
    Z0 = 3.1
    a0 = 0.95
    v0 = 0
    i0 = 0
    w0 = 0
    theta0 = 0

    

    


    P0 = np.diag([0.2**2, 1.2**2, 0.6**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2])  # initial uncertainty
    Q = np.diag([4e-6, 0.15**2, 9e-3, 0.55**2, 9e-2, 9e-1, 9e-2]) # process noise 
    R = np.diag([1**2, 0.4**2, 1.2**2, 2**2])  

    ekf = EKFHeightDepth(f, r, b, m, k, j, res, l, gr,  H0, Z0, a0, v0,i0, w0,theta0, P0, Q, R)

    depth_msg = Float32()
    depth_msg.data = -3

    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    rospy.loginfo("Image Analyzer Node Started")

    H_est, Z_est, a_est, v_est, i_est, w_est, theta_est = H0, Z0, a0, v0, i0, w0, theta0


    image_wheel_dz = 0
    last_image_update_time = rospy.get_time()

    wheel_velocity_buffer = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    last_mean_wheel_velocity = 0.0
    velocity = 0.0
    old_zm = 0.0
    old_enc_update = (0, 0)

    #kpk = 36
    #motor_controller = PIController(kp=0.45*33, ki=(0.45*36)/0.13, dt=1/30, V_nom=12.0) #joint damping=2
    #motor_controller = PIController(kp=0.45*6.75, ki=(0.45*6.75)/0.1818, dt=1/60, V_nom=12.0) #joint damping=0.05
    #motor_controller = PIController(kp=0.45*5.8, ki=(0.45*5.8)/0.125, dt=1/60, V_nom=12.0) #higher mass 60hz
    #motor_controller = PIController(kp=0.45*4.9, ki=(0.45*4.9)/0.1, dt=1/100, V_nom=12.0) #higher mass 100hz
    motor_controller = PIController(kp=0.45*1.3, ki=(0.45*1.3)/0.30, dt=1/100, V_nom=12.0) #higher mass 100hz
    

    i = 0
    rate = rospy.Rate(100)

    try:
        while not rospy.is_shutdown():

              
            
            rate.sleep()
            dt = rospy.get_time() - last_image_update_time
            last_image_update_time = rospy.get_time()
            dt = 1/100
            if dt == 0:
                continue
            #ekf.predict(0.1, dt)
            #fl_motor_cmd_pub.publish(Float32(0.1/12))
            #fr_motor_cmd_pub.publish(Float32(0.1/12))
            #continue

            position_target = 3.0 + 0.9*math.sin(0.7* rospy.get_time()) + 0.5*math.sin(0.2* rospy.get_time()) + 0.3*math.sin(0.3* rospy.get_time())
            #position_target = 5.2
            #position_target = 2 * math.sin(0.8 * rospy.get_time())

            #position_target = 2 * abs((rospy.get_time() * 0.1) % 2 - 1)- 1

            #position_target = 0 if rospy.get_time() < 10 else 0.3
            position_target = 3.1 if rospy.get_time() < 10 else position_target

            

            velocity_target = 0.5*(position_target - Z_est)
            #velocity_target = 0.3*position_target


            dz = -((2*3.141592*0.05)/(4096))*(latest_enc[0] - old_enc[0])
            #print(dz)

            if (latest_enc[1] - old_enc[1]) > 0:
                new_velocity = dz / (latest_enc[1] - old_enc[1])
                print(f"new_enc={latest_enc[0]:.3f}, old_enc={old_enc[0]:.3f}, dt={latest_enc[1] - old_enc[1]:.3f}")
                #print(new_velocity/0.05)


                velocity = 0.3*new_velocity + 0.7*(sum([vel[0] for vel in wheel_velocity_buffer]) / len(wheel_velocity_buffer))
                #velocity = new_velocity
                wheel_velocity_buffer.append((velocity, latest_enc[1]))
                wheel_velocity_buffer.pop(0)

                
                #ekf.update_velocity((sum([vel[0] for vel in wheel_velocity_buffer]) / len(wheel_velocity_buffer)))

                velocity_target_msg = Float32()
                velocity_target_msg.data = velocity_target
                vel_targ_pub.publish(velocity_target_msg)

                velocity_msg = Float32()
                velocity_msg.data = sum([vel[0] for vel in wheel_velocity_buffer]) / len(wheel_velocity_buffer)
                vel_enc_pub.publish(velocity_msg)

            voltage = motor_controller.compute_command(velocity_target/0.05, (sum([vel[0] for vel in wheel_velocity_buffer]) / len(wheel_velocity_buffer))/0.05)
            #voltage = 0.5*math.sin(0.7* rospy.get_time())
            #voltage = position_target
            #voltage = -0.2

            #ekf.check_observability(velocity_target, dt)
            
            #print("Voltage command:", voltage)
            ekf.predict(-voltage*12, dt)
            #print(last_image_update_time-rospy.get_time())
            

            if latest_height is not None and not height_used:

                current_height = latest_height[0]
                current_abs_height = 0.28*latest_height[0]/latest_height[2]

                ekf.update(latest_height[0], (1-0.05)*0.28*(latest_height[0])/(latest_height[2]), -(sum([vel[0] for vel in wheel_velocity_buffer]) / len(wheel_velocity_buffer))/0.05, ((2*3.141592)/(4096))*latest_enc[0])
                height_used = True


                (H_est, Z_est, a_est, v_est, i_est, w_est, theta_est), P_est = ekf.current_state()
                #rospy.loginfo(f"Current Covariances:\n{P_est}")
                #print(f"current state: H_est={H_est:.2f}, Z_est={Z_est:.2f}, a_est={a_est:.2f}, v_est={v_est:.2f}, i_est={i_est:.2f}, w_est={w_est:.2f}, theta_est={theta_est:.2f},")
            

            

            depth_msg.data = -1*Z_est - 0.5 - 0.4
            depth_pub.publish(depth_msg)

            old_enc = latest_enc
            
            #print(velocity_target, voltage)

            fl_motor_cmd_pub.publish(Float32(-voltage))
            fr_motor_cmd_pub.publish(Float32(-voltage))

            #(H_est, Z_est, a_est, v_est, i_est, w_est, theta_est), P_est = ekf.current_state()

            #fl_motor_cmd_pub.publish(Float32(-0.5*(position_target - Z_est)))
            #fr_motor_cmd_pub.publish(Float32(-0.5*(position_target - Z_est)))

            #fl_motor_cmd_pub.publish(Float32(-0.2))
            #fr_motor_cmd_pub.publish(Float32(-0.2))

            (H_est, Z_est, a_est, v_est, i_est, w_est, theta_est), P_est = ekf.current_state()

            ekf_H_pub.publish(Float32(w_est))
            ekf_Z_pub.publish(Float32(Z_est))
            ekf_a_pub.publish(Float32(i_est))
            ekf_v_pub.publish(Float32(v_est))


    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Image Analyzer Node")
    finally:
        cv2.destroyAllWindows()
