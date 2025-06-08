#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Quaternion
from tf.transformations import quaternion_inverse, quaternion_multiply
import numpy as np
import tf2_ros
import tf2_geometry_msgs   # registers converters for PoseStamped, PointStamped, etc.
import tf_conversions

class OdomComparator:
    def __init__(self):
        rospy.init_node('odom_comparator')

        self.model_name = rospy.get_param('~model_name', 'cubic_robot')

        self.odom_pose = None
        self.gt_pose = None

        # Initialize transform from world to odom: (translation, rotation)
        self.odom_offset = {
            "translation": np.zeros(3),
            "rotation": [0, 0, 0, 1]  # Identity quaternion
        }

        # Publisher for dynamic transform
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)  

        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/ground_truth/odometry', Odometry, self.ground_truth_callback)

        rospy.Timer(rospy.Duration(0.01), self.broadcast_transform)

        self.rate = rospy.Rate(10)  # Hz
        self.run()

    def odom_callback(self, msg):
        self.odom_pose = msg

    def ground_truth_callback(self, msg):
        self.gt_pose = msg

    def pose_to_np(self, pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z])

    def compute_odometry_difference(self):
        def _to_target_pose(odom):
            # Wrap the Odometry pose in a PoseStamped
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose = odom.pose.pose

            # Transform into target frame
            try:
                transformed = self.tf_buffer.transform(ps, self.gt_pose.header.frame_id, rospy.Duration(0.1))
            except Exception as e:
                rospy.logerr("Transform error: %sm returning zeroed pose", e)
                # Return a zeroed pose if transformation fails
                zeroed_pose = Pose()
                zeroed_pose.position.x = 0.0
                zeroed_pose.position.y = 0.0
                zeroed_pose.position.z = 0.0
                zeroed_pose.orientation.x = 0.0
                zeroed_pose.orientation.y = 0.0
                zeroed_pose.orientation.z = 0.0
                zeroed_pose.orientation.w = 1.0
                return zeroed_pose

            rospy.loginfo(f"Transformed pose: {transformed.pose}")
            return transformed.pose

        # Transform both odometries into the target frame
        trans_odom_pose = _to_target_pose(self.odom_pose)

        # Position difference
        dx = trans_odom_pose.position.x - self.gt_pose.pose.pose.position.x
        dy = trans_odom_pose.position.y - self.gt_pose.pose.pose.position.y
        dz = trans_odom_pose.position.z - self.gt_pose.pose.pose.position.z
        pos_diff = np.array([dx, dy, dz])

        # Orientation difference: q_diff = inv(q_a) * q_b
        q_gt = (
            self.gt_pose.pose.pose.orientation.x,
            self.gt_pose.pose.pose.orientation.y,
            self.gt_pose.pose.pose.orientation.z,
            self.gt_pose.pose.pose.orientation.w
        )
        q_odom = (
            trans_odom_pose.orientation.x,
            trans_odom_pose.orientation.y,
            trans_odom_pose.orientation.z,
            trans_odom_pose.orientation.w
        )
        q_a_inv = quaternion_inverse(q_gt)
        q_diff = quaternion_multiply(q_a_inv, q_odom)
        ori_diff = Quaternion(*q_diff)

        rospy.loginfo(f"Ground truth pose: {self.gt_pose.pose.pose}")
        rospy.loginfo(f"Position difference: {pos_diff}")
        rospy.loginfo(f"Orientation difference: {ori_diff}")

        return pos_diff, ori_diff

    def broadcast_transform(self, event):
        if self.odom_pose and self.gt_pose and self.odom_offset is not None:
            t = TransformStamped()
            t.header.stamp = self.odom_pose.header.stamp
            t.header.frame_id = "world"
            t.child_frame_id = "odom"

            # Use stored offset
            t.transform.translation.x = self.odom_offset["translation"][0]
            t.transform.translation.y = self.odom_offset["translation"][1]
            t.transform.translation.z = self.odom_offset["translation"][2]

            quat = self.odom_offset["rotation"]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            self.tf_broadcaster.sendTransform(t)

    def run(self):
        while not rospy.is_shutdown():
            if self.odom_pose and self.gt_pose:
                
                error_lin, error_ori = self.compute_odometry_difference()
                odom_np = self.pose_to_np(self.odom_pose.pose.pose)
                gt_np = self.pose_to_np(self.gt_pose.pose.pose)

                print(f"Odom: {odom_np}, GT: {gt_np}, Error: {np.linalg.norm(error_lin):.4f}")

                if np.linalg.norm(error_lin) > 2:
                    # Update the odom_offset using the linear and orientation errors
                    self.odom_offset["translation"] -= error_lin

                    # Update the rotation offset by multiplying the current offset with the inverse of the orientation error
                    error_ori_inv = quaternion_inverse([error_ori.x, error_ori.y, error_ori.z, error_ori.w])
                    self.odom_offset["rotation"] = quaternion_multiply(error_ori_inv, self.odom_offset["rotation"])
                    rospy.loginfo(f"Reseting odom transform with offset: {self.odom_offset}")
            else:
                print("Waiting for odometry or ground truth data...")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        OdomComparator()
    except rospy.ROSInterruptException:
        pass
