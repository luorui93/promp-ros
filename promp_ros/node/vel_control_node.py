#! /usr/bin/env python
from lib2to3.pytree import Base
import queue
import rospy
import rospkg
import numpy as np
import sys
from promp_ros.promp import ProMP
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from multiprocessing import Lock
# from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
import message_filters
import matplotlib.pyplot as plt
import copy
from scipy import signal
from scipy.interpolate import CubicSpline, splrep, splev
import os
import tf2_ros


class OTPEstimator(object):
    def __init__(self, data_addr):
        ## Pro-MP
        self.dt = 0.0333
        self.data_addr = data_addr

        ## OTP calculation
        self.phi = 0
        self.traj_lock = Lock()
        self.tracker_lock = Lock()
        self.start_timer = False
        self.start_record = False
        self.start_experiment = False

        ## ROS subscribers
        tracker_sub = rospy.Subscriber("/wrist_pose", PoseStamped, self.tracker_pose_cb)

        ## ROS publishers
        self.robot_vel_pub = rospy.Publisher("/my_gen3/in/joint_velocity", Base_JointSpeeds, queue_size=1)

        ## trajectories
        self.init_time = rospy.Time.now()
        self.start_record = False
        self.t = 0.0


        # real time tracker traj
        self.real_tracker_traj = []
        self.sample_time = []


        

    def query_trajectory(self, tracker_traj=None):
        """
        Query a full robot trajectory from ProMP given OTP
        """
        # use trajectory file
        if tracker_traj is None:
            return self.query_dynamic_mean_traj()
        else:
            human_traj = np.array(tracker_traj)

        # we need to pad the partial trajectory as the same standard format, fill unknown value with np.nan
        # here the robot trajectory is unknown
        nan_traj = np.empty((human_traj.shape[0], 7))
        nan_traj[:] = np.nan
        padded_traj = np.hstack((nan_traj, human_traj))
        # alpha, phase = self.promp.estimate_phase(padded_traj)
        # print(f"estimated alpha: {alpha}, pahse: {phase}")

        # return the updated trajectory mean and cov
        traj_stat, phase = self.promp.predict(padded_traj, phase_estimation=False)
        # self.promp.plot_trajectory(traj_stat, 1, title="joint 1", plot_error=True)

        format_traj = self.promp.reshape_trajectory(traj_stat[0])

        return format_traj[:, 0:7]

    def query_dynamic_mean_traj(self):
        """
        Query the current dynamic mean otp from ProMP
        """
        mean = self.promp.weights['mean']
        cov = self.promp.weights['cov']
        traj_mean, traj_cov = self.promp.compute_trajectory_stat(mean, cov)
        format_traj = self.promp.reshape_trajectory(traj_mean)

        # last_point = format_traj[-1,0:7]
        # otp = JointTrajectoryPoint()
        # otp.positions = last_point

        return format_traj[:, 0:7]
        
    
    
    def tracker_pose_cb(self, msg):
        if (self.start_timer):
            with self.tracker_lock:
                self.sock_pose = msg.pose
                pose = []
                pose.append(self.sock_pose.position.x)
                pose.append(self.sock_pose.position.y)
                pose.append(self.sock_pose.position.z)
                self.real_tracker_traj.append(pose)

                if (self.start_experiment):
                    self.predicted_robot_vel_traj = self.query_trajectory(self.real_tracker_traj)

    def publish_predicted_traj(self, index):
        base_joint_speeds = Base_JointSpeeds()
        
        for i, joint_vel in enumerate(self.predicted_robot_vel_traj[index]):
            joint_speed = JointSpeed()
            joint_speed.joint_identifier = i
            joint_speed.value = joint_vel
            base_joint_speeds.joint_speeds.append(joint_speed)
        self.robot_vel_pub.publish(base_joint_speeds)

    def train_promp(self, n_basis, n_demos, n_dof):
        self.promp = ProMP(self.dt, n_basis=n_basis, demo_addr=self.data_addr, n_demos=n_demos, n_dof=n_dof)



    ############################TEST FUNCTIONS############################3

    def test_vel(self):
        r = rospy.Rate(90)
        input("Enter to start experiment")
        self.start_timer = True
        start_time = rospy.Time.now()
        self.predicted_robot_vel_traj = self.query_trajectory()
        self.start_experiment = True

        i = 0
        while (i < len(self.predicted_robot_vel_traj)):
            try:
                self.publish_predicted_traj(i)
                i += 1
            except Exception as e:
                print(e)
                break
            r.sleep()

        base_joint_speeds = Base_JointSpeeds()
        
        for i in range(7):
            joint_speed = JointSpeed()
            joint_speed.joint_identifier = i
            joint_speed.value = 0
            base_joint_speeds.joint_speeds.append(joint_speed)
        self.robot_vel_pub.publish(base_joint_speeds)



if __name__ == "__main__":
    rospy.init_node("otp_estimator", log_level=rospy.INFO)

    r = rospkg.RosPack()
    path = r.get_path('promp_ros')+'/training/plug/mixed/hrc_traj_'

    path = '/home/riverlab/kinova_ws/src/kortex_playground/train_data/sync'
    opte = OTPEstimator(path)

    ##### train promp
    rospy.loginfo("Start training")
    opte.train_promp(n_basis=20, n_demos=10, n_dof=10)
    opte.promp.main()
    rospy.loginfo("Training completed")


    opte.test_vel()
    
    



