#! /usr/bin/env python
import rospy
import numpy as np
from promp_ros.promp import ProMP
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from multiprocessing import Lock
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
import trajectory_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import message_filters

class OTPEstimator(object):
    def __init__(self):
        ## OTP calculation
        self.otp_d = Pose()
        self.otp_s = Pose()
        self.otp_i = Pose()
        self.phi = 0
        self.lock = Lock()

        ## ROS subscribers
        kinova_sub = message_filters.Subscriber("/my_gen3/joint_states", JointState)
        tag_sub = message_filters.Subscriber("/tag_detections", AprilTagDetectionArray)
        time_sync = message_filters.ApproximateTimeSynchronizer([kinova_sub, tag_sub], 10, slop=0.2)
        time_sync.registerCallback(self.data_filter_cb)

        ## ROS publishers
        self.planned_traj_pub = rospy.Publisher('/planned_trajectory', JointTrajectory, queue_size=1, latch=True)

        ## trajectories
        self.init_time = rospy.Time.now()
        self.start_record = False
        self.t = 0.0

        self.robot_joint_trajectory = JointTrajectory()
        self.socket_trajectory = []
        self.sample_time = []

    def compute_static_otp(self):
        """
        Compute static otp
        """
        pass

    def compute_integrate_otp(self):
        """
        Compute integrated otp with static otp and dynamic otp
        """
        pass

    def query_trajectory(self, otp):
        """
        Query a full robot trajectory from ProMP given OTP
        """
        pass

    def query_dynamic_otp(self):
        """
        Query the current dynamic otp from ProMP
        """
        pass

    def human_traj_callback(self):
        """
        Callback function to subscribe human trajectory
        """
        pass

    def publish_robot_traj(self):
        """
        Publish the robot trajectory for the next period
        """
        pass

    def robot_joint_state_cb(self, msg):
        """
        Read robot joint trajectories from topic
        The valid publication rate for joint state is about 100hz
        """
        with self.lock:
        # joint position time in millisecond
            if (len(self.sample_time) == 0):
                self.init_time = msg.header.stamp
                t = 0.0
                self.sample_time.append(t)
            else:
                diff_time = msg.header.stamp - self.init_time
                t = diff_time.secs * 1000 + diff_time.nsecs * 1e-6
                if t - self.sample_time[-1] >= 1:
                    self.sample_time.append(t)

    def diff_time(self, cur_time, init_time):
        """
        return time difference in milliseconds
        """
        diff = cur_time - init_time
        t = diff.secs * 1e3 + diff.nsecs * 1e-6
        return t
    
    def data_filter_cb(self, robot_msg, socket_msg):
        """
        Synchronize robot joint states trajectory and socket pose trajectory
        """
        if (self.start_record):
            if (len(self.sample_time) == 0):
                self.init_time = robot_msg.header.stamp
                self.robot_joint_trajectory.joint_names = robot_msg.name[0:7]
            robot_time = robot_msg.header.stamp
            socket_time = socket_msg.header.stamp
            t = self.diff_time(robot_time, self.init_time)
            self.sample_time.append(t)
            # self.sample_time.append(self.diff_time(socket_time, self.init_time))

            # add sampled joint to trajectory
            sample_point = JointTrajectoryPoint()
            sample_point.positions = robot_msg.position[0:7]
            sec = t / 1000
            sample_point.time_from_start = rospy.Duration(secs=sec)

            self.robot_joint_trajectory.points.append(sample_point)
    
    def publish_planned_trajectory(self):
        self.robot_joint_trajectory.header.stamp = rospy.Time.now()
        self.planned_traj_pub.publish(self.robot_joint_trajectory)
        
    def start_recording(self):
        self.start_record = True

if __name__ == "__main__":
    rospy.init_node("otp_estimator", log_level=rospy.DEBUG)
    
    opte = OTPEstimator()

    # rospy.Timer(rospy.Duration(0.001), lambda timer_event: print(opte.sample_time[-1]))
    # opte.data_filter(20)
    opte.start_recording()
    rospy.sleep(2)
    opte.publish_planned_trajectory()

    # rospy.spin()


