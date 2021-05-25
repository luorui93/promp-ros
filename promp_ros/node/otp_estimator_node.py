#! /usr/bin/env python
import rospy
import numpy as np
from promp_ros.promp import ProMP
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from multiprocessing import Lock
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import message_filters
import matplotlib.pyplot as plt
import copy

class OTPEstimator(object):
    def __init__(self):
        ## Pro-MP
        self.promp = ProMP(0.1, 20, "", 40)

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
        #debug subscriber
        time_sync.registerCallback(self.message_sync_cb)

        ## ROS publishers
        self.planned_traj_pub = rospy.Publisher('/planned_trajectory', JointTrajectory, queue_size=1, latch=True)

        ## trajectories
        self.init_time = rospy.Time.now()
        self.start_record = False
        self.t = 0.0

        # each element of robot_joint_trajectory is a trajectory_msgs/JointTrajectoryPoint
        self.robot_joint_trajectory = []
        self.res_robot_joint_traj = []
        # each element of socket trajectory is a geometry_msgs/Pose 
        self.socket_trajectory = []
        self.res_socket_traj = []
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
    
    def message_sync_cb(self, robot_msg, socket_msg):
        """
        Synchronize robot joint states trajectory and socket pose trajectory
        """
        if (self.start_record):
            if (len(self.sample_time) == 0):
                self.init_time = robot_msg.header.stamp
            robot_time = robot_msg.header.stamp
            socket_time = socket_msg.header.stamp
            # add a small displacement to time
            t = self.diff_time(robot_time, self.init_time) + 30
            self.sample_time.append(t)
            # self.sample_time.append(self.diff_time(socket_time, self.init_time))

            # add sampled robot joint to trajectory
            sample_point = JointTrajectoryPoint()
            sample_point.positions = robot_msg.position[0:7]
            sec = t / 1000
            sample_point.time_from_start = rospy.Duration(secs=sec)

            # add socket pose in kinect frame to trajectory
            # geometry_msgs/Pose
            self.socket_trajectory.append(socket_msg.detections[0].pose.pose.pose)

            self.robot_joint_trajectory.append(sample_point)
    
    def save_training_data(self, index):
        """
        Save trajectory data for training
        """
        # convert robot joint trajectory into numpy array
        joint_array = np.array()
        np.savetxt('../training/plug/hrc_traj_'+index+'.csv', joint_array, delimiter=",")

    def publish_trajectory(self, traj_list):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        for p in traj_list:
            traj_msg.points.append(p)
        self.planned_traj_pub.publish(traj_msg)

    def record_data(self, time):
        input("Enter to start recording")
        self.start_record = True
        rospy.sleep(time)
        self.start_record = False
        rospy.loginfo("Finished recording")
    
    def trajectory_resampler(self, traj, f):
        """
        Sample trajectory at a given frequency
        """
        ## the default frequency of sampling data is about 30hz
        assert(f <= 30), "the resampled frequency should be less than 30hz"
        l = len(traj)
        step = 30 // f
        res_robot_joint_traj = copy.deepcopy(traj[0:l:step])
        return res_robot_joint_traj
    
    def list_to_array(self):
        # socket_pose = np.zeros((1,3))
        # socket_pose[0] = socket_msg.detections.pose.pose.pose.position.x
        # socket_pose[1] = socket_msg.detections.pose.pose.pose.position.y
        # socket_pose[2] = socket_msg.detections.pose.pose.pose.position.z
        pass

    ############################DEBUG FUNCTIONS############################3
    
    def plot_traj(self, joint_trajectory, index=1):
        fig = plt.figure(index, figsize=(10,20))

        ax1 = fig.add_subplot(311)
        t = np.arange(len(joint_trajectory.points))
        y = []
        for point in joint_trajectory.points:
            y.append(point.positions)
        y = np.array(y)
        for joint in range(y.shape[1]):
           ax1.plot(t, y[:,joint], label=f"joint{joint}")
        ax1.set_title("Position")
        plt.legend(loc="upper left")

        ax2 = fig.add_subplot(312)
        y = []
        for point in joint_trajectory.points:
            y.append(point.velocities)
        y = np.array(y)
        for joint in range(y.shape[1]):
           ax2.plot(t, y[:,joint], label=f"joint{joint}")
        ax2.set_title("Velocity")
        plt.legend(loc="upper left")

        ax3 = fig.add_subplot(313)
        y = []
        for point in joint_trajectory.points:
            y.append(point.accelerations)
        y = np.array(y)
        for joint in range(y.shape[1]):
           ax3.plot(t, y[:,joint], label=f"joint{joint}")
        ax3.set_title("Acceleration")
        plt.legend(loc="upper left")
        
if __name__ == "__main__":
    rospy.init_node("otp_estimator", log_level=rospy.INFO)
    
    opte = OTPEstimator()

    # rospy.Timer(rospy.Duration(0.001), lambda timer_event: print(opte.sample_time[-1]))
    # opte.data_filter(20)
    opte.record_data(2)
    # opte.plot_traj(opte.robot_joint_trajectory)

    resampled_traj = opte.trajectory_resampler(opte.robot_joint_trajectory, 10)
    opte.publish_trajectory(resampled_traj)

    # msg = rospy.wait_for_message("/smoothed_trajectory", JointTrajectory)
    # opte.plot_traj(msg, 1)
    # plt.show()

    rospy.spin()



