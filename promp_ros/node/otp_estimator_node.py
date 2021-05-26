#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
import sys
from promp_ros.promp import ProMP
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from multiprocessing import Lock
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import message_filters
import matplotlib.pyplot as plt
import copy
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline, splrep, splev


class OTPEstimator(object):
    def __init__(self, data_addr):
        ## Pro-MP
        self.dt = 0.0333
        self.data_addr = data_addr

        ## OTP calculation
        self.otp_d = Pose()
        self.otp_s = Pose()
        self.otp_i = Pose()
        self.phi = 0
        self.traj_lock = Lock()
        self.socket_lock = Lock()

        ## ROS subscribers
        kinova_sub_sync = message_filters.Subscriber("/my_gen3/joint_states", JointState)
        tag_sub_sync = message_filters.Subscriber("/tag_detections", AprilTagDetectionArray)
        time_sync = message_filters.ApproximateTimeSynchronizer([kinova_sub_sync, tag_sub_sync], 10, slop=0.2)
        tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.socket_pose_cb)

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
        socket_position = np.empty(3)
        socket_position[0] = self.sock_pose.position.x
        socket_position[1] = self.sock_pose.position.y
        socket_position[2] = self.sock_pose.position.z

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

    # def robot_joint_state_cb(self, msg):
    #     """
    #     Read robot joint trajectories from topic
    #     The valid publication rate for joint state is about 100hz
    #     """
    #     with self.lock:
    #     # joint position time in millisecond
    #         if (len(self.sample_time) == 0):
    #             self.init_time = msg.header.stamp
    #             t = 0.0
    #             self.sample_time.append(t)
    #         else:
    #             diff_time = msg.header.stamp - self.init_time
    #             t = diff_time.secs * 1000 + diff_time.nsecs * 1e-6
    #             if t - self.sample_time[-1] >= 1:
    #                 self.sample_time.append(t)
    
    def socket_pose_cb(self, msg):
        with self.socket_lock:
            if len(msg.detections) > 0:
                self.sock_pose = msg.detections[0].pose.pose.pose
            else:
                rospy.logwarn_throttle(10, "tag not detected")

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
        if (self.start_record and len(socket_msg.detections)>0):
            if (len(self.sample_time) == 0):
                self.init_time = robot_msg.header.stamp
            robot_time = robot_msg.header.stamp
            socket_time = socket_msg.header.stamp
            # add a small displacement to time
            t = self.diff_time(robot_time, self.init_time)
            self.sample_time.append(t)
            # self.sample_time.append(self.diff_time(socket_time, self.init_time))

            # add sampled robot joint to trajectory
            sample_point = JointTrajectoryPoint()
            sample_point.positions = robot_msg.position[0:7]
            sample_point.velocities = robot_msg.velocity[0:7]
            # sample_point.accelerations = robot_msg.accelerations[0:7]
            sec = self.dt * len(self.robot_joint_trajectory)
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
        robot_array = self.list_to_array(self.robot_joint_trajectory, type="robot")
        socket_array = self.list_to_array(self.socket_trajectory, type="socket")
        # left: robot array    right: socket array
        training_array = np.concatenate((robot_array, socket_array), axis=1)

        np.savetxt(self.data_addr+str(index)+'.csv', training_array, delimiter=",")
        rospy.loginfo("Training data saved to "+self.data_addr+'/hrc_traj_'+str(index)+'.csv')

    def publish_trajectory(self, traj_list):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        for p in traj_list:
            traj_msg.points.append(p)
        self.planned_traj_pub.publish(traj_msg)
    
    def publish_target_position(self, traj_list):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.points.append(traj_list[-1])
        self.planned_traj_pub.publish(traj_msg)

    def record_data(self, duration):
        input("Enter to start recording")
        self.start_record = True
        
        input("Enter again to stop recording")
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
    
    def interpolate_traj(self, traj, spline_order=3):
        interp_traj = []
        for point in traj:
            interp_traj.append(point.positions)
        interp_traj = np.array(interp_traj)

        # interpolate trajectory using a cubice spline
        t = self.dt * np.arange(interp_traj.shape[0])
        dense_t = np.linspace(0, t[-1], 1000)
        interp_pos = np.empty((len(dense_t), interp_traj.shape[1]))
        interp_vel = np.empty(interp_pos.shape)
        interp_acc = np.empty(interp_pos.shape)
        if spline_order == 3:
            for d in range(interp_traj.shape[1]):
                # b-spline interpolation
                bs = splrep(t, interp_traj[:,d], k=3)
                interp_pos[:,d] = splev(dense_t, bs, 0)
                interp_vel[:,d] = splev(dense_t, bs, 1)
                interp_acc[:,d] = splev(dense_t, bs, 2)
            # cs = CubicSpline(t, interp_traj, axis=0)
            # interp_pos = cs(dense_t, 0)
            # interp_vel = cs(dense_t, 1)
            # interp_acc = cs(dense_t, 2)

        self.plot_interp_traj(dense_t, interp_pos, interp_vel, interp_acc)

    
    def list_to_array(self, traj_list, type):
        """
        Convert robot joint or socket pose trajectory list to numpy array
        """
        if type == "socket":
            traj_array = np.empty((len(traj_list), 3))
            for ind, pose in enumerate(traj_list):
                socket_pose = np.zeros(3)
                socket_pose[0] = pose.position.x
                socket_pose[1] = pose.position.y
                socket_pose[2] = pose.position.z
                traj_array[ind] = socket_pose
        
        elif type == "robot":
            # save both positions and velocities
            robot_dof = len(traj_list[0].positions)
            traj_array = np.empty((len(traj_list), robot_dof*2))
            for ind, point in enumerate(traj_list):
                traj_array[ind] = np.concatenate((np.array(point.positions),np.array(point.velocities)))

        
        else:
            rospy.logerr("Unsupported trajectory type")
            return None

        return traj_array

    def train_promp(self, n_basis, n_demos, n_dof):
        self.promp = ProMP(self.dt, n_basis=n_basis, demo_addr=self.data_addr, n_demos=n_demos, n_dof=n_dof)

    ############################TEST FUNCTIONS############################3
    def test_conversion(self, traj=None):
        if traj is None:
            socket_array = self.list_to_array(self.socket_trajectory, type="socket")
            print(f"socket array ({socket_array.shape[0]}): \n{socket_array}")
            robot_array = self.list_to_array(self.robot_joint_trajectory, type="robot")
            print(f"robot array ({robot_array.shape[0]}):\n{robot_array}")
        else:
            robot_array = self.list_to_array(traj, type="robot")
            print(f"robot array ({robot_array.shape[0]}):\n{robot_array}")
            

    
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
        # filter the velocities
        # filtered_v = uniform_filter1d(y, axis=0, size=3)
        for joint in range(y.shape[1]):
           ax2.plot(t, y[:,joint], label=f"joint{joint}")
        ax2.set_title("Velocity")
        plt.legend(loc="upper left")

        ## acceleration
        ax3 = fig.add_subplot(313)
        # next_filtered_v = filtered_v[1:,:]
        # next_filtered_v = np.vstack((next_filtered_v, next_filtered_v[-1,:]))
        
        # diff = next_filtered_v - filtered_v
        y = []
        for point in joint_trajectory.points:
            y.append(point.accelerations)
        y = np.array(y)
        for joint in range(y.shape[1]):
           ax3.plot(t, y[:,joint], label=f"joint{joint}")
        ax3.set_title("Acceleration")
        plt.legend(loc="upper left")
        
        # self.plot_interp_traj(joint_trajectory)
    
    def plot_interp_traj(self, t, pos, vel, acc):
        fig = plt.figure(2, figsize=(10,20))
        ax1 = fig.add_subplot(311)
        ax1.plot(t, pos)
        ax1.set_title("Position")
        ax2 = fig.add_subplot(312)
        ax2.plot(t, vel)
        ax2.set_title("Velocity")
        ax3 = fig.add_subplot(313)
        ax3.plot(t, acc)
        ax3.set_title("Acceleration")
        
if __name__ == "__main__":
    rospy.init_node("otp_estimator", log_level=rospy.INFO)

    r = rospkg.RosPack()
    path = r.get_path('promp_ros')
    opte = OTPEstimator(path+'/training/plug/hrc_traj_')

    ##### record data in time secs
    # opte.record_data(duration=3)
    # opte.plot_traj(opte.robot_joint_trajectory)

    ##### print numpy array format data
    # opte.test_conversion()
    # opte.interpolate_traj(opte.robot_joint_trajectory, 3)

    # resampled_traj = opte.trajectory_resampler(opte.robot_joint_trajectory, 10)
    # opte.publish_trajectory(resampled_traj)
    # opte.publish_target_position(opte.robot_joint_trajectory)
    # opte.test_conversion(resampled_traj)

    # if (len(sys.argv) < 2):
    #     raise SyntaxError("Insufficient arguments")
    # opte.save_training_data(int(sys.argv[1]))
    # msg = rospy.wait_for_message("/smoothed_trajectory", JointTrajectory)
    # opte.plot_traj(msg, 1)
    # plt.show()

    ##### train promp
    rospy.loginfo("Start training")
    opte.train_promp(20, 20, 7)
    opte.promp.main()
    rospy.loginfo("Training completed")
    plot_joint = 0
    rospy.loginfo(f"plot joint {plot_joint}")
    opte.promp.plot(plot_joint=plot_joint, plot_error=True)
    # mean, _ = pmp.compute_trajectory_stat(pmp.weights['mean'], pmp.weights['cov'])
    # traj = pmp.reshape_trajectory(mean)
    # print(f"trained trajectory mean: \n{traj}")
    # print(f"trained cov: {pmp.weights['cov']}")ning completed")

    # rospy.spin()



