#! /usr/bin/env python
import rospy
import rospkg
import numpy as np
import sys
from promp_ros.promp import ProMP
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from multiprocessing import Lock
from apriltag_ros.msg import AprilTagDetectionArray, AprilTagDetection
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
import message_filters
import matplotlib.pyplot as plt
import copy
from scipy import signal
from scipy.interpolate import CubicSpline, splrep, splev
import os
import tf2_ros

def alarm():
    duration = 0.5 # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

class OTPEstimator(object):
    def __init__(self, data_addr):
        ## Pro-MP
        self.dt = 0.0333
        self.data_addr = data_addr

        ## OTP calculation
        self.phi = 0
        self.traj_lock = Lock()
        self.socket_lock = Lock()
        self.start_timer = False
        self.start_record = False

        ## ROS subscribers
        kinova_sub_sync = message_filters.Subscriber("/my_gen3/joint_states", JointState)
        tag_sub_sync = message_filters.Subscriber("/tag_detections", AprilTagDetectionArray)
        tracker_sub_sync = message_filters.Subscriber("/wrist_pose", PoseStamped)
        time_sync = message_filters.ApproximateTimeSynchronizer([kinova_sub_sync, tracker_sub_sync], 10, slop=0.2)
        tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.socket_pose_cb)

        #debug subscriber
        time_sync.registerCallback(self.message_sync_cb)

        ## ROS publishers
        self.planned_traj_pub = rospy.Publisher('/planned_trajectory', JointTrajectory, queue_size=1, latch=True)
        self.robot_traj_pub = rospy.Publisher('/robot_trajectory', RobotTrajectory, queue_size=1, latch=True)

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

        # real time socket traj
        self.real_socket_traj = []
        self.sample_time = []

    def compute_static_otp(self):
        """
        Compute static otp
        """
        # socket_position = np.empty(3)
        # socket_position[0] = self.sock_pose.position.x
        # socket_position[1] = self.sock_pose.position.y
        # socket_position[2] = self.sock_pose.position.z
        otp_s = np.array([5.8413, 5.9208, 3.1209, 4.4577, 0.0883, 5.3299, 1.1647])
        return otp_s

    def compute_integrate_otp(self, phase, otp_d):
        """
        Compute integrated otp with static otp and dynamic otp
        """
        lamb = (phase - 1)**3 + 1
        otp_s = self.compute_static_otp()
        otp_i = (1-lamb)*otp_s + lamb*otp_d
        return otp_i
        

    def query_trajectory(self, otp):
        """
        Query a full robot trajectory from ProMP given OTP
        """
        pass

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

        return format_traj
    
    def query_dynamic_otp(self, socket_traj=None):
        """
        Query the dynamic OTP given partial socket pose trajectory
        """
        # use trajectory file
        if socket_traj is None:
            test_traj = np.loadtxt(open(self.data_addr+"32.csv"), delimiter=",")
            human_traj = test_traj[0:100, [14, 15, 16]]
        else:
            human_traj = socket_traj
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

        last_point = format_traj[-1,0:7]
        otp = JointTrajectoryPoint()
        otp.positions = last_point

        return otp
        
    def query_dynamic_otp_target(self, query_points, time):
        assert(query_points.shape[0] == len(time)), "query points length should be the same as time"
        nan_traj = np.empty((query_points.shape[0], 7))
        nan_traj[:] = np.nan
        padded_traj = np.hstack((nan_traj, query_points))

        traj_stat, phase = self.promp.predict(padded_traj, phase_estimation=False, given_time=time)

        format_traj = self.promp.reshape_trajectory(traj_stat[0])

        last_point = format_traj[-1,0:7]
        otp = JointTrajectoryPoint()
        otp.positions = last_point

        return otp
    
    
    def socket_pose_cb(self, msg):
        if (self.start_timer):
            with self.socket_lock:
                if len(msg.detections) > 0:
                    self.sock_pose = msg.detections[0].pose.pose.pose
                    pose = []
                    pose.append(self.sock_pose.position.x)
                    pose.append(self.sock_pose.position.y)
                    pose.append(self.sock_pose.position.z)
                    self.real_socket_traj.append(pose)
                else:
                    rospy.logwarn_throttle(10, "tag not detected")

    def diff_time(self, cur_time, init_time):
        """
        return time difference in milliseconds
        """
        diff = cur_time - init_time
        t = diff.secs * 1e3 + diff.nsecs * 1e-6
        return t
    
    def message_sync_cb(self, robot_msg, tracker_msg):
        """
        Synchronize robot joint states trajectory and socket pose trajectory
        """
        if (self.start_record):
            if (len(self.sample_time) == 0):
                self.init_time = robot_msg.header.stamp
            robot_time = robot_msg.header.stamp
            socket_time = tracker_msg.header.stamp
            # add a small displacement to time
            t = self.diff_time(robot_time, self.init_time)
            self.sample_time.append(t)
            # self.sample_time.append(self.diff_time(socket_time, self.init_time))

            # add sampled robot joint to trajectory
            sample_point = JointTrajectoryPoint()
            # sync representation range of joint angles 
            sample_point.positions = robot_msg.position[0:7]
            sample_point.velocities = robot_msg.velocity[0:7]
            # sample_point.accelerations = robot_msg.accelerations[0:7]
            sec = self.dt * len(self.robot_joint_trajectory)
            sample_point.time_from_start = rospy.Duration(secs=sec)

            # add socket pose in kinect frame to trajectory
            # geometry_msgs/Pose
            self.socket_trajectory.append(tracker_msg.pose)

            self.robot_joint_trajectory.append(sample_point)
    
    def save_training_data(self, index):
        """
        Save trajectory data for training
        """
        # convert robot joint trajectory into numpy array
        assert(len(self.robot_joint_trajectory) > 0), "robot trajectory empty"
        assert(len(self.socket_trajectory) > 0), "socket trajectory empty"
        assert(len(self.robot_joint_trajectory) == len(self.socket_trajectory)), "socket and robot trajectory must be at the same length"

        robot_array = self.list_to_array(self.robot_joint_trajectory, type="robot")
        socket_array = self.list_to_array(self.socket_trajectory, type="socket")
        # left: robot array    right: socket array
        training_array = np.concatenate((robot_array, socket_array), axis=1)

        np.savetxt(self.data_addr+str(index)+'.csv', training_array, delimiter=",")
        rospy.loginfo("Training data saved to "+self.data_addr+str(index)+'.csv')

    def publish_trajectory(self, traj_array):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        time = 0.0
        dt = 5.0 / traj_array.shape[0]
        # we need to convert back to the joint states value range
        for p in traj_array:
            time += dt
            s1 = p > np.pi
            s2 = p <= 2*np.pi
            selector = np.logical_and(s1, s2)
            p[selector] = p[selector] - 2*np.pi
            traj_msg.points.append(JointTrajectoryPoint(positions=list(p), time_from_start=rospy.Duration(time)))
        self.planned_traj_pub.publish(traj_msg)
    
    def publish_target_position(self, traj_list):
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.points.append(traj_list[-1])
        self.planned_traj_pub.publish(traj_msg)

    def record_data(self, duration=None):
        input("Enter to start recording")
        self.start_record = True
        
        if (duration is None):
            input("Enter again to stop recording")
        else:
            rospy.sleep(duration)
        self.start_record = False
        rospy.loginfo("Finished recording")
    
    def trajectory_resampler(self, traj, f=None, duration=None):
        """
        Sample trajectory at a given frequency or duration
        """
        ## the default frequency of sampling data is about 30hz
        ## resample the trajectory with different collection frequency (same duration)
        if f is not None:
            assert(f <= 30), "the resampled frequency should be less than 30hz"
            l = traj.shape[0]
            step = 30 // f
            selector = [i for i in range(0, l, step)]
            res_robot_joint_traj_array = traj[selector, :]
            return res_robot_joint_traj_array

        ## resample the trajectory with different duration (same frequency)
        elif duration is not None:
            pass
        else:
            rospy.logerr("Unsupported resampling method")
            raise ValueError()

    
    def interpolate_traj(self, traj_msg, spline_order=3):
        """
        Interpolate a trajectory_msgs/JointTrajectory trajectory
        """
        jt = traj_msg.points
        assert (len(jt) != 0), "trajectory message is empty"
        sparse_pos = []
        sparse_vel = []
        sparse_acc = []
        sparse_time = []
        for jtp in jt:
            sparse_pos.append(jtp.positions)
            sparse_vel.append(jtp.velocities)
            sparse_acc.append(jtp.accelerations)
            sparse_time.append(jtp.time_from_start.to_sec())
        
        dof = len(sparse_pos[0])
        dense_time = np.arange(0, sparse_time[-1], 0.001)
        interp_pos = np.empty((len(dense_time), dof))
        interp_vel = np.empty(interp_pos.shape)
        interp_acc = np.empty(interp_pos.shape)
        for d in range(dof):
            bs_p = splrep(np.array(sparse_time), np.array(sparse_pos)[:,d], k=1)
            interp_pos[:, d] = splev(dense_time, bs_p, 0)
            bs_v = splrep(np.array(sparse_time), np.array(sparse_vel)[:,d], k=2)
            interp_vel[:, d] = splev(dense_time, bs_v, 0)
            bs_a = splrep(np.array(sparse_time), np.array(sparse_acc)[:,d], k=3)
            interp_acc[:, d] = splev(dense_time, bs_a, 0)
            
        # interp_traj = []
        # for point in traj:
        #     interp_traj.append(point.positions)
        # interp_traj = np.array(interp_traj)

        # # interpolate trajectory using a cubice spline
        # t = self.dt * np.arange(interp_traj.shape[0])
        # dense_t = np.linspace(0, t[-1], 1000)
        # interp_pos = np.empty((len(dense_t), interp_traj.shape[1]))
        # interp_vel = np.empty(interp_pos.shape)
        # interp_acc = np.empty(interp_pos.shape)
        # for d in range(interp_traj.shape[1]):
        #     # b-spline interpolation
        #     bs = splrep(t, interp_traj[:,d], k=spline_order)
        #     interp_pos[:,d] = splev(dense_t, bs, 0)
        #     interp_vel[:,d] = splev(dense_t, bs, 1)
        #     interp_acc[:,d] = splev(dense_t, bs, 2)
        # cs = CubicSpline(t, interp_traj, axis=0)
        # interp_pos = cs(dense_t, 0)
        # interp_vel = cs(dense_t, 1)
        # interp_acc = cs(dense_t, 2)
        
        rospy.loginfo(f"Successfully interpolated {interp_pos.shape[0]} points")
        self.plot_interp_traj(dense_time, interp_pos, interp_vel, interp_acc)
        return self.convert_to_traj_msg(dense_time, interp_pos, interp_vel, interp_acc)

    def convert_to_traj_msg(self, dense_time, interp_pos, interp_vel, interp_acc):
        """
        Convert interpolated trajectory array to RobotTrajectory message
        """
        traj_msg = RobotTrajectory()
        assert(len(dense_time) == interp_pos.shape[0]), "Time and trajectory length not equal"
        steps = dense_time.shape[0]
        for i in range(steps):
            jtp = JointTrajectoryPoint(
                positions=list(interp_pos[i,:]),
                velocities=list(interp_vel[i,:]),
                accelerations=list(interp_acc[i,:]),
                time_from_start=rospy.Duration(dense_time[i])
            )
            traj_msg.joint_trajectory.points.append(jtp)
            traj_msg.joint_trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        
        return traj_msg

    
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

    def test_target(self):
        r = rospy.Rate(30)
        input("Enter to start experiment")
        self.start_timer = True
        start_time = rospy.Time.now()
        input("Enter to send target")
        while (not rospy.is_shutdown()):
            samples = 75
            phase = samples/150
            if len(self.real_socket_traj) > 0:
                given_time = [samples-1]
                obs_socket_traj = np.array([self.real_socket_traj[-1]])
                updated_otp = self.query_dynamic_otp_target(obs_socket_traj, given_time)
                otp_d = np.array(updated_otp.positions)
                otp_i = self.compute_integrate_otp(phase, otp_d)
                updated_otp.positions = list(otp_i)
                print("target position:\n", updated_otp.positions)
                print(f"Phase: {phase}")
                self.publish_target_position([updated_otp])
                alarm()
                break
            r.sleep()

        input("Enter to save time")
        end_time = rospy.Time.now()
        exp_time = (end_time - start_time).to_sec()

        input("Enter to calculate pose error")
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        received_tf = False
        while not received_tf:
            try:
                trans = tfBuffer.lookup_transform("socket_updated_pre", 'cable_tip', rospy.Time())
                received_tf = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                r.sleep()
                continue

        pos = (trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z)

        return exp_time, pos

    def test_otp(self):
        r = rospy.Rate(30)
        input("Enter to start experiment")
        self.start_timer = True
        start_time = rospy.Time.now()
        while (not rospy.is_shutdown()):
            try:
                # print(self.real_socket_traj[-1][1])
                samples = 150
                phase = samples / 150
                if len(self.real_socket_traj) == samples:
                    obs_socket_traj = np.array(self.real_socket_traj)
                    updated_otp = self.query_dynamic_otp(obs_socket_traj)
                    otp_d = np.array(updated_otp.positions)
                    otp_i = self.compute_integrate_otp(phase, otp_d)
                    updated_otp.positions = list(otp_i)
                    print("target position:\n", updated_otp.positions)
                    print(f"Phase: {phase}")
                    self.publish_target_position([updated_otp])
                    alarm()
                    break

            except rospy.ROSInterruptException:
                break
            r.sleep()
        
        input("Enter to save time")
        end_time = rospy.Time.now()
        exp_time = (end_time - start_time).to_sec()

        input("Enter to calculate pose error")
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        received_tf = False
        r = rospy.Rate(30)
        while not received_tf:
            try:
                trans = tfBuffer.lookup_transform("socket_updated_pre", 'cable_tip', rospy.Time())
                received_tf = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                r.sleep()
                continue

        pos = (trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z)

        return exp_time, pos

    
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
    path = r.get_path('promp_ros')+'/training/plug/mixed/hrc_traj_'

    path = '/home/riverlab/kinova_ws/src/kortex_playground/sync_data/sync'
    opte = OTPEstimator(path)

    ##### record training data
    # opte.record_data(3)
    # if (len(sys.argv) < 2):
    #     raise SyntaxError("Insufficient arguments")
    # opte.save_training_data(int(sys.argv[1]))
    # alarm()

    ##### train promp
    rospy.loginfo("Start training")
    opte.train_promp(n_basis=20, n_demos=10, n_dof=10)
    opte.promp.main()
    rospy.loginfo("Training completed")

    ##### print numpy array format data
    # opte.test_conversion()
    # opte.interpolate_traj(opte.robot_joint_trajectory, 4)

    # resampled_traj = opte.trajectory_resampler(opte.robot_joint_trajectory, 10)
    # opte.publish_trajectory(resampled_traj)
    # opte.test_conversion(resampled_traj)

    ##### publish mean trajectory
    # mean_traj = opte.query_dynamic_mean_traj()
    # resampled_traj = opte.trajectory_resampler(mean_traj, 2)
    # print(resampled_traj)
    # opte.publish_trajectory(resampled_traj)
    # msg = rospy.wait_for_message("/smoothed_trajectory", JointTrajectory)
    # robot_msg = opte.interpolate_traj(msg)
    # opte.robot_traj_pub.publish(robot_msg)
    # opte.plot_traj(msg, 1)

# 
    ##### plot mean trajectory
    # plot_joint = 4
    # rospy.loginfo(f"plot joint {plot_joint}")
    # opte.promp.plot_mean_trajectory(plot_joint=plot_joint, plot_error=True)
    # target_point = opte.query_dynamic_mean_otp()
    # print("target position:\n", target_point.positions)
    # opte.publish_target_position([target_point])
    # opte.publish_trajectory(opte.robot_joint_trajectory)

    ##### predict new trajectory given partial observation
    # updated_otp = opte.query_dynamic_otp()
    # print("target position:\n", updated_otp.positions)
    # opte.publish_target_position([updated_otp])

    ##### predict new trajectory with real human
    exp_time, pos_diff = opte.test_otp()
    ## directly send end target
    # exp_time, pos_diff = opte.test_target()
    pos_diff = np.array(pos_diff)
    pos_dis = 100*np.linalg.norm(pos_diff)

    print(f"Experiment time: {exp_time}")
    print(f"Position distance: {pos_dis}")

    # np.savetxt(path+'/data/straight.csv', np.array(opte.real_socket_traj), delimiter=",")
    # np.savetxt(path+'/data/curved.csv', np.array(opte.real_socket_traj), delimiter=",")
    # np.savetxt(path+'/data/accelerated.csv', np.array(opte.real_socket_traj), delimiter=",")
    print("file saved")
   
    # plt.show()
    # rospy.spin()



