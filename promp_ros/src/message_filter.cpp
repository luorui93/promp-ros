#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/JointState.h>


// void callback(const )

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "message_filter_node");
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::JointState> sub(nh, "/my_gen3/joint_states", 1);
    // message_filters::TimeTimeSequencer<sensor_msgs::JointState> seq(sub, ros::Duration(0.1), 
    ROS_INFO("HELLO");
    return 0;
}