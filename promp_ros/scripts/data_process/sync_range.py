#! /usr/bin/env python

"""
This script is only used for data without syncing joint angles representation range
"""

import numpy as np
import glob
import rospkg

r = rospkg.RosPack()
path = r.get_path('promp_ros')

file_list = glob.glob(path+"/training/plug/fix_time_front_back/*.csv")
print (file_list)

for file in file_list:
    data = np.loadtxt(open(file), delimiter=',')
    robot_data = data[:,0:7]
    robot_data[robot_data < 0] = robot_data[robot_data < 0] + 2*np.pi
    # save_file = file + 'd'
    np.savetxt(file, data, delimiter=',')
    



