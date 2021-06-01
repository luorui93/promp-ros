#! /usr/bin/env python
from os import sync
import numpy as np
import glob
import rospkg
import matplotlib.pyplot as plt
import re

r = rospkg.RosPack()
path = r.get_path('promp_ros')


def sync_range():
    """
    Sync joint angels from -pi ~ pi to 0 ~ 2*pi
    """
    file_list = glob.glob(path+"/training/plug/mixed/90_samples/*.csv")
    print (file_list)

    for file in file_list:
        data = np.loadtxt(open(file), delimiter=',')
        robot_data = data[:,0:7]
        robot_data[robot_data < 0] = robot_data[robot_data < 0] + 2*np.pi
        # save_file = file + 'd'
        np.savetxt(file, data, delimiter=',')

    
def resample_data(duration = 3):
    """
    Resample training data to speed up the data
    """
    file_list = glob.glob(path+"/training/plug/mixed/*.csv")
    print(file_list)
    new_t = np.arange(int(duration*30))
    default_t = np.arange(150)

    for file in file_list:
        data = np.loadtxt(open(file), delimiter=',')
        resampled_data = np.empty((len(new_t), data.shape[1]))
        alpha = len(default_t)/len(new_t)
        for col in range(data.shape[1]):
            resampled_data[:, col] = np.interp(new_t*alpha, default_t, data[:,col])
            # plt.plot(new_t*alpha, resampled_data[:, col], 'bo', default_t, data[:,col], 'ro')
            # plt.pause(1)
            # plt.clf()
        id = re.findall('[0-9]+', file)
        if (id):
            new_file = path+f"/training/plug/mixed/{duration}s/hrc_traj_{id[0]}.csv"
            np.savetxt(new_file, resampled_data, delimiter=",")
            print(f"Saved to {new_file}")

# resample_data()
sync_range()




