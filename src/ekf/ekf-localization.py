#!/usr/bin/python3
import sys
import os

sys.path.insert(1, os.getcwd() + "/" + "src/utils/")
from load_dataset import *
from plot_values import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np

# robot-dependent motion noise parameters (see equation 3)
alpha_values = np.array([0.2, 0.03, 0.09, 0.08, 0.0, 0.0])
SAMPLE_TIME = .02

# robot-dependent sensor noise parameters (see equation 9)
sigma_range = 2
sigma_bearing = 3
sigma_id = 1

# Noise matrices
Q_t = np.matrix([[sigma_range**2, 0, 0], [0, sigma_bearing**2, 0], [0, 0, sigma_id**2]])
print (Q_t)

measurement_probability = 0
numnber_robots = 1
robot_number = 1

# load and resample raw data from UTIAS data set
# The robot's position groundtruth, odometry, and measurements 
# are stored in Robots

def main(barcode_data_file, landmark_gt_data_file, robot_data_files):
    barcode_data, landmark_groundtruth_data, \
    robot_groundtruth_data, \
    robot_measurement_data, \
    robot_odometry_data = getDataFromFiles(barcode_data_file, landmark_gt_data_file, robot_data_files)

    robot_groundtruth_data["t"] = robot_groundtruth_data["t"] - robot_groundtruth_data.at[0, "t"]
    robot_measurement_data["t"] = robot_measurement_data["t"] - robot_measurement_data.at[0, "t"]
    robot_odometry_data["t"] = robot_odometry_data["t"] - robot_odometry_data.at[0, "t"]

    fig = plt.figure(constrained_layout=True)
    grids = fig.add_gridspec(4, 2)

    fig_ax0 = fig.add_subplot(grids[0:2,0])
    fig_ax0.set_title("Landmark Groundtruth")
    plotLandmarksGroundtruth(fig_ax0, "landmark_groundtruth_data", landmark_groundtruth_data, barcode_data, 'r')

    fig_ax1 = fig.add_subplot(grids[0:2,1])
    fig_ax1.set_title("Robot Grountruth Poses")
    plotRobotPose(fig_ax1, "robot_groundtruth_data", robot_groundtruth_data, 'r')
    robot_groundtruth_sampled_data = sampleData("robot_groundtruth_data", robot_groundtruth_data, SAMPLE_TIME)
    plotRobotPose(fig_ax1, "robot_groundtruth_data", robot_groundtruth_sampled_data, 'b')

    fig_ax2_0 = fig.add_subplot(grids[2,0])
    fig_ax2_1 = fig.add_subplot(grids[3,0])

    plotRobotMeasurement(fig_ax2_0, fig_ax2_1, "robot_measurement_data", robot_measurement_data, 'g')
    robot_measurement_sampled_data = sampleData("robot_measurement_data", robot_measurement_data, SAMPLE_TIME)
    plotRobotMeasurement(fig_ax2_0, fig_ax2_1, "robot_measurement_data", robot_measurement_sampled_data, 'r')

    fig_ax3_0 = fig.add_subplot(grids[2,1])
    fig_ax3_1 = fig.add_subplot(grids[3,1])

    plotRobotOdometry(fig_ax3_0, fig_ax3_1, "robot_odometry_data", robot_odometry_data, 'g')
    robot_odometry_sampled_data = sampleData("robot_odometry_data", robot_odometry_data, SAMPLE_TIME)
    plotRobotOdometry(fig_ax3_0, fig_ax3_1, "robot_odometry_data", robot_odometry_sampled_data, 'r')

    plt.show()

# [Robots, timesteps] = sampleMRCLAMdataSet(Robots, deltaT)

# add pose estimate matrix to Robots
# data will be added to this as the program runs
# Robots{robot_num}.Est = zeros(size(Robots{robot_num}.G,1), 4)

# initialize time, and pose estimate
# start index is set to 600 because earlier data was found to cause problems
start = 600
# t = Robots{robot_num}.G(start, 1); % set start time

# set starting pose mean to pose groundtruth at start time
# poseMean = [Robots{robot_num}.G(start,2);
#             Robots{robot_num}.G(start,3);
#             Robots{robot_num}.G(start,4)];

# poseCov = np.matrix([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01];


if __name__ == "__main__":
	print ("Number of arguments: %d " %(len(sys.argv)))
	print ("Arguments provided: %s " %(str(sys.argv)))
	folder_name = sys.argv[1]
	print ("Folder name: %s " %(folder_name))
	barcode_data_file, landmark_gt_data_file, robot_data_files = getDataFiles(folder_name, "Robot1")
	main(barcode_data_file, landmark_gt_data_file, robot_data_files)