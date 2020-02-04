import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plotRobotPose(fig_ax, name, df, color):
    times  = df["t"].tolist()
    data_x = df["x"].tolist()
    data_y = df["y"].tolist()

    fig_ax.set_xlabel("Robot Groundtruth Y axis")
    fig_ax.set_ylabel("Robot Groundtruth X axis")
    fig_ax.plot(data_y, data_x, color)

def plotRobotMeasurement(fig_ax0, fig_ax1, name, df, color):
    times  = df["t"].tolist()

    fig_ax0.set_title("Robot Measurements - Data Range")
    fig_ax0.set_ylabel("Robot Measurement Range axis")
    data_range = df["data_range"].tolist()
    fig_ax0.plot(times, data_range, color)

    fig_ax1.set_title("Robot Measurements - Data Bearing")
    fig_ax1.set_ylabel("Robot Measurement Bearing axis")
    data_bearing = df["data_bearing"].tolist()
    fig_ax1.plot(times, data_bearing, color)

def plotRobotOdometry(fig_ax0, fig_ax1, name, df, color):
    times  = df["t"].tolist()

    fig_ax0.set_title("Robot Odometry - Velocity")
    fig_ax0.set_ylabel("Robot Odometry Velocity axis")
    velocity = df["velocity"].tolist()
    fig_ax0.plot(times, velocity, color)

    fig_ax1.set_title("Robot Odometry - Omega")
    fig_ax1.set_ylabel("Robot Odometry Omega axis")
    omega = df["omega"].tolist()
    fig_ax1.plot(times, omega, color)

def plotLandmarksGroundtruth(fig_ax, name, ld_gt_dict, barcodes_dict, col):
    fig_ax.set_title("Landmark Grountruth")
    for key in ld_gt_dict.keys():
        dict_list = ld_gt_dict[key]
        e = Ellipse(xy = [dict_list[0], dict_list[1]], width = 2 * dict_list[2], height = 2 * dict_list[3], angle = 0, color=col, lw=7, fill=True, label = key)
        fig_ax.annotate(str([key, barcodes_dict[key]]), (dict_list[0], dict_list[1]))
        fig_ax.grid()
        fig_ax.add_patch(e)
        fig_ax.autoscale()

def plotCollectedDataTimestamps(robot_groundtruth_timestamps, robot_measurement_timestamps, robot_odometry_timestamps):
    plt.title("Time stamps")
    plt.plot(robot_groundtruth_timestamps, range(len(robot_groundtruth_timestamps)), 'r*')
    plt.plot(robot_measurement_timestamps, range(len(robot_measurement_timestamps)), 'go')
    plt.plot(robot_odometry_timestamps, range(len(robot_odometry_timestamps)), 'b+')
    plt.show()