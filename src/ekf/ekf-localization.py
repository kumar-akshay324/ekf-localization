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
import math

# --------------------------------------- # 

PLOT_CONDITION = False
PLOT_CONDITION_ORIGINAL = False
PLOT_CONDITION_RESAMPLED = False
PLOT_TIMESTAMPS = False
PLOT_RESULTS = True

DEBUG = False
DEBUG_MATRICES = False
DEBUG_OPERATIONS = False

# Format for ROBOT DATA = [ sampled groundtruth, sampled measurement, sampled odometry] (all constituents are pandas Data frames)
# Format for REFERENCE DATA = [ barcodes' dictionary, landmark groundtruth] (all constituents are pandas Data frames)

global ROBOT_NAME
global REFERENCE_DATA, ROBOT_DATA, TOTAL_TIMESTEPS
global LANDMARKS
LANDMARKS = {}

SAMPLE_TIME = 0.02

# robot-dependent motion noise parameters (see equation 3)
ALPHA_VALUES = np.array([0.2, 0.03, 0.09, 0.08, 0.0, 0.0])

# robot-dependent sensor noise parameters (see equation 9)
SIGMA_RANGE = 2
SIGMA_BEARING = 3
SIGMA_ID = 1

# --------------------------------------- #

# Noise matrices
Q_t = np.array([[SIGMA_RANGE**2, 0, 0], [0, SIGMA_BEARING**2, 0], [0, 0, SIGMA_ID**2]])
print (Q_t)

measurement_probability = 0
numnber_robots = 1
robot_number = 1

# load and resample raw data from UTIAS data set
# The robot's position groundtruth, odometry, and measurements 
# are stored in Robots

def getFinalData(barcode_data_file, landmark_gt_data_file, robot_data_files):
    barcode_data, landmark_groundtruth_data, \
    robot_groundtruth_data, \
    robot_measurement_data, \
    robot_odometry_data = getDataFromFiles(barcode_data_file, landmark_gt_data_file, robot_data_files)

    robot_groundtruth_data["t"] = robot_groundtruth_data["t"] - robot_groundtruth_data.at[0, "t"]
    robot_measurement_data["t"] = robot_measurement_data["t"] - robot_measurement_data.at[0, "t"]
    robot_odometry_data["t"] = robot_odometry_data["t"] - robot_odometry_data.at[0, "t"]

    robot_groundtruth_sampled_data = sampleData("robot_groundtruth_data", robot_groundtruth_data, SAMPLE_TIME)
    robot_measurement_sampled_data = sampleData("robot_measurement_data", robot_measurement_data, SAMPLE_TIME)
    robot_odometry_sampled_data = sampleData("robot_odometry_data", robot_odometry_data, SAMPLE_TIME)

    if (PLOT_CONDITION):
        fig = plt.figure(constrained_layout=True)
        grids = fig.add_gridspec(4, 2)

        fig_ax0 = fig.add_subplot(grids[0:2,0])
        fig_ax0.set_title("Landmark Groundtruth")

        fig_ax1 = fig.add_subplot(grids[0:2,1])
        fig_ax1.set_title("Robot Groundtruth Poses")
        fig_ax1.set_ylabel("Robot Groundtruth X axis")
        fig_ax1.set_xlabel("Robot Groundtruth Y axis")

        fig_ax2_0 = fig.add_subplot(grids[2,0])
        fig_ax2_1 = fig.add_subplot(grids[3,0])
    
        fig_ax3_0 = fig.add_subplot(grids[2,1])
        fig_ax3_1 = fig.add_subplot(grids[3,1])

        plotLandmarksGroundtruth(fig_ax0, "landmark_groundtruth_data", landmark_groundtruth_data, barcode_data, 'r')

        if (PLOT_CONDITION_ORIGINAL):
            plotRobotPose(fig_ax1, "robot_groundtruth_data", robot_groundtruth_data, 'r')
            plotRobotMeasurement(fig_ax2_0, fig_ax2_1, "robot_measurement_data", robot_measurement_data, 'g')
            plotRobotOdometry(fig_ax3_0, fig_ax3_1, "robot_odometry_data", robot_odometry_data, 'g')

        if (PLOT_CONDITION_RESAMPLED):
            plotRobotPose(fig_ax1, "robot_groundtruth_data", robot_groundtruth_sampled_data, 'b')
            plotRobotMeasurement(fig_ax2_0, fig_ax2_1, "robot_measurement_data", robot_measurement_sampled_data, 'r')
            plotRobotOdometry(fig_ax3_0, fig_ax3_1, "robot_odometry_data", robot_odometry_sampled_data, 'r')

        plt.show()

    if (PLOT_TIMESTAMPS):
        plotCollectedDataTimestamps(robot_groundtruth_sampled_data["t"].tolist(), \
                                    robot_measurement_sampled_data["t"].tolist(), \
                                    robot_odometry_sampled_data["t"].tolist())

    total_timesteps = robot_groundtruth_sampled_data.shape[0]
    print ("Final total timesteps: %s" %(str(total_timesteps)))
    return [barcode_data, landmark_groundtruth_data], \
            [robot_groundtruth_sampled_data, \
            robot_measurement_sampled_data, \
            robot_odometry_sampled_data], total_timesteps

def main(barcode_data_file, landmark_gt_data_file, robot_data_files):
    global REFERENCE_DATA, ROBOT_DATA, TOTAL_TIMESTEPS
    REFERENCE_DATA, ROBOT_DATA, TOTAL_TIMESTEPS = getFinalData(barcode_data_file, landmark_gt_data_file, robot_data_files)
    new_dict = dict([[value,key] for key,value in REFERENCE_DATA[0].items()])
    REFERENCE_DATA[0] = new_dict
    # We need to create a series for the new pose estimates for the timesteps
    # print ("Robot Measurement Data: %s" %(str(ROBOT_DATA[1])))
    execute()

def getMeasurementStartIndex(start_timestamp):
    measurement_data = ROBOT_DATA[1]
    for index, row in measurement_data.iterrows():
        print ("Current row: %s " %(str(row)))
        if (row["t"]) > start_timestamp:
            return index

def execute(start_timesample=0):
    # start_timestamp is the desired time at which the localization process is started and
    # start_timesample is the corresponding sample index for the groundtruth as well as the robot odometry data
    # measurement_start_timesample is sample index for robot provided measurement corresponding to this time stamp
    start_timestamp = ROBOT_DATA[0].at[start_timesample, "t"]
    measurement_start_timesample = getMeasurementStartIndex(start_timestamp)
    previous_measurement_index = measurement_start_timesample

    pose_mean, pose_covariance = initializeValues(start_timesample)
    start_timesample = start_timesample + 1

    # 
    x_odom, y_odom, thetas_odom = [], [], []
    x_final, y_final, thetas_final = [], [], []
    errors = []

    # print ("Odometry Data: \n%s\n" %(str(ROBOT_DATA[2])))
    for current_timesample in range(start_timesample, int(TOTAL_TIMESTEPS/10)):
        print ("--------------------------------------")
        theta = pose_mean[2]
        current_timestamp = ROBOT_DATA[0].at[current_timesample, "t"]
        print ("Executing time sample: %d and time stamp: %f" %(current_timesample, current_timestamp))

        input_series = ROBOT_DATA[2].loc[current_timesample]
        inputs = np.array([input_series["velocity"], input_series["omega"]])
        print ("Odometry inputs: %s" %(inputs))

        delta_rotation = inputs[1] * SAMPLE_TIME
        half_delta_rotation = delta_rotation / 2
        delta_translation = inputs[0] * SAMPLE_TIME

        # Jacobian of this motion or prediction model
        G_matrix = np.array([[1, 0, delta_translation * math.sin(theta + half_delta_rotation)], \
                         [0, 1, delta_translation * math.sin(theta + half_delta_rotation)], \
                         [0, 0, 1]])
        # Motion covariance in control space
        M_matrix = np.array([ [ALPHA_VALUES[0] * math.fabs(inputs[0]) + ALPHA_VALUES[1] * math.fabs(inputs[1])**2, 0], \
                              [0, ALPHA_VALUES[2] * math.fabs(inputs[0]) + ALPHA_VALUES[3] * math.fabs(inputs[1])**2] ])

        # Jacobian to convert covariance from Action space to State space
        V_matrix = np.array([[math.cos(theta + half_delta_rotation), -0.5 * math.sin(theta + half_delta_rotation)], \
                              [math.sin(theta + half_delta_rotation),  0.5 * math.cos(theta + half_delta_rotation)], \
                              [0, 1]])


        # Update the pose as per the motion
        pose_update_delta = np.array([delta_translation * math.cos(theta + half_delta_rotation), \
                                      delta_translation * math.sin(theta + half_delta_rotation), \
                                      delta_rotation])
        pose_update_delta = pose_update_delta.transpose()

        pose_updated = pose_mean + pose_update_delta
 
        x_odom.append(pose_updated[0])
        y_odom.append(pose_updated[1])
        thetas_odom.append(pose_updated[2])
 
        # Estimated pose covariance
        pose_covariance_bar = pose_covariance

        if ((G_matrix.shape[1] == pose_covariance.shape[0]) and (V_matrix.shape[1] == M_matrix.shape[0])):
            pose_covariance_bar = np.matmul(np.matmul(V_matrix, M_matrix), V_matrix.transpose()) + \
                                  np.matmul(np.matmul(G_matrix, pose_covariance), G_matrix.transpose())
            # print ("Shape: ", pose_covariance_bar.shape)
        else:
            print ("Matrix shape mismatch, could not updated pose covariance")
            continue

        observations, previous_measurement_index = getObservations(current_timestamp, previous_measurement_index)

        if DEBUG:
            if DEBUG_MATRICES:
                print ("G Matrix: \n%s \n" %(G_matrix))
                print ("M Vector: \n%s \n" %(M_matrix))
                print ("V Matrix: \n%s \n" %(V_matrix))

            if DEBUG_OPERATIONS:
                print ("Initial Pose: %s\nOdom Updated Pose: %s, EKF Updated Pose: \n" %(str(pose_mean), str(pose_updated), str(pose_final)))
                print ("Measurement time sample: %d and time stamp: %f" %(previous_measurement_index, current_timestamp))
                print ("Observations: \n%s" %(observations))

        observations_hat = []
        S = {}
        for df_row in observations:
            observed_landmark_barcode = df_row["subject"]
            # Here the dictionary has already been inverted from Subject ID -> Barcode to Barcode -> Subject ID
            # and makes search easier
            if observed_landmark_barcode in REFERENCE_DATA[0].keys():
                global LANDMARKS
                landmark_id = REFERENCE_DATA[0][observed_landmark_barcode]
                LANDMARKS[landmark_id] = df_row

                # Actual position of the landmark just observed
                landmark_groundtruth_pose = REFERENCE_DATA[1][landmark_id]

                # Based on this actual pose, what is the expected measurement of the landmark
                expected_distance_x  = (landmark_groundtruth_pose[0] - pose_updated[0])
                expected_distance_y  = (landmark_groundtruth_pose[0] - pose_updated[1])
                expected_distance_sq =  expected_distance_x** 2 + expected_distance_y ** 2
                expected_distance    =  math.sqrt(expected_distance_sq)

                expected_bearing = constraintBearing(math.atan2(expected_distance_y, expected_distance_x) - pose_updated[2])
                observations_hat.append(pd.DataFrame({"subject": [landmark_id], "data_range": [expected_distance], \
                                                     "data_bearing": [expected_bearing]}))

                # Jacobian of measurement model
                H_matrix = np.array([[-1 * (expected_distance_x/expected_distance), -1 * (expected_distance_y/expected_distance), 0], \
                                     [(expected_distance_y/expected_distance), -1 * (expected_distance_x / expected_distance), -1], \
                                     [0, 0, 0]]) 

                # S for all the landmarks
                if H_matrix.shape[1] == pose_covariance_bar.shape[0]:
                    S[landmark_id] = np.matmul(np.matmul(H_matrix, pose_covariance_bar), H_matrix) + Q_t
                else:
                    print ("S calculation failed due to failed multiplication")
                # Calculate the Kalman Gain

                if (pose_covariance.shape[1] == H_matrix.shape[0]):
                    K_matrix = np.matmul(np.matmul(pose_covariance, H_matrix.transpose()), np.linalg.inv(S[landmark_id]))
                else:
                    print ("Kalman Gain Failed due to failed multiplication")

                # Update pose and covariances
                observations_delta_vector = np.array([(df_row["data_range"] - observations_hat[-1]["data_range"]), \
                                               (df_row["data_bearing"] - observations_hat[-1]["data_bearing"]), \
                                               0])

                if (K_matrix.shape[1] == observations_delta_vector.shape[0]):
                    print ("Shapes K: %s and Obs_delta: %s" %(str(K_matrix.shape), str(observations_delta_vector.shape)))
                    print ("K: %s and Obs_delta: %s" %(str(K_matrix), str(observations_delta_vector)))
                    pose_delta_list = list(np.matmul(K_matrix, observations_delta_vector))
                    if (len(pose_delta_list) == len(pose_updated)):
                        pose_updated = [sum(element_a, element_b) for element_a, element_b in zip(pose_updated, pose_delta_list)]
                        pose_covariance_bar = np.matmul((np.identity(3, dtype=float) - np.matmul(K_matrix, H_matrix)), pose_covariance_bar)
                else:
                    print ("Pose and pose covariance update failed due to failed multiplication")

        pose_mean = pose_updated
        pose_mean[2] = constraintBearing(pose_updated[2])
        pose_covariance = pose_covariance_bar

        # Updated mean pose at this state 
        x_final.append(pose_mean[0])
        y_final.append(pose_mean[1])
        thetas_final.append(pose_mean[2])

        current_groundtruth_row = ROBOT_DATA[0].loc[current_timesample]
        pose_groundtruth = [ current_groundtruth_row["x"], \
                             current_groundtruth_row["y"], \
                             current_groundtruth_row["theta"]]

        errors.append([ a - b for a,b in zip(pose_groundtruth, pose_mean)])

    robot_pose_odom = pd.DataFrame({"x": x_odom, "y": y_odom, "theta": thetas_odom})
    robot_pose_final = pd.DataFrame({"x": x_final, "y": y_final, "theta": thetas_final})

    if PLOT_RESULTS:

        fig = plt.figure(constrained_layout=True)
        grids = fig.add_gridspec(4, 1)

        fig_ax0 = fig.add_subplot(grids[0:3,0])
        fig_ax0.set_title("Different Robot poses")

        fig_ax1 = fig.add_subplot(grids[3,0])
        fig_ax1.set_title("Different Robot pose error components")

        line_1 = plotRobotPose(fig_ax0, "Robot Pose Results", ROBOT_DATA[0], 'g')
        line_2 = plotRobotPose(fig_ax0, "Robot Pose Results", robot_pose_odom, 'r-')
        line_3 = plotRobotPose(fig_ax0, "Robot Pose Results", robot_pose_final[0], 'b')
        plt.legend((line__1, line_2, line_3), ('Groundtruth Poses', 'Odometry Poses', 'EKF Poses'))

        line_4 = fig_ax1.plot(errors[0])
        line_5 = fig_ax1.plot(errors[1])
        line_6 = fig_ax1.plot(errors[2])
        plt.legend((line__4, line_5, line_6), ('Error X', 'Error Y', 'Error Theta'))

        plt.show()

def getObservations(current_timestamp, previous_measurement_index):
    measurement_data = ROBOT_DATA[1]
    # if the observation/measurement time is less than a certain delta_time, it will be considered applicable for the current measurement
    observations = []
    delta_time = SAMPLE_TIME / 5
    print ("Input current timestamp: %f and previous index: %d" %(current_timestamp, previous_measurement_index))
    for index, row in measurement_data.iterrows():
        if (index > previous_measurement_index):
            if (row["t"] > (current_timestamp - delta_time)) and (row["t"] <= current_timestamp):
                print ("Data Appended")
                observations.append(row)
            if (row["t"] < current_timestamp):
                break
    if len(observations) == 0:
        return observations, previous_measurement_index
    # observations.index = element_id
    return observations, index

def initializeValues(start_timesample):
    # Guess based initialization for the pose covariance
    pose_covariance = np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]])
    print ("Initialized pose covariance: \n%s \n" %(pose_covariance))
    # Start with the first groundtruth as the initial pose of the robot
    robot_pose = ROBOT_DATA[0].loc[start_timesample]
    pose_mean = np.array([robot_pose["x"], robot_pose["y"], robot_pose["theta"]])
    print ("Initialized pose mean: %s \n" %(pose_mean))
    return pose_mean.transpose(), pose_covariance

def constraintBearing(bearing):
    if bearing < -math.pi:
        return bearing + 2 * math.pi
    elif bearing > math.pi:
        return bearing - 2 *  math.pi

if __name__ == "__main__":
	print ("Number of arguments: %d " %(len(sys.argv)))
	print ("Arguments provided: %s " %(str(sys.argv)))
	folder_name = sys.argv[1]
    # ROBOT_NAME = sys.argv[2]
	print ("Folder name: %s " %(folder_name))
	barcode_data_file, landmark_gt_data_file, robot_data_files = getDataFiles(folder_name, "Robot1")
	main(barcode_data_file, landmark_gt_data_file, robot_data_files)