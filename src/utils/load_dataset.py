#!/usr/bin/python3

import sys
import os

import math
import pandas as pd

global DATASET_FOLDER_LOCATION

def getDataFiles(folder_name, robot_name):
	global DATASET_FOLDER_LOCATION
	DATASET_FOLDER_LOCATION = os.getcwd() +  "/" + folder_name
	all_file_names = []
	robot_data_files = {"name": robot_name, "odometry_filename": "", "measurement_filename": "", "groundtruth_filename": ""}
	barcode_data_file = ""
	landmark_gt_data_file = ""

	for file_name in os.listdir(DATASET_FOLDER_LOCATION):
		print("File Name: %s" %(file_name))
		all_file_names.append(DATASET_FOLDER_LOCATION + "/" + file_name)

		split_file_name = file_name.split(".")

		# Check for Barcode information
		if split_file_name[0] == "Barcodes":
			barcode_data_file = file_name
			print("Barcode data file name: %s" %(barcode_data_file))
			continue
		
		# Check for landmarks' groundtruth
		if split_file_name[0] == "Landmark_Groundtruth":
			landmark_gt_data_file = file_name
			print("Landmark Groudtruths data file name: %s" %(landmark_gt_data_file))
			continue

		split_file_name = split_file_name[0].split("_")

		if (split_file_name[0] == robot_name):
			if (split_file_name[1] == "Odometry"):
				robot_data_files["odometry_filename"] = file_name
			if (split_file_name[1] == "Measurement"):
				robot_data_files["measurement_filename"] =  file_name
			if (split_file_name[1] == "Groundtruth"):
				robot_data_files["groundtruth_filename"] =  file_name
		if robot_data_files["odometry_filename"] is not "" and robot_data_files["measurement_filename"] is not "" \
			and robot_data_files["groundtruth_filename"] is not "":
			print ("Robot File Info: %s" %(robot_data_files))
			break
	return barcode_data_file, landmark_gt_data_file, robot_data_files

def getDataFromFiles(barcode_data_file, landmark_gt_data_file, robot_data_files):
	global DATASET_FOLDER_LOCATION
	barcode_data_file = DATASET_FOLDER_LOCATION + "/" + barcode_data_file
	barcode_data = getBarCodeData(barcode_data_file)

	landmark_gt_data_file = DATASET_FOLDER_LOCATION + "/" + landmark_gt_data_file
	landmark_groundtruth_data = getLandmarkGrouthtruthData(landmark_gt_data_file)

	robot_gt_data_file = DATASET_FOLDER_LOCATION + "/" + robot_data_files["groundtruth_filename"]
	robot_groundtruth_data = getRobotGrouthtruthData(robot_gt_data_file)

	robot_measurement_data_file = DATASET_FOLDER_LOCATION + "/" + robot_data_files["measurement_filename"]
	robot_measurement_data = getRobotMeasurementData(robot_measurement_data_file)

	robot_measurement_odometry_file = DATASET_FOLDER_LOCATION + "/" + robot_data_files["odometry_filename"]
	robot_odometry_data = getRobotOdometryData(robot_measurement_odometry_file)

	return barcode_data, landmark_groundtruth_data, robot_groundtruth_data, robot_measurement_data, robot_odometry_data

def getBarCodeData(barcode_file):
	# Dictionary with entity number : Barcode where entity number 1 to 5 are the 5 robots and 6 to 20 are the 15 landmarks
	entity_barcodes = {}
	with open (barcode_file, "r") as file:
		for line in file:
			line.strip()
			if line[0] == "#":
				continue
			line_content = line.split()
			updated_line_content = []
			for item in line_content:
				try:
					updated_line_content.append(int(item))
				except:
					pass
			entity_barcodes[updated_line_content[0]] = updated_line_content[1]
	# print ( "Entity Barcode Content:\n %s \n" %(str(entity_barcodes)))
	return entity_barcodes

def getLandmarkGrouthtruthData(landmark_gt_data_file):
	# Dictionary with landmarks and their groundtruth positions
	# Groundtruth positions have [x position, y position , x posiiton std dev, y position std dev ]
	landmark_groundtruths = {}
	with open (landmark_gt_data_file, "r") as file:
		for line in file:
			line.strip()
			if line[0] == "#":
				continue
			line_content = line.split()
			updated_line_content = []

			for item in line_content:
				try:
					temp1 = int(item)
					temp2 = float(item)
					if (temp1/temp2 == 1.0):
						updated_line_content.append(int(item))
						continue
				except:
					pass

				try:
					updated_line_content.append(float(item))
				except:
					pass
			landmark_groundtruths[updated_line_content[0]] = [item for item in updated_line_content[1:]]
	print ( "Landmark Line Content:\n %s \n" %(str(landmark_groundtruths)))
	return landmark_groundtruths

def getRobotGrouthtruthData(robot_gt_data_file):
	# Pandas dataframe with robots and their groundtruth positions
	# Groundtruth positions have [x position, y position , orientation]
	t, x, y, theta = [], [], [], []
	with open (robot_gt_data_file, "r") as file:
		for line in file:
			line.strip()
			if line[0] == "#":
				continue
			line_content = line.split()
			updated_line_content = []

			for item in line_content:
				try:
					updated_line_content.append(float(item))
				except:
					pass
			t.append(updated_line_content[0])
			x.append(updated_line_content[1])
			y.append(updated_line_content[2])
			theta.append(updated_line_content[3])
	
	robot_groundtruth_data = pd.DataFrame({"t": t, "x": x, "y": y, "theta": theta})
	# print ( "Robot Groundtruth Line Content:\n %s \n" %(str(robot_groundtruth_data)))
	return robot_groundtruth_data

def getRobotMeasurementData(robot_measurement_data_file):
	# Pandas dataframe with robots and their groundtruth positions
	# Groundtruth positions have [x position, y position , orientation]
	t, subject, data_range, data_bearing = [], [], [], []
	with open (robot_measurement_data_file, "r") as file:
		for line in file:
			line.strip()
			if line[0] == "#":
				continue
			line_content = line.split()
			updated_line_content = []

			for item in line_content:
				try:
					temp1 = int(item)
					temp2 = float(item)
					if (temp1/temp2 == 1.0):
						updated_line_content.append(int(item))
						continue
				except:
					pass

				try:
					updated_line_content.append(float(item))
				except:
					pass
			t.append(updated_line_content[0])
			subject.append(updated_line_content[1])
			data_range.append(updated_line_content[2])
			data_bearing.append(updated_line_content[3])
	
	robot_measurement_data = pd.DataFrame({"t": t, "subject": subject, "data_range": data_range, "data_bearing": data_bearing})
	# print ( "Robot Measurement Line Content:\n %s \n" %(str(robot_measurement_data)))
	return robot_measurement_data

def getRobotOdometryData(robot_odometry_data_file):
	# Pandas dataframe with robots and their groundtruth positions
	# Groundtruth positions have [x position, y position , orientation]
	t, velocity, omega = [], [], []
	with open (robot_odometry_data_file, "r") as file:
		for line in file:
			line.strip()
			if line[0] == "#":
				continue
			line_content = line.split()
			updated_line_content = []

			for item in line_content:
				try:
					updated_line_content.append(float(item))
				except:
					pass
			t.append(updated_line_content[0])
			velocity.append(updated_line_content[1])
			omega.append(updated_line_content[2])
	
	robot_odometry_data = pd.DataFrame({"t": t, "velocity": velocity, "omega": omega})
	# print ( "Robot Odometry Line Content:\n %s \n" %(str(robot_odometry_data)))
	return robot_odometry_data

def sampleData(df_name, df, sample_time):
	column_headers = df.columns.values.tolist()
	print (" \n ")
	print ("Dataframe name: %s" %(df_name))
	print ("------------------------------------------------------")
	print ("Column Headers: %s" %(column_headers))
	num_samples = df.shape[0]

	values = [df.at[0, "t"], df.at[(num_samples-1), "t"]]
	total_timespan = values[1] - values[0]
	new_num_samples = math.floor(total_timespan / sample_time) + 1

	print ("Dataframe number of samples:  Original: %d and Resampled: %d" %(num_samples, new_num_samples))
	print ("First and last time values: %s and total time span (in seconds): %f" %(str(values), total_timespan))

	# if (df_name == "robot_groundtruth_data"):
	# 	times, x, y, theta = [], [], [], []
	# 	current_time = df["t"]

	# 	x.append(df.at[0, "x"])
	# 	y.append(df.at[0, "y"])
	# 	theta.append(df.at[0, "theta"])

	# 	i = 0
	# 	# count = 0
	# 	while (current_time <=  ):
	# 		times.append(current_time)
	# 		while (df.at["t", i] < current_time):
	# 			if (i == num_samples):
	# 				break
	# 			i += 1
	# 		interpolation_point = (current_time - df.at["t", i-1]) / (df.at["t", i] - df.at["t", i-1])
	# 		new_x = df.at["x", i-1] + (df.at["x", i] - df.at["x", i-1]) * interpolation_point
	# 		x.append(new_x)
	# 		new_y = df.at["y", i-1] + (df.at["y", i] - df.at["y", i-1]) * interpolation_point
	# 		y.append(new_y)

if __name__ == "__main__":
	print ("Number of arguments: %d " %(len(sys.argv)))
	print ("Arguments provided: %s " %(str(sys.argv)))
	folder_name = sys.argv[1]
	print ("Folder name: %s " %(folder_name))
	barcode_data_file, landmark_gt_data_file, robot_data_files = getDataFiles(folder_name, "Robot1")
	getDataFromFiles(barcode_data_file, landmark_gt_data_file, robot_data_files)