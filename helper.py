import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import transforms3d.euler as t3d
import helper
import torch

###################### Data Downloading Operations #########################


def download_data(file):
	from google_drive_downloader import GoogleDriveDownloader as gdd

	if file == 'train_data':
		file_id = '16YU-tdayVNBwM3XlPDgFrrzlPjhQN3PB'
	elif file == 'car_data':
		file_id = '1k9W75uhUFTfA_iK7YePGn5t9f4JhtgSe'

	if not os.path.exists(os.path.join(os.getcwd(), 'data', file)):
		gdd.download_file_from_google_drive(file_id=file_id,
											dest_path=os.path.join(
												os.getcwd(), 'data', file+'.zip'),
											showsize=True,
											unzip=True)

		os.remove(os.path.join(os.getcwd(), 'data', file+'.zip'))


###################### Data Handling Operations #########################

# Read the templates from a given file.
def read_templates(file_name, templates_dict):
	with open(os.path.join('data', templates_dict, file_name), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		data = []
		for row in csvreader:
			row = [float(i) for i in row]
			data.append(row)
	return data 										# n2 x 2048 x 3

# Read the file names having templates.


def template_files(templates_dict):
	with open(os.path.join('data', templates_dict, 'template_filenames.txt'), 'r') as file:
		files = file.readlines()
	files = [x.strip() for x in files]
	print(files)
	return files 										# 1 x n1

# Read the templates from each file.


def templates_data(templates_dict):
	# Read the available file names.
	files = template_files(templates_dict)
	data = []
	for i in range(len(files)):
		temp = read_templates(files[i], templates_dict)
		for i in temp:
			data.append(i)
	return np.asarray(data)								# (n1 x n2 x 2048 x 3) & n = n1 x n2

# Preprocess the templates and rearrange them.


def process_templates(templates_dict):
	data = templates_data(templates_dict)								# Read all the templates.
	print(data.shape[0]/2048)
	templates = []
	for i in range(data.shape[0]/2048):
		start_idx = i*2048
		end_idx = (i+1)*2048
		templates.append(data[start_idx:end_idx, :])
	return np.asarray(templates)						# Return all the templates (n x 2048 x 3)

# Read poses from given file.


def read_poses(templates_dict, filename):
	# Arguments:
		# filename:			Read data from a given file (string)
	# Output:
		# poses:			Return array of all the poses in the file (n x 6)

	with open(os.path.join('data', templates_dict, filename), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		poses = []
		for row in csvreader:
			row = [float(i) for i in row]
			poses.append(row)
	return np.asarray(poses)

# Read names of files in given data_dictionary.


def read_files(data_dict):
	with open(os.path.join('data', data_dict, 'files.txt')) as file:
		files = file.readlines()
		files = [x.split()[0] for x in files]
	return files[0]

# Read data from h5 file and return as templates.


def read_h5(file_name):
	import h5py
	f = h5py.File(file_name, 'r')
	templates = np.array(f.get('templates'))
	f.close()
	return templates


def read_noise_data(data_dict):
	import h5py
	f = h5py.File(os.path.join('data', data_dict, 'noise_data.h5'), 'r')
	templates = np.array(f.get('templates'))
	sources = np.array(f.get('sources'))
	f.close()
	return templates, sources


def read_pairs(data_dict, file_name):
	with open(os.path.join('data', data_dict, file_name), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		pairs = []
		for row in csvreader:
			row = [int(x) for x in row]
			pairs.append(row)
	return np.asarray(pairs)

# Main function to load data and return as templates array.


def loadData(data_dict):
	files = read_files(data_dict)  # Read file names.
	print(files)
	# Read templates from h5 file using given file_name.
	templates = read_h5(files)
	return templates

###################### Transformation Operations #########################


def rotate_point_cloud_by_angle_y(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  BxNx3 array, original batch of point clouds
			Return:
			  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape(
			(-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data


def rotate_point_cloud_by_angle_x(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  BxNx3 array, original batch of point clouds
			Return:
			  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[1, 0, 0],
									[0, cosval, -sinval],
									[0, sinval, cosval]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape(
			(-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data


def rotate_point_cloud_by_angle_z(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  BxNx3 array, original batch of point clouds
			Return:
			  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, -sinval, 0],
									[sinval, cosval, 0],
									[0, 0, 1]])
		shape_pc = batch_data[k, ...]
		# rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
		rotated_data[k, ...] = np.dot(rotation_matrix, shape_pc.reshape(
			(-1, 3)).T).T 		# Pre-Multiplication (changes done)
	return rotated_data

# Translate the data as per given translation vector.


def translate(data, shift):
	# Arguments:
		# data:					Point Cloud data (1 x num_points x 3)
		# shift:				Translation vector (1 x 3)

	try:
		data = np.asarray(data)
	except:
		pass
	return data+shift

# Apply the given transformation to given point cloud data.


# Transformation function for (2 & 4c, loss 8b)
def apply_transformation(datas, poses):
	# Arguments:
		# datas: 			Point Clouds (batch_size x num_points x 3)
		# poses: 			translation+euler (Batch_size x 6)
	# Output:
		# transformed_data: Transformed Point Clouds by given poses (batch_size x num_points x 3)
	transformed_data = np.copy(datas)
	for i in range(datas.shape[0]):
		transformed_data[i, :, :] = rotate_point_cloud_by_angle_z(
			transformed_data[i, :, :], poses[i, 5])
		transformed_data[i, :, :] = rotate_point_cloud_by_angle_y(
			transformed_data[i, :, :], poses[i, 4])
		transformed_data[i, :, :] = rotate_point_cloud_by_angle_x(
			transformed_data[i, :, :], poses[i, 3])
		transformed_data[i, :, :] = translate(transformed_data[i, :, :], [
											  poses[i, 0], poses[i, 1], poses[i, 2]])
	return transformed_data

# Convert poses from 6D to 7D. 			# For loss function ( 8a )


def poses_euler2quat(poses):
	# Arguments:
		# poses: 			6D pose (translation + euler) (batch_size x 6)
	# Output:
		# new_poses: 		7D pose (translation + quaternions) (batch_size x 7)

	new_poses = []					# Store 7D poses
	for i in range(poses.shape[0]):
		# Convert from euler to quaternion. (1x4)
		temp = t3d.euler2quat(poses[i, 3], poses[i, 4], poses[i, 5])
		temp1 = [poses[i, 0], poses[i, 1], poses[i, 2], temp[0], temp[1],
				 temp[2], temp[3]]		# Add translation & Quaternion (1x7)
		new_poses.append(temp1)
	return np.asarray(new_poses)

# Geenerate random poses equal to batch_size.


def generate_poses(batch_size):
	# Arguments:
		# batch_size:		No of 6D poses required.
	# Output:
		# poses:			Array of poses with translation and rotation (euler angles in radians) (batch_size x 6)

	poses = []					# List to store the 6D poses.
	for i in range(batch_size):
		# Generate random translations.
		x = np.round(2*np.random.random_sample()-1, 2)
		y = np.round(2*np.random.random_sample()-1, 2)
		z = np.round(2*np.random.random_sample()-1, 2)
		# Generate random rotations.
		x_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2), 3)
		y_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2), 3)
		z_rot = np.round(np.pi*np.random.random_sample()-(np.pi/2), 3)
		poses.append([x, y, z, x_rot, y_rot, z_rot])
	return np.array(poses).reshape((batch_size, 6))

# Convert 6D poses to transformation matrix.	# (for 4b)


def transformation(poses):
	# Arguments:
		# poses: 					6D (x,y,z,euler_x,euler_y,euler_z) (in radians)
	# Output
		# transformation_matrix: 	batch_size x 4 x 4

	transformation_matrix = np.zeros((poses.shape[0], 4, 4))
	transformation_matrix[:, 3, 3] = 1
	for i in range(poses.shape[0]):
		# Calculate rotation matrix using transforms3d
		rot = t3d.euler2mat(poses[i, 5], poses[i, 4], poses[i, 3], 'szyx')
		# Store rotation matrix in transformation matrix.
		transformation_matrix[i, 0:3, 0:3] = rot
		# Store translations in transformation matrix.
		transformation_matrix[i, 0:3, 3] = poses[i, 0:3]
	return transformation_matrix

# Convert poses (quaternions) to transformation matrix and apply on point cloud.


def transformation_quat2mat(poses, TRANSFORMATIONS, templates_data):		# (for 4b)
	# Arguments:
		# poses: 					7D (x,y,z,quat_q0,quat_q1,quat_q2,quat_q3) (in radians) (batch_size x 7)
		# TRANSFORMATIONS: 			Overall tranformation matrix.
		# template_data: 			Point Cloud (batch_size x num_points x 3)
	# Output
		# TRANSFORMATIONS: 			Batch_size x 4 x 4
		# templates_data:			Transformed template data (batch_size x num_points x 3)

	poses = np.array(poses)												# Convert poses to array.
	poses = poses.reshape(poses.shape[-2], poses.shape[-1])
	for i in range(poses.shape[0]):
		transformation_matrix = np.zeros((4, 4))
		transformation_matrix[3, 3] = 1
		# Calculate rotation matrix using transforms3d
		rot = t3d.quat2mat(
			[poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6]])
		# Store rotation matrix in transformation matrix.
		transformation_matrix[0:3, 0:3] = rot
		# Store translations in transformation matrix.
		transformation_matrix[0:3, 3] = poses[i, 0:3]
		# 4b (Multiply tranfromation matrix to Initial Transfromation Matrix)
		TRANSFORMATIONS[i, :, :] = np.dot(
			transformation_matrix, TRANSFORMATIONS[i, :, :])
		# Apply Rotation to Template Data
		templates_data[i, :, :] = np.dot(rot, templates_data[i, :, :].T).T
		templates_data[i, :, :] = templates_data[i, :, :] + \
			poses[i, 0:3]			# Apply translation to template data
	return TRANSFORMATIONS, templates_data

# Convert the Final Transformation Matrix to Translation + Orientation (Euler Angles in Degrees)


def find_final_pose(TRANSFORMATIONS):
	# Arguments:
		# TRANSFORMATIONS: 			transformation matrix (batch_size x 4 x 4)
	# Output:
		# final_pose:				final pose predicted by network (batch_size x 6)

	# Array to store the poses.
	final_pose = np.zeros((TRANSFORMATIONS.shape[0], 6))
	for i in range(TRANSFORMATIONS.shape[0]):
		rot = TRANSFORMATIONS[i, 0:3, 0:3]					# Extract rotation matrix.
		# Convert rotation matrix to euler angles. (Pre-multiplication)
		euler = t3d.mat2euler(rot, 'szyx')
		final_pose[i, 3:6] = [euler[2], euler[1],
							  euler[0]]		# Store the translation
		# Store the euler angles.
		final_pose[i, 0:3] = TRANSFORMATIONS[i, 0:3, 3].T
	return final_pose

# Convert the Final Transformation Matrix to Translation + Orientation (Euler Angles in Degrees)


def find_final_pose_inv(TRANSFORMATIONS_ip):
	# Arguments:
		# TRANSFORMATIONS: 			transformation matrix (batch_size x 4 x 4)
	# Output:
		# final_pose:				final pose predicted by network (batch_size x 6)

	TRANSFORMATIONS = np.copy(TRANSFORMATIONS_ip)
	# Array to store the poses.
	final_pose = np.zeros((TRANSFORMATIONS.shape[0], 6))
	for i in range(TRANSFORMATIONS.shape[0]):
		TRANSFORMATIONS[i] = np.linalg.inv(TRANSFORMATIONS[i])
		rot = TRANSFORMATIONS[i, 0:3, 0:3]					# Extract rotation matrix.
		# Convert rotation matrix to euler angles. (Pre-multiplication)
		euler = t3d.mat2euler(rot, 'szyx')
		final_pose[i, 3:6] = [euler[2], euler[1],
							  euler[0]]		# Store the translation
		# Store the euler angles.
		final_pose[i, 0:3] = TRANSFORMATIONS[i, 0:3, 3].T
	return final_pose

# Subtract the centroids from source and template (Like ICP) and then find the pose.


def centroid_subtraction(source_data, template_data):
	# Arguments:
		# source_data:			Source Point Clouds (batch_size x num_points x 3)
		# template_data:		Template Point Clouds (batch_size x num_points x 3)
	# Output:
		# source_data:					Centroid subtracted from source point cloud (batch_size x num_points x 3)
		# template_data:				Centroid subtracted from template point cloud (batch_size x num_points x 3)
		# centroid_translation_pose:	Apply this pose after final iteration. (batch_size x 7)

	centroid_translation_pose = np.zeros((source_data.shape[0], 7))
	for i in range(source_data.shape[0]):
		source_centroid = np.mean(source_data[i], axis=0)
		template_centroid = np.mean(template_data[i], axis=0)
		source_data[i] = source_data[i] - source_centroid
		template_data[i] = template_data[i] - template_centroid
		centroid_translation = source_centroid - template_centroid
		centroid_translation_pose[i] = np.array(
			[centroid_translation[0], centroid_translation[1], centroid_translation[2], 1, 0, 0, 0])
	return source_data, template_data, centroid_translation_pose


def inverse_pose(pose):
	transformation_pose = np.zeros((4, 4))
	transformation_pose[3, 3] = 1
	transformation_pose[0:3, 0:3] = t3d.euler2mat(
		pose[5], pose[4], pose[3], 'szyx')
	transformation_pose[0, 3] = pose[0]
	transformation_pose[1, 3] = pose[1]
	transformation_pose[2, 3] = pose[2]
	transformation_pose = np.linalg.inv(transformation_pose)
	pose_inv = np.zeros((1, 6))[0]
	pose_inv[0] = transformation_pose[0, 3]
	pose_inv[1] = transformation_pose[1, 3]
	pose_inv[2] = transformation_pose[2, 3]
	orient_inv = t3d.mat2euler(transformation_pose[0:3, 0:3], 'szyx')
	pose_inv[3] = orient_inv[2]
	pose_inv[4] = orient_inv[1]
	pose_inv[5] = orient_inv[0]
	return pose_inv


###################### Shuffling Operations #########################

# Randomly shuffle given array of poses for training procedure.
def shuffle_templates(templates):
	# Arguments:
		# templates:			Input array of templates to get randomly shuffled (batch_size x num_points x 3)
	# Output:
		# shuffled_templates:	Randomly ordered poses (batch_size x num_points x 3)

	# Array to store shuffled templates.
	shuffled_templates = np.zeros(templates.shape)
	templates_idxs = np.arange(0, templates.shape[0])
	# Randomly shuffle template indices.
	np.random.shuffle(templates_idxs)
	for i in range(templates.shape[0]):
		# Rearrange them as per shuffled indices.
		shuffled_templates[i, :, :] = templates[templates_idxs[i], :, :]
	return shuffled_templates

# Randomly shuffle given array of poses for training procedure.


def shuffle_poses(poses):
	# Arguments:
		# poses:			Input array of poses to get randomly shuffled (batch_size x n)
	# Output:
		# shuffled_poses:	Randomly ordered poses (batch_size x n)

	shuffled_poses = np.zeros(poses.shape)				# Array to store shuffled poses.
	poses_idxs = np.arange(0, poses.shape[0])
	np.random.shuffle(poses_idxs)						# Shuffle the indexes of poses.
	for i in range(poses.shape[0]):
		# Rearrange them as per shuffled indexes.
		shuffled_poses[i, :] = poses[poses_idxs[i], :]
	return shuffled_poses

# Generate random transformation/pose for data augmentation.


def random_trans():
	# Output:
		# 6D pose with first 3 translation values and last 3 euler angles in radian about x,y,z-axes. (1x6)

	# Generate random translations.
	x_trans, y_trans, z_trans = 0.4*np.random.uniform()-0.2, 0.4*np.random.uniform() - \
		0.2, 0.4*np.random.uniform()-0.2
	# Generate random rotation angles.
	x_rot, y_rot, z_rot = (np.pi/9)*np.random.uniform()-(np.pi/18), (np.pi/9) * \
		np.random.uniform()-(np.pi/18), (np.pi/9)*np.random.uniform()-(np.pi/18)
	return [x_trans, y_trans, z_trans, x_rot, y_rot, z_rot]

# Generate random poses for each batch to train the network.


def generate_random_poses(batch_size):
	# Arguments:
		# Batch_size:		No of poses in the output
	# Output:
		# poses:			Randomly generated poses (batch_size x 6)

	poses = []
	for i in range(batch_size):
		x_trans, y_trans, z_trans = 2*np.random.uniform()-1, 2*np.random.uniform() - \
			1, 2*np.random.uniform()-1										# Generate random translation
		x_rot, y_rot, z_rot = (np.pi)*np.random.uniform()-(np.pi/2), (np.pi)*np.random.uniform()-(
			np.pi/2), (np.pi)*np.random.uniform()-(np.pi/2)  # Generate random orientation
		poses.append([np.round(x_trans, 4), np.round(y_trans, 4), np.round(z_trans, 4), np.round(
			x_rot, 4), np.round(y_rot, 4), np.round(z_rot, 4)])		# round upto 4 decimal digits
	return np.array(poses)


def select_random_points(source_data, num_point):
	random_source_data = np.copy(source_data)
	# Find indexes of source data.
	idx = np.arange(random_source_data.shape[1])
	np.random.shuffle(idx)									# Shuffle indexes.
	# Shuffle data as per shuffled indexes.
	random_source_data = random_source_data[:, idx, :]
	return random_source_data[:, 0:num_point, :]


def add_noise(source_data):
	for i in range(source_data.shape[0]):
		mean = 0
		for j in range(source_data.shape[1]):
			# Generate random variance value b/w 0 to 0.1
			sigma = 0.04*np.random.random_sample()
			# Add gaussian noise.
			source_data[i, j, :] = source_data[i, j, :] + \
				np.random.normal(mean, sigma, source_data[i, j].shape)
	return source_data

###################### Tensor Operations #########################


def rotate_point_cloud_by_angle_y_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  Nx3 array, original batch of point clouds
			Return:
			  Nx3 array, rotated batch of point clouds
	"""
	cosval = torch.cos(rotation_angle)
	sinval = torch.sin(rotation_angle)
	rotation_matrix = torch.reshape(
		[[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], [3, 3])
	data = data.reshape(-1, 3)
	rotated_data = torch.t(torch.dot(rotation_matrix, torch.t(data)))
	return rotated_data


def rotate_point_cloud_by_angle_x_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  Nx3 array, original batch of point clouds
			Return:
			  Nx3 array, rotated batch of point clouds
	"""
	cosval = torch.cos(rotation_angle)
	sinval = torch.sin(rotation_angle)
	rotation_matrix = torch.reshape(
		[[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]], [3, 3])
	data = torch.reshape(data, [-1, 3])
	rotated_data = torch.t(torch.dot(rotation_matrix, torch.t(data)))
	return rotated_data


def rotate_point_cloud_by_angle_z_tensor(data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
			Input:
			  Nx3 array, original batch of point clouds
			Return:
			  Nx3 array, rotated batch of point clouds
	"""
	cosval = torch.cos(rotation_angle)
	sinval = torch.sin(rotation_angle)
	rotation_matrix = torch.reshape(
		[[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]], [3, 3])
	data = torch.reshape(data, [-1, 3])
	rotated_data = torch.t(torch.dot(rotation_matrix, torch.t(data)))
	return rotated_data


def translate_tensor(data, shift):
	# Add the translation vector to given tensor. (num_point x 3)
	return data + shift

# Tranform the data as per given poses with orientation as euler in degrees.


def transformation_tensor(datas, poses):
	# Arguments:
		# datas: 				Tensor of Point Cloud (batch_size x num_points x 3)
		# poses: 				Tensor of Poses (translation + euler angles in degrees) (batch_size x num_points x 3)
	# Ouput:
		# transformed_data:		Tensor of transformed point cloud (batch_size x num_points x 3)

	# Tensor to store the transformed point clouds as tensor.
	transformed_data = torch.zeros([datas.shape[1], datas.shape[2]])
	for i in range(datas.shape[0]):
		transformed_data_t = rotate_point_cloud_by_angle_x_tensor(
			datas[i, ...], poses[i, 3])				# Rotate about x-axis
		transformed_data_t = rotate_point_cloud_by_angle_y_tensor(
			transformed_data_t, poses[i, 4])		# Rotate about y-axis
		transformed_data_t = rotate_point_cloud_by_angle_z_tensor(
			transformed_data_t, poses[i, 5])		# Rotate about z-axis
		# Translate by given vector.
		transformed_data_t = translate_tensor(
			transformed_data_t, [poses[i, 0], poses[i, 1], poses[i, 2]])
		# Append the transformed tensor point cloud.
		transformed_data = torch.cat([transformed_data, transformed_data_t], 0)
	# Reshape the transformed tensor and remove first one. (batch_size x num_point x 3)
	transformed_data = torch.reshape(
		transformed_data, [-1, datas.shape[1], datas.shape[2]])[1:]
	return transformed_data

# Tranform the data as per given poses with orientation as quaternion.


def transformation_quat_tensor(data, quat, translation):
	# Arguments:
		# data:					Tensor of Point Cloud. (batch_size x num_point x 3)
		# quat:					Quaternion tensor to generate rotation matrix.	(batch_size x 4)
		# translation:			Translation tensor to translate the point cloud. (batch_size x 3)
	# Outputs:
		# transformed_data: 	Tensor of Rotated and Translated Point Cloud Data. (batch_size x num_points x 3)

	# Tensor to store transformed data.
	transformed_data = torch.zeros([data.shape[1], 3])
	for i in range(quat.shape[0]):
		# Seperate each quaternion value.
		q0 = quat[i, 0]
		q1 = quat[i, 1]
		q2 = quat[i, 2]
		q3 = quat[i, 3]
		# Convert quaternion to rotation matrix.
		# Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
		# A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
		R = torch.Tensor([[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
		 	[2*(q1*q2+q0*q3), q0*q0+q2*q2-q1*q1-q3*q3, 2*(q2*q3-q0*q1)],
		 	[2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0+q3*q3-q1*q1-q2*q2]])

		# Convert R into a single tensor of shape 3x3.
		R = torch.reshape(R, [3, 3])
		# torch.tensordot: Arg: tensor1, tensor2, axes
		# axes defined for tensor1 & tensor2 should be of same size.
		# axis 1 of R is of size 3 and axis 0 of data (3xnum_points) is of size 3.

		# Rotate the data. (num_points x 3)
		#temp_rotated_data = torch.t(torch.mm(R, torch.t(data[i, ...])))
		temp_rotated_data = torch.t(R.matmul(torch.t(data[i, ...]).cpu()))
		# Add the translation (num_points x 3)
		temp_rotated_data = temp_rotated_data + translation[i].cpu()
		# Append data (batch_size x num_points x 3)
		transformed_data = torch.cat([transformed_data, temp_rotated_data], 0)

	# Reshape data and remove first point cloud. (batch_size x num_point x 3)
	transformed_data = torch.reshape(
		transformed_data, [-1, data.shape[1], 3])[1:]
	return transformed_data


###################### Display Operations #########################

# Display data inside ModelNet files.
def display_clouds(filename, model_no):
	# Arguments:
		# filename:			Name of file to read the data from. (string)
		# model_no:			Number to choose the model inside that file. (int)

	data = []
	# Read the entire data from that file.
	with open(os.path.join('data', 'templates', filename), 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			row = [float(x) for x in row]
			data.append(row)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	data = np.asarray(data)

	start_idx = model_no*2048
	end_idx = (model_no+1)*2048
	# Choose specific data related to the given model number.
	data = data[start_idx:end_idx, :]

	X, Y, Z = [], [], []
	for row in data:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	ax.scatter(X, Y, Z)
	plt.show()

# Display given Point Cloud Data in blue color (default).


def display_clouds_data(data):
	# Arguments:
		# data: 		array of point clouds (num_points x 3)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	try:
		data = data.tolist()
	except:
		pass
	X, Y, Z = [], [], []
	for row in data:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	ax.scatter(X, Y, Z)
	plt.show()

# Display given template, source and predicted point cloud data.


def display_three_clouds(data1, data2, data3, title):
	# Arguments:
		# data1 		Template Data (num_points x 3) (Red)
		# data2			Source Data (num_points x 3) (Green)
		# data3			Predicted Data (num_points x 3) (Blue)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	try:
		data1 = data1.tolist()
		data2 = data2.tolist()
		data3 = data3.tolist()
	except:
		pass
	# Add Template Data in Plot
	X, Y, Z = [], [], []
	for row in data1:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l1 = ax.scatter(X, Y, Z, c=[1, 0, 0, 1])
	# Add Source Data in Plot
	X, Y, Z = [], [], []
	for row in data2:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l2 = ax.scatter(X, Y, Z, c=[0, 1, 0, 0.5])
	# Add Predicted Data in Plot
	X, Y, Z = [], [], []
	for row in data3:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l3 = ax.scatter(X, Y, Z, c=[0, 0, 1, 0.5])

	# Add details to Plot.
	plt.legend((l1, l2, l3), ('Template Data', 'Source Data',
							  'Predicted Data'), prop={'size': 15}, markerscale=4)
	ax.tick_params(labelsize=10)
	ax.set_xlabel('X-axis', fontsize=15)
	ax.set_ylabel('Y-axis', fontsize=15)
	ax.set_zlabel('Z-axis', fontsize=15)
	# ax.set_xlim(-1,1.25)
	# ax.set_ylim(-1,1)
	# ax.set_zlim(-0.5,1.25)
	plt.title(title, fontdict={'fontsize': 25})
	ax.xaxis.set_tick_params(labelsize=15)
	ax.yaxis.set_tick_params(labelsize=15)
	ax.zaxis.set_tick_params(labelsize=15)
	plt.show()

# Display template, source, predicted point cloud data with results after each iteration.


def display_itr_clouds(data1, data2, data3, ITR, title):
	# Arguments:
		# data1 		Template Data (num_points x 3) (Red)
		# data2			Source Data (num_points x 3) (Green)
		# data3			Predicted Data (num_points x 3) (Blue)
		# ITR 			Point Clouds obtained after each iteration (iterations x batch_size x num of points x 3) (Yellow)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	print(ITR.shape)		# Display Number of Point Clouds in ITR.
	try:
		data1 = data1.tolist()
		data2 = data2.tolist()
		data3 = data3.tolist()
	except:
		pass
	# Add Template Data in Plot
	X, Y, Z = [], [], []
	for row in data1:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l1 = ax.scatter(X, Y, Z, c=[1, 0, 0, 1])
	# Add Source Data in Plot
	X, Y, Z = [], [], []
	for row in data2:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l2 = ax.scatter(X, Y, Z, c=[0, 1, 0, 1])
	# Add Predicted Data in Plot
	X, Y, Z = [], [], []
	for row in data3:
		X.append(row[0])
		Y.append(row[1])
		Z.append(row[2])
	l3 = ax.scatter(X, Y, Z, c=[0, 0, 1, 1])
	# Add point clouds after each iteration in Plot.
	for itr_data in ITR:
		X, Y, Z = [], [], []
		for row in itr_data[0]:
			X.append(row[0])
			Y.append(row[1])
			Z.append(row[2])
		ax.scatter(X, Y, Z, c=[1, 1, 0, 0.5])

	# Add details to Plot.
	plt.legend((l1, l2, l3), ('Template Data', 'Source Data',
							  'Predicted Data'), prop={'size': 15}, markerscale=4)
	ax.tick_params(labelsize=10)
	ax.set_xlabel('X-axis', fontsize=15)
	ax.set_ylabel('Y-axis', fontsize=15)
	ax.set_zlabel('Z-axis', fontsize=15)
	plt.title(title, fontdict={'fontsize': 25})
	ax.xaxis.set_tick_params(labelsize=15)
	ax.yaxis.set_tick_params(labelsize=15)
	ax.zaxis.set_tick_params(labelsize=15)
	plt.show()

# Log test results to a folder


def log_test_results(LOG_DIR, filename, log):

	# It will log the data in following sequence in csv format:
	# Sr. No., time taken, number of iterations, translation error, rotation error.

	# If log dir doesn't exists create one.
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)

	# Find params from the dictionary.
	ITR, TIME = log['ITR'], log['TIME']
	Trans_Err, Rot_Err = log['Trans_Err'], log['Rot_Err']
	idxs_5_5, idxs_10_1, idxs_20_2 = log['idxs_5_5'], log['idxs_10_1'], log['idxs_20_2']
	num_batches = log['num_batches']

	# Find mean and variance.
	TIME_mean = sum(TIME)/len(TIME)

	# Log all the data in a csv file.
	import csv
	with open(os.path.join(LOG_DIR, filename+'.csv'), 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		for i in range(len(TIME)):
			csvwriter.writerow([i, TIME[i], ITR[i], Trans_Err[i], Rot_Err[i]])

	if len(idxs_5_5) != 0:
		accuray_5_5 = len(idxs_5_5)/(num_batches*1.0)
		mean_5_5_rot_err = np.sum(np.array(Rot_Err)[idxs_5_5])/len(idxs_5_5)
		var_5_5_rot_err = np.var(np.array(Rot_Err)[idxs_5_5])
		mean_5_5_trans_err = np.sum(
			np.array(Trans_Err)[idxs_5_5])/len(idxs_5_5)
		var_5_5_trans_err = np.var(np.array(Trans_Err)[idxs_5_5])
		mean_5_5_itr = np.sum(np.array(ITR)[idxs_5_5])/len(idxs_5_5)
		var_5_5_itr = np.var(np.array(ITR)[idxs_5_5])
		mean_5_5_time = np.sum(np.array(TIME)[idxs_5_5])/len(idxs_5_5)
		var_5_5_time = np.var(np.array(TIME)[idxs_5_5])
	else:
		accuray_5_5, mean_5_5_rot_err, var_5_5_rot_err, mean_5_5_trans_err, var_5_5_trans_err, mean_5_5_itr, var_5_5_itr, mean_5_5_time, var_5_5_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

	if len(idxs_10_1) != 0:
		accuray_10_1 = len(idxs_10_1)/(num_batches*1.0)
		mean_10_1_rot_err = np.sum(np.array(Rot_Err)[idxs_10_1])/len(idxs_10_1)
		var_10_1_rot_err = np.var(np.array(Rot_Err)[idxs_10_1])
		mean_10_1_trans_err = np.sum(
			np.array(Trans_Err)[idxs_10_1])/len(idxs_10_1)
		var_10_1_trans_err = np.var(np.array(Trans_Err)[idxs_10_1])
		mean_10_1_itr = np.sum(np.array(ITR)[idxs_10_1])/len(idxs_10_1)
		var_10_1_itr = np.var(np.array(ITR)[idxs_10_1])
		mean_10_1_time = np.sum(np.array(TIME)[idxs_10_1])/len(idxs_10_1)
		var_10_1_time = np.var(np.array(TIME)[idxs_10_1])
	else:
		accuray_10_1, mean_10_1_rot_err, var_10_1_rot_err, mean_10_1_trans_err, var_10_1_trans_err, mean_10_1_itr, var_10_1_itr, mean_10_1_time, var_10_1_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

	if len(idxs_20_2) != 0:
		# Find accuracies:
		accuray_20_2 = len(idxs_20_2)/(num_batches*1.0)
		# Find mean rotation error.
		mean_20_2_rot_err = np.sum(np.array(Rot_Err)[idxs_20_2])/len(idxs_20_2)
		# Find variance of rotation error.
		var_20_2_rot_err = np.var(np.array(Rot_Err)[idxs_20_2])
		# Find mean translation error.
		mean_20_2_trans_err = np.sum(
			np.array(Trans_Err)[idxs_20_2])/len(idxs_20_2)
		# Find variance of translation error.
		var_20_2_trans_err = np.var(np.array(Trans_Err)[idxs_20_2])
		# Find mean iterations.
		mean_20_2_itr = np.sum(np.array(ITR)[idxs_20_2])/len(idxs_20_2)
		# Find variance of iterations.
		var_20_2_itr = np.var(np.array(ITR)[idxs_20_2])
		# Find mean time required.
		mean_20_2_time = np.sum(np.array(TIME)[idxs_20_2])/len(idxs_20_2)
		# Find variance of time.
		var_20_2_time = np.var(np.array(TIME)[idxs_20_2])
	else:
		accuray_20_2, mean_20_2_rot_err, var_20_2_rot_err, mean_20_2_trans_err, var_20_2_trans_err, mean_20_2_itr, var_20_2_itr, mean_20_2_time, var_20_2_time = 0, 0, 0, 0, 0, 0, 0, 0, 0

	with open(os.path.join(LOG_DIR, filename+'.txt'), 'w') as file:
		file.write("Mean of Time: {}\n".format(TIME_mean))

		file.write("\n")

		file.write("###### 5 Degree & 0.05 Units ######\n")
		file.write("Accuray: {}%\n".format(accuray_5_5*100))
		file.write("Mean rotational error: {}\n".format(mean_5_5_rot_err))
		file.write("Mean translation error: {}\n".format(mean_5_5_trans_err))
		file.write("Mean time: {}\n".format(mean_5_5_time))
		file.write("Var time: {}\n".format(var_5_5_time))
		file.write("Var translation error: {}\n".format(var_5_5_trans_err))
		file.write("Var rotational error: {}\n".format(var_5_5_rot_err))
		file.write("Mean Iterations: {}\n".format(mean_5_5_itr))
		file.write("Var Iterations: {}\n".format(var_5_5_itr))

		file.write("\n")

		file.write("###### 10 Degree & 0.1 Units ######\n")
		file.write("Accuray: {}%\n".format(accuray_10_1*100))
		file.write("Mean rotational error: {}\n".format(mean_10_1_rot_err))
		file.write("Mean translation error: {}\n".format(mean_10_1_trans_err))
		file.write("Mean time: {}\n".format(mean_10_1_time))
		file.write("Var time: {}\n".format(var_10_1_time))
		file.write("Var translation error: {}\n".format(var_10_1_trans_err))
		file.write("Var rotational error: {}\n".format(var_10_1_rot_err))
		file.write("Mean Iterations: {}\n".format(mean_10_1_itr))
		file.write("Var Iterations: {}\n".format(var_10_1_itr))

		file.write("\n")

		file.write("###### 20 Degree & 0.2 Units ######\n")
		file.write("Accuray: {}%\n".format(accuray_20_2*100))
		file.write("Mean rotational error: {}\n".format(mean_20_2_rot_err))
		file.write("Mean translation error: {}\n".format(mean_20_2_trans_err))
		file.write("Mean time: {}\n".format(mean_20_2_time))
		file.write("Var time: {}\n".format(var_20_2_time))
		file.write("Var translation error: {}\n".format(var_20_2_trans_err))
		file.write("Var rotational error: {}\n".format(var_20_2_rot_err))
		file.write("Mean Iterations: {}\n".format(mean_20_2_itr))
		file.write("Var Iterations: {}\n".format(var_20_2_itr))

	plt.hist(Rot_Err, np.arange(0, 185, 5))
	plt.xlim(0, 180)
	plt.savefig(os.path.join(LOG_DIR, 'rot_err_hist.jpeg'),
				dpi=500, quality=100)
	plt.figure()
	plt.hist(Trans_Err, np.arange(0, 1.01, 0.01))
	plt.xlim(0, 1)
	plt.savefig(os.path.join(LOG_DIR, 'trans_err_hist.jpeg'),
				dpi=500, quality=100)


if __name__ == '__main__':
	# a = np.array([[0,0,0,0,0,0],[0,0,0,90,0,0]])
	# print a.shape
	# a = poses_euler2quat(a)
	# print(a[1,3]*a[1,3]+a[1,4]*a[1,4]+a[1,5]*a[1,5]+a[1,6]*a[1,6])
	# print(a[0,3]*a[0,3]+a[0,4]*a[0,4]+a[0,5]*a[0,5]+a[0,6]*a[0,6])
	# print a.shape
	# display_clouds('airplane_templates.csv',0)

	templates = helper.process_templates('multi_model_templates')
	# templates = helper.process_templates('templates')
	# airplane = templates[0,:,:]
	idx = 199
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# start = idx*2048
	# end = (idx+1)*2048
	ax.scatter(templates[idx, :, 0],
			   templates[idx, :, 1], templates[idx, :, 2])
	plt.show()
	print(templates.shape)
