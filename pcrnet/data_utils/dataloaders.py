import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
from .. ops import transform_functions, se3

def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))

def load_data(train):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def deg_to_rad(deg):
	return np.pi / 180 * deg

def create_random_transform(dtype, max_rotation_deg, max_translation):
	max_rotation = deg_to_rad(max_rotation_deg)
	rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
	trans = np.random.uniform(-max_translation, max_translation, [1, 3])
	quat = transform_functions.euler_to_quaternion(rot, "xyz")

	vec = np.concatenate([quat, trans], axis=1)
	vec = torch.tensor(vec, dtype=dtype)
	return vec

class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names

class RegistrationData(Dataset):
	def __init__(self, algorithm, data_class=ModelNet40Data()):
		super(RegistrationData, self).__init__()
		self.algorithm = 'iPCRNet'
		
		self.set_class(data_class)
		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			from .. ops.transform_functions import PCRNetTransform
			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
		self.transforms.index = index				# for fixed transformations in PCRNet.
		source = self.transforms(template)
		igt = self.transforms.igt
		return template, source, igt