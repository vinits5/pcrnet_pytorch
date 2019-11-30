#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PCRNet
from util import transform_point_cloud, npmat2euler
import numpy as np
import helper

# Use this function to read the data.
def read_data():
	# Output -> 
		# source:		Torch tensor on CPU [Nx3]
		# template:		Torch tensor on CPU [Nx3]
		# rotation_ab:	Torch tensor on CPU [Nx3]
	template = helper.loadData('train_data')
	template = template[0,0:1024,:].reshape(1,-1,3)
	poses = np.array([[0, 0.5, 0, 0*(np.pi/180), 40*(np.pi/180), 0*(np.pi/180)]])
	source = helper.apply_transformation(template, poses)
	return torch.from_numpy(source), torch.from_numpy(template)

def test_one_pair(args, net):
	source, template = read_data()
	net.eval()
	src = source.to(args.device)
	target = template.to(args.device)
	src = src.permute(0,2,1)
	target = target.permute(0,2,1)
	batch_size = src.size(0)
	rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)	
	transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
	transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

	transformed_src = transformed_src.permute(0,2,1)
	print(rotation_ab_pred, translation_ab_pred)

	return source.numpy(), template.numpy(), transformed_src.cpu().detach().numpy(), rotation_ab_pred, translation_ab_pred

def main():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	
	# Settings for network.
	parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--model', type=str, default='pcrnet', metavar='N',
						choices=['dcp'],
						help='Model to use, [dcp]')
	parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
						choices=['pointnet', 'dgcnn'],
						help='Embedding nn to use, [pointnet, dgcnn]')
	parser.add_argument('--pointer', type=str, default='identity', metavar='N',
						choices=['identity', 'transformer'],
						help='Attention-based pointer generator to use, [identity, transformer]')
	parser.add_argument('--head', type=str, default='mlp', metavar='N',
						choices=['mlp', 'svd', ],
						help='Head to use, [mlp, svd]')
	parser.add_argument('--iterations', type=int, default=1, help='[No of iterations for PCRNet]')
	
	# Settings for training
	parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
						help='Dimension of embeddings')
	parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
						help='Size of batch)')
	parser.add_argument('--epochs', type=int, default=250, metavar='N',
						help='number of episode to train ')
	parser.add_argument('--device', action='store_true', default=False,
						help='enables CUDA training')
	parser.add_argument('--seed', type=int, default=1234, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--eval', action='store_true', default=False,
						help='evaluate the model')
	
	# Settings for attention
	parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
						help='Num of blocks of encoder&decoder')
	parser.add_argument('--n_heads', type=int, default=4, metavar='N',
						help='Num of heads in multiheadedattention')
	parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
						help='Num of dimensions of fc in transformer')
	parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
						help='Dropout ratio in transformer')
	
	
	parser.add_argument('--cycle', type=bool, default=False, metavar='N',
						help='Whether to use cycle consistency')
	
	# Settings for dataset
	parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
						help='Wheter to add gaussian noise')
	parser.add_argument('--unseen', type=bool, default=False, metavar='N',
						help='Wheter to test on unseen category')
	parser.add_argument('--num_points', type=int, default=1024, metavar='N',
						help='Num of points to use')
	parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
						help='dataset to use')
	parser.add_argument('--factor', type=float, default=4, metavar='N',
						help='Divided factor for rotations')
	parser.add_argument('--model_path', type=str, default='./checkpoints/pcrnet_1/models/model.best.t7', metavar='N',
						help='Pretrained model path')

	args = parser.parse_args()

	use_cuda = torch.cuda.is_available()
	args.device = torch.device("cuda" if use_cuda else "cpu")

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	if args.model == 'pcrnet':
		net = PCRNet(args).to(args.device)
		model_path = args.model_path
		net.load_state_dict(torch.load(model_path), strict=False)
	else:
		raise Exception('Not implemented')
	
	source, template, transformed_src, _, _ = test_one_pair(args, net)
	print(source.shape, template.shape, transformed_src.shape)

	import helper
	helper.display_three_clouds(template[0], source[0], transformed_src[0], 'Results')

	print('FINISH')


if __name__ == '__main__':
	main()
