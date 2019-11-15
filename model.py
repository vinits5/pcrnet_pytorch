#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import quat2mat, transform_point_cloud, combine_transformations


# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def nearest_neighbor(src, dst):
	inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
	distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
																										   dim=0,
																										   keepdim=True)
	distances, indices = distances.topk(k=1, dim=-1)
	return distances, indices

def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

	idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
	return idx

def get_graph_feature(x, k=20):
	# x = x.squeeze()
	idx = knn(x, k=k)  # (batch_size, num_points, k)
	batch_size, num_points, _ = idx.size()
	device = torch.device('cuda')

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

	idx = idx + idx_base

	idx = idx.view(-1)

	_, num_dims, _ = x.size()

	x = x.transpose(2,
					1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

	feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

	return feature


class PointNet(nn.Module):
	def __init__(self, emb_dims=512):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		return x


class DGCNN(nn.Module):
	def __init__(self, emb_dims=512):
		super(DGCNN, self).__init__()
		self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
		self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(emb_dims)

	def forward(self, x):
		batch_size, num_dims, num_points = x.size()
		x = get_graph_feature(x)
		x = F.relu(self.bn1(self.conv1(x)))
		x1 = x.max(dim=-1, keepdim=True)[0]

		x = F.relu(self.bn2(self.conv2(x)))
		x2 = x.max(dim=-1, keepdim=True)[0]

		x = F.relu(self.bn3(self.conv3(x)))
		x3 = x.max(dim=-1, keepdim=True)[0]

		x = F.relu(self.bn4(self.conv4(x)))
		x4 = x.max(dim=-1, keepdim=True)[0]

		x = torch.cat((x1, x2, x3, x4), dim=1)

		x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
		return x


class MLPHead(nn.Module):
	def __init__(self, args):
		super(MLPHead, self).__init__()
		emb_dims = args.emb_dims
		self.emb_dims = emb_dims
		self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
								nn.BatchNorm1d(emb_dims // 2),
								nn.ReLU(),
								nn.Linear(emb_dims // 2, emb_dims // 4),
								nn.BatchNorm1d(emb_dims // 4),
								nn.ReLU(),
								nn.Linear(emb_dims // 4, emb_dims // 8),
								nn.BatchNorm1d(emb_dims // 8),
								nn.ReLU())
		self.proj_rot = nn.Linear(emb_dims // 8, 4)
		self.proj_trans = nn.Linear(emb_dims // 8, 3)

	def forward(self, *input):
		src_embedding = input[0]
		tgt_embedding = input[1]
		embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
		embedding = self.nn(embedding.max(dim=-1)[0])
		rotation = self.proj_rot(embedding)
		rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
		translation = self.proj_trans(embedding)
		return quat2mat(rotation), translation


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, *input):
		return input


class PCRNet(nn.Module):
	def __init__(self, args):
		super(PCRNet, self).__init__()
		self.emb_dims = args.emb_dims
		self.cycle = args.cycle
		self.iterations = args.iterations
		if args.emb_nn == 'pointnet':
			self.emb_nn = PointNet(emb_dims=self.emb_dims)
		elif args.emb_nn == 'dgcnn':
			self.emb_nn = DGCNN(emb_dims=self.emb_dims)
		else:
			raise Exception('Not implemented')

		if args.pointer == 'identity':
			self.pointer = Identity()
		else:
			raise Exception("Not implemented")

		if args.head == 'mlp':
			self.head = MLPHead(args=args)
		else:
			raise Exception('Not implemented')

	def forward(self, *input):
		src = input[0]
		tgt = input[1]
		src_embedding = self.emb_nn(src)
		tgt_embedding = self.emb_nn(tgt)

		# src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)
		# src_embedding = src_embedding + src_embedding_p
		# tgt_embedding = tgt_embedding + tgt_embedding_p

		if self.iterations > 1:
			rotation_ab_temp, translation_ab_temp = self.head(src_embedding, tgt_embedding, src, tgt)
			rotation_ab, translation_ab = rotation_ab_temp.clone(), translation_ab_temp.clone()
			
			for itr in range(self.iterations-1):
				src = transform_point_cloud(src, rotation_ab_temp, translation_ab_temp)
				src_embedding = self.emb_nn(src)

				rotation_ab_temp, translation_ab_temp = self.head(src_embedding, tgt_embedding, src, tgt)
				rotation_ab, translation_ab = combine_transformations(rotation_ab_temp, translation_ab_temp, rotation_ab, translation_ab)
		else:
			rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt)

		if self.cycle:
			rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src)
		else:
			rotation_ba = rotation_ab.transpose(2, 1).contiguous()
			translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
		return rotation_ab, translation_ab, rotation_ba, translation_ba
