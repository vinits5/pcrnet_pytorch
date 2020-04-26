import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling


class PointNet(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc"):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(PointNet, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		self.layers = self.create_structure()

	def create_structure(self):
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 64, 1)
		self.conv3 = torch.nn.Conv1d(64, 64, 1)
		self.conv4 = torch.nn.Conv1d(64, 128, 1)
		self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
		self.relu = torch.nn.ReLU()

		layers = [self.conv1, self.relu,
				  self.conv2, self.relu, 
				  self.conv3, self.relu,
				  self.conv4, self.relu,
				  self.conv5, self.relu]
		return layers


	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		if self.input_shape == "bnc":
			num_points = input_data.shape[1]
			input_data = input_data.permute(0, 2, 1)
		else:
			num_points = input_data.shape[2]
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		output = input_data
		for idx, layer in enumerate(self.layers):
			output = layer(output)

		return output


if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	pn = PointNet(use_bn=True)
	y = pn(x)
	print("Network Architecture: ")
	print(pn)
	print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)