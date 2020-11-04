import open3d as o3d
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import transforms3d

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))

from pcrnet.models import PointNet, iPCRNet
from pcrnet.losses import ChamferDistanceLoss
from pcrnet.data_utils import RegistrationData, ModelNet40Data


def display_open3d(template, source, transformed_source):
	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	transformed_source_ = o3d.geometry.PointCloud()
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	transformed_source_.paint_uniform_color([0, 0, 1])
	o3d.visualization.draw_geometries([template_, source_, transformed_source_])

# Find error metrics.
def find_errors(igt_R, pred_R, igt_t, pred_t):
	# igt_R:				Rotation matrix [3, 3] (source = igt_R * template)
	# pred_R: 			Registration algorithm's rotation matrix [3, 3] (template = pred_R * source)
	# igt_t:				translation vector [1, 3] (source = template + igt_t)
	# pred_t: 			Registration algorithm's translation matrix [1, 3] (template = source + pred_t)

	# Euler distance between ground truth translation and predicted translation.
	igt_t = -np.matmul(igt_R.T, igt_t.T).T			# gt translation vector (source -> template)
	translation_error = np.sqrt(np.sum(np.square(igt_t - pred_t)))

	# Convert matrix remains to axis angle representation and report the angle as rotation error.
	error_mat = np.dot(igt_R, pred_R)							# matrix remains [3, 3]
	_, angle = transforms3d.axangles.mat2axangle(error_mat)
	return translation_error, abs(angle*(180/np.pi))

def compute_accuracy(igt_R, pred_R, igt_t, pred_t):
	errors_temp = []
	for igt_R_i, pred_R_i, igt_t_i, pred_t_i in zip(igt_R, pred_R, igt_t, pred_t):
		errors_temp.append(find_errors(igt_R_i, pred_R_i, igt_t_i, pred_t_i))
	return np.mean(errors_temp, axis=0)

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	errors = []

	for i, data in enumerate(tqdm(test_loader)):
		if i > 2: break
		template, source, igt, igt_R, igt_t = data

		template = template.to(device)
		source = source.to(device)
		igt = igt.to(device)

		source_original = source.clone()
		template_original = template.clone()
		igt_t = igt_t - torch.mean(source, dim=1).unsqueeze(1)
		source = source - torch.mean(source, dim=1, keepdim=True)
		template = template - torch.mean(template, dim=1, keepdim=True)

		output = model(template, source)
		est_R = output['est_R']
		est_t = output['est_t']

		errors.append(compute_accuracy(igt_R.detach().cpu().numpy(), est_R.detach().cpu().numpy(),
									   igt_t.detach().cpu().numpy(), est_t.detach().cpu().numpy()))

		transformed_source = torch.bmm(est_R, source.permute(0, 2, 1)).permute(0,2,1) + est_t
		display_open3d(template.detach().cpu().numpy()[0], source_original.detach().cpu().numpy()[0], transformed_source.detach().cpu().numpy()[0])

		loss_val = ChamferDistanceLoss()(template, output['transformed_source'])

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	errors = np.mean(np.array(errors), axis=0)
	return test_loss, errors[0], errors[1]

def test(args, model, test_loader):
	test_loss, translation_error, rotation_error = test_one_epoch(args.device, model, test_loader)
	print("Test Loss: {}, Rotation Error: {} & Translation Error: {}".format(test_loss, rotation_error, translation_error))

def options():
	parser = argparse.ArgumentParser(description='Point Cloud Registration')
	parser.add_argument('--exp_name', type=str, default='exp_ipcrnet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PointNet
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')

	# settings for on training
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=20, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--pretrained', default='pcrnet/pretrained/exp_ipcrnet/models/best_model.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()

	testset = RegistrationData('PCRNet', ModelNet40Data(train=False), is_testing=True)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	ptnet = PointNet(emb_dims=args.emb_dims)
	model = iPCRNet(feature_model=ptnet)
	model = model.to(args.device)

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	test(args, model, test_loader)

if __name__ == '__main__':
	main()