
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import SparseDatasetOnline, SparseDatasetOffline
import os
import torch.multiprocessing
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import cv2

# from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

#from models.superpoint import SuperPoint
from models.superglue import SuperGlue
#from models.matchingForTraining import MatchingForTraining
from models.superglueLoss import superglueLoss
from dataset.data_builder import DataBuilder

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.multiprocessing.set_start_method("spawn")
# torch.cuda.set_device(0)

# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass

def configParser():
	parser = argparse.ArgumentParser(
		description='Image pair matching and pose evaluation with SuperGlue',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument(
		'--viz', action='store_true',
		help='Visualize the matches and dump the plots')
	parser.add_argument(
		'--eval', action='store_true',
		help='Perform the evaluation'
				' (requires ground truth pose and intrinsics)')

	parser.add_argument(
		'--keypoint_threshold', type=float, 
		#default=0.005,
		default=0.010,
		help='SuperPoint keypoint detector confidence threshold')
	parser.add_argument(
		'--nms_radius', type=int, default=4,
		help='SuperPoint Non Maximum Suppression (NMS) radius'
		' (Must be positive)')
	parser.add_argument(
		'--resize_float', action='store_true',
		help='Resize the image after casting uint8 to float')
	parser.add_argument(
		'--cache', action='store_true',
		help='Skip the pair if output .npz files are already found')
	parser.add_argument(
		'--show_keypoints', action='store_true',
		help='Plot the keypoints in addition to the matches')
	parser.add_argument(
		'--fast_viz', action='store_true',
		help='Use faster image visualization based on OpenCV instead of Matplotlib')
	parser.add_argument(
		'--viz_extension', type=str, default='png', choices=['png', 'pdf'],
		help='Visualization file extension. Use pdf for highest-quality.')
	parser.add_argument(
		'--opencv_display', action='store_true',
		help='Visualize via OpenCV before saving output images')
	parser.add_argument(
		'--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
		help='Path to the list of image pairs for evaluation')
	parser.add_argument(
		'--shuffle', action='store_true',
		help='Shuffle ordering of pairs before processing')
	parser.add_argument(
		'--max_length', type=int, default=-1,
		help='Maximum number of pairs to evaluate')
	parser.add_argument(
		'--learning_rate', type=int, default=0.0001,
		help='Learning rate')
	parser.add_argument(
		'--eval_output_dir', type=str, default='dump_match_pairs/',
		help='Path to the directory in which the .npz results and optional,'
				'visualizations are written')
	
	
	
	parser.add_argument(
		'--sinkhorn_iterations', type=int, default=20,
		help='Number of Sinkhorn iterations performed by SuperGlue')
	parser.add_argument(
		'--match_threshold', type=float, default=0.2,
		help='SuperGlue match threshold')

	parser.add_argument(
		'--resize', type=int, nargs='+', default=[320, 240],
		help='Resize the input image before running inference. If two numbers, '
				'resize to the exact dimensions, if one number, resize the max '
				'dimension, if -1, do not resize')
	parser.add_argument(
		'--max_keypoints', type=int, default=300,
		help='Maximum number of keypoints detected by Superpoint'
				' (\'-1\' keeps all keypoints)')
	parser.add_argument(
		'--feature_dim', type=int, default=256, help='superpoint feature dim')
	parser.add_argument(
		'--batch_size', type=int, default=1,
		help='batch_size')
	parser.add_argument('--train_path', type=str, 
		default = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/data/CHILD_BOOK/dataset/dataset/ref/card/", 
		#default = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/data/CHILD_BOOK/dataset/dataset/ref/inner/", 
		#default = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/data/CHILD_BOOK/dataset/dataset/ref/title/", 
		help='Path to the directory of training imgs.')
	parser.add_argument('--hand_path', type=str, 
		default = "", 
		#default = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/PROJECTS/SuperGlue-pytorch/turingAug/data/image_hand/", 
					   )
	parser.add_argument('--superpoint_weight', type=str, 
						#default="./utils/model/superpoint/superpoint128_ft07.pth", 
						default="./models/weights/superpoint_v1.pth"
					   )
	parser.add_argument('--pretrained', type=str, 
						default="", 
						#default="./checkpoints/model_epoch_0.pth", 
					   )

	parser.add_argument('--dataset_online', type=int, default=0)
	parser.add_argument('--dataset_offline_rebuild', type=int, default=1)
	parser.add_argument(
		'--epoch', type=int, default=20,
		help='Number of epoches')
	
	
	parser.add_argument('--tensorboardLabel', type=str, 
						#default='cardTest1',
						default='innerNeg1', 
					   )
	opt = parser.parse_args()
	
	config = {
		'superpoint': {
			'nms_radius': opt.nms_radius,
			'keypoint_threshold': opt.keypoint_threshold,
			'max_keypoints': opt.max_keypoints, 
			'feature_dim': opt.feature_dim, 
			'weights_path': opt.superpoint_weight, # feature_dim=128
		},
		'superglue': {
			'keypoint_encoder': [32, 64, 128, 256],
			#'keypoint_encoder': [32, 64, 128],
			'GNN_layers': ['self', 'cross'] * 9,
			'descriptor_dim': opt.feature_dim,
			#'pretrained': './weights/superglue_indoor.pth',
			'pretrained': opt.pretrained,
			'sinkhorn_iterations': opt.sinkhorn_iterations,
			'match_threshold': opt.match_threshold,
		}
	}
	return opt, config


if __name__ == '__main__':
	opt, config = configParser()
	print(opt)

	assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
	assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
	assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
	assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

	# store viz results
	eval_output_dir = Path(opt.eval_output_dir)
	eval_output_dir.mkdir(exist_ok=True, parents=True)
	print('Will write visualization images to',
		'directory \"{}\"'.format(eval_output_dir))
	
	if opt.dataset_online:
		dataBuilder = DataBuilder(config['superpoint'], './dataset/warped/', './dataset/sp/', numProcess=1)
		train_set = SparseDatasetOnline(opt.train_path, opt.hand_path, dataBuilder)
	else:
		if opt.dataset_offline_rebuild:
			dataBuilder = DataBuilder(config['superpoint'], './dataset/warped/', './dataset/sp/', numProcess=7)
			dataBuilder.buildAll(opt.train_path, opt.hand_path, batchSizeMax=64, saveFlag=1, debug=0)
		train_set = SparseDatasetOffline('./dataset/sp/')
	train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

	# superpoint = SuperPoint(config.get('superpoint', {}))
	superglue = SuperGlue(config.get('superglue', {}))
	if torch.cuda.is_available():
		# superpoint.cuda()
		superglue.cuda()
	else:
		print("### CUDA not available ###")
	optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
	N = train_loader.dataset.__len__() // opt.batch_size

	writer = SummaryWriter("./logs/"+opt.tensorboardLabel)
	mean_loss = []
	cudaKey = set(['keypoints0', 'keypoints1', 'descriptors0', 'descriptors1', 'scores0', 'scores1'])
	for epoch in range(0, opt.epoch):
		epoch_loss = 0
		superglue.train()
		# train_loader = tqdm(train_loader)
		t_end = time.time()
		for i, data in enumerate(train_loader):
			t1 = time.time()
			for k in cudaKey:
				data[k] = data[k].cuda()
			pred = superglue(data)
			t2 = time.time()
			if pred['nokp_flag'] == 1: # image has no keypoint
				continue
			Loss = superglueLoss(pred, data)
			t3 = time.time()
			superglue.zero_grad()
			#Loss = pred['loss']
			epoch_loss += Loss.item()
			mean_loss.append(Loss)
			Loss.backward()
			optimizer.step()
			t4 = time.time()
			writer.add_scalar('loss_step', Loss.item(), epoch*N+i)

			if (i) % 1 == 0:
				current_loss = torch.mean(torch.stack(mean_loss)).item()
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					.format(epoch, opt.epoch, i+1, len(train_loader), current_loss))
				print('data time: {:.2f}, forward time: {:.2f}, loss time: {:.2f}, back time: {:.2f}'.format(t1-t_end, t2-t1,t3-t2,t4-t3))
				mean_loss = []
				
			if (i+1) % 1000 == 0: # eval, Visualize the training set matches.
				#superglue.eval()
				#image0, image1 = data['image0'].cpu().numpy()[0,0]*255., data['image1'].cpu().numpy()[0,0]*255.
				image0 = cv2.imread(data['image_file'][0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
				image1 = cv2.imread(data['warped_file'][0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
				kpts0, kpts1 = data['keypoints0'].cpu().numpy()[0], data['keypoints1'].cpu().numpy()[0]
				matches, conf = pred['matches0'].cpu().detach().numpy()[0], pred['matching_scores0'].cpu().detach().numpy()[0]
				image0 = read_image_modified(image0, opt.resize, opt.resize_float)
				image1 = read_image_modified(image1, opt.resize, opt.resize_float)
				valid = matches > -1
				mkpts0 = kpts0[valid]
				mkpts1 = kpts1[matches[valid]]
				mconf = conf[valid]
				viz_path = eval_output_dir / '{}_{}_matches.{}'.format(str(epoch), str(i), opt.viz_extension)
				color = cm.jet(mconf)
				stem = data['image_file'][0]
				text = []
				make_matching_plot(
					image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
					text, viz_path, stem, stem, opt.show_keypoints,
					opt.fast_viz, opt.opencv_display, 'Matches')
			
			if (i+1) % 2000 == 0:
				model_out_path = "./checkpoints/model_epoch_{}_{}.pth".format(epoch, i)
				torch.save(superglue.state_dict(), model_out_path)
				print ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}' 
					.format(epoch, opt.epoch, i+1, len(train_loader), model_out_path)) 
			
			t_end = time.time()

		epoch_loss /= len(train_loader)
		model_out_path = "./checkpoints/model_epoch_{}.pth".format(epoch)
		torch.save(superglue.state_dict(), model_out_path)
		print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
			.format(epoch, opt.epoch, epoch_loss, model_out_path))
        

