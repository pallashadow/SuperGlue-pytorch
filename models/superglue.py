# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
from copy import deepcopy
from pathlib import Path
from torch import nn
import numpy as np
import time


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpts, image_shape):
	size = image_shape.flip(1).type(torch.Tensor).to(kpts.device) # shape=(b,2) ; w, h
	center = size / 2
	scaling = (size.max(1, keepdim=True).values * 0.7).to(kpts.device)
	return (kpts - center[:, None, :]) / scaling[:, None, :]

def normalize_keypoints0(kpts, image_shape):
	""" Normalize keypoints locations based on image image_shape"""
	#_, _, h, w = image_shape
	h, w = image_shape[0]
	one = kpts.new_tensor(1)
	size = torch.stack([one*w, one*h])[None]
	center = size / 2
	scaling = size.max(1, keepdim=True).values * 0.7
	return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
	""" Joint encoding of visual appearance and location using MLPs"""
	def __init__(self, feature_dim, layers):
		super().__init__()
		self.encoder = MLP([3] + layers + [feature_dim])
		nn.init.constant_(self.encoder[-1].bias, 0.0)

	def forward(self, kpts, scores):
		inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
		return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class SuperGlue(nn.Module):
	"""SuperGlue feature matching middle-end
	Given two sets of keypoints and locations, we determine the
	correspondences by:
	  1. Keypoint Encoding (normalization + visual feature and location fusion)
	  2. Graph Neural Network with multiple self and cross-attention layers
	  3. Final projection layer
	  4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
	  5. Thresholding matrix based on mutual exclusivity and a match_threshold
	The correspondence ids use -1 to indicate non-matching points.
	Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
	Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
	Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
	"""
	default_config = {
		'descriptor_dim': 128,
		'pretrained': '',
		#'pretrained': './utils/model/superglue/superglue_0.1.pth',
		'keypoint_encoder': [32, 64, 128],
		'GNN_layers': ['self', 'cross'] * 2,
		'sinkhorn_iterations': 5,
		'match_threshold': 0.2,
	}

	def __init__(self, config):
		super().__init__()
		self.config = {**self.default_config, **config}

		self.kenc = KeypointEncoder(
			self.config['descriptor_dim'], self.config['keypoint_encoder'])

		self.gnn = AttentionalGNN(
			self.config['descriptor_dim'], self.config['GNN_layers'])

		self.final_proj = nn.Conv1d(
			self.config['descriptor_dim'], self.config['descriptor_dim'],
			kernel_size=1, bias=True)

		bin_score = torch.nn.Parameter(torch.tensor(1.))
		self.register_parameter('bin_score', bin_score)

		if len(self.config['pretrained'])>0:
			#assert self.config['weights'] in ['indoor', 'outdoor']
			#print(torch.load(self.config['pretrained']))
			self.load_state_dict(torch.load(self.config['pretrained']))
			print('Loaded SuperGlue model (\"{}\" weights)'.format(self.config['pretrained']))

	def forward(self, data):
		"""Run SuperGlue on a pair of keypoints and descriptors"""
		desc0, desc1 = data['descriptors0'], data['descriptors1']
		kpts0, kpts1 = data['keypoints0'], data['keypoints1']

		if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
			shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
			return {
				'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
				'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
				'matching_scores0': kpts0.new_zeros(shape0),
				'matching_scores1': kpts1.new_zeros(shape1),
				'nokp_flag': 1, 
			}

		# Keypoint normalization.
		kpts0 = normalize_keypoints(kpts0, data['shape0'])
		kpts1 = normalize_keypoints(kpts1, data['shape1'])
		#print(desc0.shape, kpts0.shape, data['scores0'].shape)

		# Keypoint MLP encoder.
		desc0 = desc0 + self.kenc(kpts0, data['scores0']) # (1, 256, 100), (1,100,2), (1,100)
		desc1 = desc1 + self.kenc(kpts1, data['scores1'])

		# Multi-layer Transformer network.
		desc0, desc1 = self.gnn(desc0, desc1)

		# Final MLP projection.
		mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

		# Compute matching descriptor distance.
		scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
		scores = scores / self.config['descriptor_dim']**.5

		# Run the optimal transport.
		scores = log_optimal_transport(
			scores, self.bin_score,
			iters=self.config['sinkhorn_iterations'])

		# Get the matches with score above "match_threshold".
		max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
		indices0, indices1 = max0.indices, max1.indices
		mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
		mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
		zero = scores.new_tensor(0)
		mscores0 = torch.where(mutual0, max0.values.exp(), zero)
		mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
		valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
		valid1 = mutual1 & valid0.gather(1, indices1)
		indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
		indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

		return {
			'matches0': indices0, # use -1 for invalid match
			'matches1': indices1, # use -1 for invalid match
			'matching_scores0': mscores0,
			'matching_scores1': mscores1,
			'scores': scores,
			'nokp_flag': 0, 
		}
	


class SuperGlueWarper(object):
	def __init__(self, config, useGPU=1, feature_dim=128):
		self.superglue = SuperGlue(config).eval()
		if useGPU:
			self.superglue = self.superglue.cuda()
		self.useGPU = useGPU
		self.feature_dim = feature_dim
		
	def match(self, pairList):
		t0 = time.time()
		N = len(pairList)
		#kp1, kp2, des1, des2, score1, score2, shape1, shape2, n1, n2 = pair
		kp1 = torch.stack([torch.Tensor(pair[0]) for pair in pairList], dim=0)
		kp2 = torch.stack([torch.Tensor(pair[1]) for pair in pairList], dim=0)
		des1 = torch.stack([torch.Tensor(pair[2]) for pair in pairList], dim=0).transpose(1,2)
		des2 = torch.stack([torch.Tensor(pair[3]) for pair in pairList], dim=0).transpose(1,2)
		score1 = torch.stack([torch.Tensor(pair[4]) for pair in pairList], dim=0)
		score2 = torch.stack([torch.Tensor(pair[5]) for pair in pairList], dim=0)
		shape1 = torch.stack([torch.Tensor(pair[6]) for pair in pairList], dim=0)
		shape2 = torch.stack([torch.Tensor(pair[7]) for pair in pairList], dim=0)
		n1List = [pair[8] for pair in pairList] # 有效kp个数
		n2List = [pair[9] for pair in pairList]
		t1 = time.time()
		data = {'keypoints0': kp1, 
				'keypoints1': kp2, 
				'descriptors0': des1, 
				'descriptors1': des2, 
				'scores0': score1, 
				'scores1': score2, 
			   }
		if self.useGPU:
			data = {k:v.cuda() for k, v in data.items()}
		data.update({
				'shape0': shape1, 
				'shape1': shape2, 
			   })
		t2 = time.time()
		with torch.no_grad():
			pred = self.superglue(data)
			#pred.pop('nokp_flag')
			#pred = {k:v.cpu().numpy() for k,v in pred.items()}
		indices1 = pred['matches0'].cpu().numpy()
		t3 = time.time()
		#for b in range(N):
		#	n1 = n1List[b]
		#	matches = np.array([[i, indices1[i]] for i in range(n1) if indices1[b,i]!=-1])
		matchesList = [np.array([[i, indices1[b,i]] for i in range(n1List[b]) if indices1[b,i]!=-1]) for b in range(N)]
		t4 = time.time()
		print("unpack:{:.3f}, gpu:{:.3f}, NN:{:.3f}, pack:{:.3f}".format(t1-t0, t2-t1, t3-t2, t4-t3))
		return matchesList

def debugShow(debug_inputImage, kp1, debug_refImage, kp2, homoMatches):
	assert(debug_inputImage is not None)
	kp1_ = [cv2.KeyPoint(p[0],p[1],0) for p in kp1]
	kp2_ = [cv2.KeyPoint(p[0],p[1],0) for p in kp2]
	for p in kp1:
		cv2.circle(debug_inputImage, (int(p[0]), int(p[1])), 1,(0,0,255),1)
	for p in kp2:
		cv2.circle(debug_refImage, (int(p[0]), int(p[1])), 1,(0,0,255),1)
	cv2.putText(debug_refImage, str(len(homoMatches)), (50, 50), 2, 1, (255, 0, 255), 1, 1)
	cvHomoMatches = [cv2.DMatch(i[0],i[1],0) for i in homoMatches]
	debug_image=cv2.drawMatches(debug_inputImage,kp1_,debug_refImage,kp2_,cvHomoMatches,None, matchColor=None, singlePointColor=(255, 255, 255), flags=2)
	cv2.imshow("debug_superpoint", debug_image)
	cv2.waitKey(0)
	
def inference_time_test():
	dim = 128
	k = 300
	h,w = 240,320
	config = {
		#'descriptor_dim': 256,
		'descriptor_dim': dim,
		#'pretrained': './utils/model/superglue/superglue_0.4.pth',
		#'keypoint_encoder': [32, 64, 128, 256],
		'keypoint_encoder': [32, 64, 128],
		'GNN_layers': ['self', 'cross'] * 2,
		'sinkhorn_iterations': 5,
		'match_threshold': 0.2,
	}
	loopTimes = 15
	useGPU=0
	
	b = 1
	if 1:
		superglue = SuperGlueWarper(config, useGPU=useGPU)
		if 0:
			import cv2
			from utils.model.superpoint.superpoint_init import spObj
			from utils.model.superpoint.superpoint_utils import spPostProcess
			#im1 = cv2.imread('/home/pallas/PROJECTS/cv-book-image-recognition/dataset/dataset/test/finger_test_dataset/CM_Textbook_CN_finger_398_20200707/人教部编版语文二年级上册/19/606/picbook_35041c331dca4b1c9ad5c7ac6b6402e6_AIvERmBkEiqyzWNm_112416052983410211_73HOc5yFKjIQj2s6Zjb3pLd7Xd7LWs04_1594019729867.jpg')
			im1 = cv2.imread('/home/pallas/PROJECTS/cv-book-image-recognition/dataset/dataset/test/finger_test_dataset/CM_Textbook_CN_finger_398_20200707/人教部编版语文二年级上册/27/840/picbook_35041c331dca4b1c9ad5c7ac6b6402e6_AIvERmBkEiqyzWNm_112416062145310114_vgb3fBCEPe9nQLW0fAAAtc61d6zoFTmO_1594019821486.jpg')
			im2 = cv2.imread('/home/pallas/PROJECTS/cv-book-image-recognition/dataset/dataset/test/finger_test_dataset/CM_Textbook_CN_finger_398_20200707/人教部编版语文二年级上册/19/606/picbook_35041c331dca4b1c9ad5c7ac6b6402e6_AIvERmBkEiqyzWNm_122415960233110124_BLWZymOngC8FKZzFKUvgEIQgCEI66LWs_1594018803318.jpg')
			im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
			im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
			spFeat_batch = spObj.run_batch(np.array([im1_gray, im2_gray]))
			kp1, des1, _ = spPostProcess(spFeat_batch[0], numPts=300, conf_thresh=0.010, )
			kp2, des2, _ = spPostProcess(spFeat_batch[1], numPts=300, conf_thresh=0.010, )
			shape1 = im1.shape[:2]
			shape2 = im2.shape[:2]
		else:
			kp1 = np.random.randint(0,240,(k,2))
			kp1 = np.hstack([kp1, np.random.random((k,1))])
			des1 = np.random.random((k,dim))
			shape1 = (w, h)
			kp2 = kp1; des2 = des1; shape2 = shape1; 
		n1 = len(kp1); n2 = len(kp2)
		pair = (kp1[:,:2], kp2[:,:2], des1, des2, kp1[:,2], kp2[:,2], shape1, shape2, n1, n2)
		pairList = [pair]*b
		for i in range(loopTimes):
			t1 = time.time()
			matchesList = superglue.match(pairList)
			#print(matchesList[0])
			#debugShow(im1, kp1, im2, kp2, matchesList[0])
			t2 = time.time()
			print(t2-t1)
	else:
		superglue = SuperGlue({}).eval()
		kp1 = torch.rand(b,k,2)
		des1 = torch.rand(b,dim,k)
		score1 = torch.rand(b,k)
		shape1 = torch.LongTensor([[w,h] for i in range(b)])
		kp2 = kp1; des2 = des1; shape2 = shape1; score2=score1
		data = {'keypoints0': kp1, 
				'keypoints1': kp2, 
				'descriptors0': des1, 
				'descriptors1': des2, 
				'scores0': score1, 
				'scores1': score2, 
				'shape0': shape1, 
				'shape1': shape2, 
			   }
		if useGPU:
			superglue = superglue.cuda()
			data = {k:v.cuda() for k,v in data.items()}
		for i in range(loopTimes):
			t1 = time.time()
			with torch.no_grad():
				pred = superglue(data)
				pred.pop('nokp_flag')
				pred = {k:v.cpu().numpy() for k,v in pred.items()}
			t2 = time.time()
			print(t2-t1)
	
if __name__ == "__main__":
	inference_time_test()
	