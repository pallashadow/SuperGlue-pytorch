import numpy as np
import torch
import os
import cv2
from scipy.spatial.distance import cdist
from models.superpoint import SuperPoint # official implement
#from utils.model.superpoint import superpoint, superpoint_utils
import dataset.render as render
import pickle
import multiprocessing

def mat2map(M, W, H):
	'''build reverse map from linear projection matrix'''
	M_inv = np.linalg.inv(M)
	xx, yy = np.meshgrid(np.arange(W),np.arange(H))
	map1 = np.stack([xx, yy, np.ones([H,W])], axis=2).astype(np.float32)
	mapW = map1.reshape((H*W,3)).dot(M_inv.T).reshape((H,W,3)).astype(np.float32) # multiply M_inv
	mapW = mapW[:,:,:2] / np.stack([mapW[:,:,2]]*2, axis=2) # remove homogeneous
	return mapW

def util_WarpMap(h,w, maxPixels=20.0, steps=2):
	xx, yy = np.meshgrid(np.arange(w),np.arange(h))
	for step in range(steps):
		k = np.random.random()*10+3 # 3-13
		r1 = np.random.random(2)
		pixels = (r1[0]*2-1)*maxPixels # max shift_y pixels on curve

		k2 = (1-1/np.sqrt(k))*np.cos(np.arctan(1/np.sqrt(k))) # normalized value on shift_y

		x = xx/w*k if r1[1]>0.5 else (w-xx)/w*k
		sign = 1 if r1[1]>0.5 else -1

		shift_x = np.sqrt(1/x+1)*x+1/2*np.log((2*np.sqrt(1/x+1)+2)*x+1) # distance integrate of model y
		shift_x = shift_x*sign
		shift_y = (np.sqrt(x) - x / np.sqrt(k))/k2*pixels # page curve model y
		xx = xx - shift_x
		yy = yy - shift_y
	map1 = np.stack([xx, yy], axis=2).astype(np.float32)
	return map1

def remapPropagate(map1, map2):
	map3 = np.empty(map1.shape, dtype=np.float32)
	map3[:,:,0] = cv2.remap(map1[:,:,0], map2, None, cv2.INTER_LINEAR, None, borderValue=np.nan)
	map3[:,:,1] = cv2.remap(map1[:,:,1], map2, None, cv2.INTER_LINEAR, None, borderValue=np.nan)
	return map3

def simpleWarp(image, W, H):
	corners = np.array([[0, 0], [0, H], [W, 0], [W, H]], dtype=np.float32)
	warp = np.random.randint(-50, 50, size=(4, 2)).astype(np.float32)
	M = cv2.getPerspectiveTransform(corners, corners + warp)
	mapW = mat2map(M, W, H)
	if 0: # warpPerspective using mapping interface
		warped = np.zeros((H, W), dtype=np.uint8)
		cv2.remap(image, mapW, None, cv2.INTER_LINEAR, warped, borderValue=128) 
	else:
		warped = cv2.warpPerspective(image, M, (W, H), borderValue=128)
	return warped, M, mapW

def pictureBookWarpAndRender(image, W, H, scale=1.0, borderValue=128):
	corners = np.array([[0, 0], [0, H], [W, 0], [W, H]], dtype=np.float32)
	r1 = np.random.random()*2-1
	r2 = np.random.randint(-50, 50, size=(4, 2)).astype(np.float32)
	
	center = np.array([[W/2,H/2]]*4)
	target = (corners+r2-center)*scale + center + r1*0.7*center #(1-scale)*r1*center

	M = cv2.getPerspectiveTransform(corners, target.astype(np.float32))
	mapW = mat2map(M, W, H)
	map1 = util_WarpMap(H, W, maxPixels=30.0, steps=2)
	mapW = remapPropagate(map1, mapW)
	warped = np.zeros((H, W), dtype=np.uint8)

	imgB = image
	r1 = np.random.random(2)*2-1 # [-1,1]
	imgB = render.adjustBrightAndContrast(imgB, 1.0+r1[0]*0.3, 1.0+r1[0]*0.3)
	imgB = render.randomMotionBlur(imgB, distanceMax=5,probability=0.25)
	imgB = render.randomGuassianBlur(imgB,probability=0.7)
	imgB = render.randomLightStain(imgB, maxSize = 30, up=3, down=1, probability=1.0, center=None)
	cv2.remap(imgB, mapW, None, cv2.INTER_LINEAR, warped, borderValue=borderValue) 
	mask = np.zeros((H, W))
	cv2.remap(np.ones((H, W)), mapW, None, cv2.INTER_NEAREST, mask, borderValue=0)
	return warped, M, mapW, mask.astype(np.bool)

def sampleHand(warped, hand_file, W, H):
	if hand_file is None:
		return warped, None
	hand = cv2.imread(hand_file, cv2.IMREAD_GRAYSCALE)
	if hand is None:
		return warped, None
	hand = np.flipud(cv2.resize(hand, (W, H)))
	r1 = np.random.random()*0.3+0.3
	hand, M, mapW, mask = pictureBookWarpAndRender(hand, W, H, scale=r1, borderValue=0)
	handMask = ~(hand<30)
	warped[handMask] = hand[handMask]
	handMask = cv2.dilate(handMask.astype(np.float32), (3,3)).astype(np.bool)
	return warped, handMask

def mp_warp(task):
	idx, image_file, image_file_neg, back_file, hand_file, isNegSample, W, H, save_path, saveFlag = task
	image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) 
	image = cv2.resize(image, (W, H))
	imB = cv2.imread(back_file, cv2.IMREAD_GRAYSCALE) 
	imB = np.flipud(cv2.resize(imB, (W, H)))
	if isNegSample:
		warped = cv2.imread(image_file_neg, cv2.IMREAD_GRAYSCALE)
		warped = cv2.resize(warped, (W, H))
	else:
		warped = image.copy()
	#warped, M, mapW = simpleWarp(image, W, H)
	r1 = np.random.random()*0.4+0.8
	warped, M, mapW, mask = pictureBookWarpAndRender(warped, W, H, scale=r1)
	warped[~mask] = imB[~mask]
	warped, handMask = sampleHand(warped, hand_file, W, H)
	if saveFlag:
		warped_file = save_path+str(idx)+".jpg"
		cv2.imwrite(warped_file, warped)
	image_f = image.astype(np.float32)/255
	warped_f = warped.astype(np.float32)/255
	return image_f, warped_f, mapW, handMask, isNegSample

class DataBuilder(object):
	def __init__(self, config, save_path_warped, save_path_sp, numProcess=1):
		#self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.max_keypoints)
		#self.superpointFrontend = superpoint.SuperPointFrontend(weights_path = config['weights_path'], cuda=1, device_id=0)
		self.superpoint = SuperPoint(config).cuda().eval()
		self.superpoint.load_state_dict(torch.load(config['weights_path']))
		self.feature_dim = config['feature_dim']
		self.max_keypoints = config['max_keypoints']
		self.keypoint_threshold = config['keypoint_threshold']
		self.nms_radius = config['nms_radius']
		self.save_path_warped = save_path_warped
		self.save_path_sp = save_path_sp
		self.pool = multiprocessing.Pool(numProcess)
		
	def extractSP(self, imageList):
		N = len(imageList)
		data = np.stack(imageList, axis=0)
		data = torch.from_numpy(data[:, np.newaxis,:,:])
		data = data.cuda()
		with torch.no_grad():
			pred = self.superpoint({"image":data})
		spPackList = []
		for i in range(N):
			kp1_np = pred['keypoints'][i].cpu().numpy()
			descs1 = pred['descriptors'][i].cpu().numpy()
			scores1_np = pred['scores'][i].cpu().numpy()
			n1 = len(kp1_np)
			spPackList.append((n1, kp1_np, descs1, scores1_np))
		return spPackList
		
	def match(self, sp_pack1, sp_pack2, mapW, isNegSample, image, warped, handMask, debug=0):
		nf = self.max_keypoints
		n1, kp1_np, descs1, scores1_np = sp_pack1
		n2, kp2_np, descs2, scores2_np = sp_pack2
		if isNegSample or n1==0 or n2==0:
			numMatch = 0
			numMiss = nf
			MN2 = np.vstack([np.arange(nf), nf * np.ones(nf, dtype=np.int64)]) # excluded kp1 --> last index in matrix
			MN3 = np.vstack([nf * np.ones(nf, dtype=np.int64), np.arange(nf)])
			all_matches = np.hstack([MN2, MN3]).T
			if debug:
				cv2.imshow('image', image)
				cv2.imshow('warped', warped)
				cv2.waitKey()
			return all_matches, numMatch, numMiss
		# reverse project by non-linear reverse map
		indices = kp2_np[:n2].astype(np.int32).T
		kp2_projected = mapW[indices[1], indices[0]] 
		kp2_projected = np.vstack([kp2_projected, np.zeros((nf-n2,2))])
		dists = cdist(kp1_np[:n1], kp2_projected[:n2])
		dists[np.isnan(dists)]=np.inf

		min1 = np.argmin(dists, axis=0) # kp1 which is closest from range(len(kp2))
		min2 = np.argmin(dists, axis=1) # kp2 which is closest from range(len(kp1))
		min2v = np.min(dists, axis=0) # closest distance of range(len(kp2))
		
		kp2_mutual_close = min2[min1] == np.arange(n2)
		if handMask is not None:
			kp2_blocked = ~handMask[indices[1], indices[0]] # kp2, not covered by hand
			kp2_mutual_close = np.logical_and(kp2_mutual_close, kp2_blocked)
			
		matches2 = np.where(np.logical_and(kp2_mutual_close, min2v < 3))[0] # kp2
		matches1 = min1[matches2]
		missing1 = np.setdiff1d(np.arange(nf), matches1) # kp1 which are excluded
		missing2 = np.setdiff1d(np.arange(nf), matches2) # kp2 which are excluded

		if debug: # visualize
			matches_dmatch = []
			if not isNegSample:
				for i, idx2 in enumerate(matches2):
					idx1 = matches1[i]
					dmatch = cv2.DMatch(min1[idx2], idx2, 0.0)
					matches_dmatch.append(dmatch)
			kp1 = [cv2.KeyPoint(p[0], p[1], 0) for p in kp1_np]
			#kp1 = [cv2.KeyPoint(p[0], p[1], 0) for p in kp2_projected]
			kp2 = [cv2.KeyPoint(p[0], p[1], 0) for p in kp2_np]
			out = cv2.drawMatches((image*255).astype(np.uint8), kp1, (warped*255).astype(np.uint8), kp2, matches_dmatch, None)
			cv2.imshow('a', out)
			cv2.imshow('image', image)
			cv2.imshow('warped', warped)
			cv2.waitKey()

		numMatch = len(matches1)
		numMiss = nf-numMatch
		lossWeightMiss = 0.2
		MN = np.vstack([matches1, matches2])
		MN2 = np.vstack([missing1, nf * np.ones(numMiss, dtype=np.int64)]) # excluded kp1 --> last index in matrix
		MN3 = np.vstack([nf * np.ones(numMiss, dtype=np.int64), missing2])
		MN4 = np.zeros([2, numMatch], dtype=np.int64) # zero-pad for batch training
		all_matches = np.hstack([MN, MN2, MN3, MN4]).T
		return all_matches, numMatch, numMiss
	
	
	def build(self, idxList, fileNameList, handFileList, saveFlag=False, debug=1):
		W, H = 320, 240
		batchSize = len(idxList)
		
		if handFileList is not None:
			handFiles = [handFileList[i] for i in np.random.randint(0,len(handFileList), (batchSize))]
		else:
			handFiles = [None]*batchSize
		# even idx as negative sample, force zero match
		taskList = [(idx, fileNameList[idx], fileNameList[idx-1], fileNameList[idx-2], handFiles[i], 
					 idx%2==0, W, H, self.save_path_warped, saveFlag) for i, idx in enumerate(idxList)]
		resultList = self.pool.map(mp_warp, taskList)
		
		imageList = []
		mapWList = []
		negList = []
		handMaskList = []
		for image_f, warped_f, mapW, handMask, isNegSample in resultList:
			imageList += [image_f, warped_f]
			mapWList.append(mapW)
			handMaskList.append(handMask)
			negList.append(isNegSample)
			
		spPackList = self.extractSP(imageList)
		
		dict1List = []
		for i, idx in enumerate(idxList):
			spPack1 = spPackList[i*2]
			spPack2 = spPackList[i*2+1]
			mapW = mapWList[i]
			isNegSample = negList[i]
			handMask = handMaskList[i]
			
			n1, kp1_np, descs1, scores1_np = spPack1
			n2, kp2_np, descs2, scores2_np = spPack2

			all_matches, numMatch, numMiss = self.match(spPack1, spPack2, mapW, isNegSample, 
														imageList[i*2], imageList[i*2+1], handMask, 
														debug=debug)

			image_file = fileNameList[idx]
			warped_file = self.save_path_warped+str(idx)+".jpg"
			dict1 = {
				'keypoints0': torch.Tensor(kp1_np), 
				'keypoints1': torch.Tensor(kp2_np), 
				'descriptors0': torch.Tensor(descs1), 
				'descriptors1': torch.Tensor(descs2), 
				'scores0': torch.Tensor(scores1_np), 
				'scores1': torch.Tensor(scores2_np), 

				'image_file': image_file, 
				'warped_file': warped_file, 
				'shape0': torch.Tensor([H,W]), 
				'shape1': torch.Tensor([H,W]), 
				'all_matches': all_matches,
				'num_match_list': torch.Tensor([numMatch+2*numMiss])
				 }
			if saveFlag:
				with open(self.save_path_sp+str(idx)+'.pkl', 'wb') as f:
					pickle.dump(dict1, f)
			dict1List.append(dict1)
		return dict1List
	
	def buildAll(self, train_path, hand_path="", batchSizeMax = 64, saveFlag=False, debug=1):
		#with open("/home/pallas/PROJECTS/cv-book-image-recognition/resources/alpha-book-list.txt", "r") as f:
		#	lines = f.readlines()
		#	lines = [line.strip().split("\t") for line in lines]
		#	bookCodeList = [line[2] for line in lines if "小学" in line[1]]
		#	files = [train_path + bookCode + "/" + x for bookCode in bookCodeList for x in os.listdir(train_path+bookCode)]
		files = [root +"/"+ name for root, dirs, files in os.walk(train_path, topdown=False) for name in files if name[-4:]==".jpg"]
		hands = None
		if len(hand_path)>0:
			hands = [hand_path + name for name in os.listdir(hand_path)]
		#files = files[:360]
		N = len(files)
		print("{} images in fileList".format(N))
		idxList = np.arange(N)
		i=0
		while i<N-1:
			batchSize = min(batchSizeMax, N-i)
			self.build(idxList[i:i+batchSize], files, hands, saveFlag=saveFlag, debug=debug)
			i += batchSize
			print("building training set {}/{}".format(i, N))
	
if __name__ == "__main__":
	import sys
	sys.path.append("./")
	dataBuilder = DataBuilder({"weights_path": "./models/weights/superpoint_v1.pth", 
							   "feature_dim": 128, 
							  "max_keypoints": 300, 
							  "keypoint_threshold": 0.01, 
							  "nms_radius":4}, 
							  './dataset/warped/', './dataset/sp/', numProcess=1)
	train_path = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/data/CHILD_BOOK/dataset/dataset/ref/inner/"
	hand_path = ""
	#hand_path = "/media/pallas/69c96109-1b7a-4adc-91e9-e72166a8d823/PROJECTS/SuperGlue-pytorch/turingAug/data/image_hand/"
	#dataBuilder.buildAll(train_path, hand_path, batchSizeMax=64, saveFlag=1, debug=0)
	dataBuilder.buildAll(train_path, hand_path, batchSizeMax=1, saveFlag=0, debug=1)
	