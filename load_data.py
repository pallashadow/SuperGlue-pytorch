import numpy as np
import torch
import os
import math
import datetime
import time
from torch.utils.data import Dataset
import pickle


class SparseDatasetOffline(Dataset):
	def __init__(self, train_path):
		self.files = [train_path + f for f in os.listdir(train_path)]
		
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, idx):
		with open(self.files[idx], 'rb') as f:
			dict1 = pickle.load(f)
		return dict1

class SparseDatasetOnline(Dataset):
	"""Sparse correspondences dataset."""

	def __init__(self, train_path, hand_path, dataBuilder):
		#self.files = [train_path + f for f in os.listdir(train_path)]
		self.files = [root +"/"+ name for root, dirs, files in os.walk(train_path, topdown=False) for name in files if name[-4:]==".jpg"]
		self.hand_files = [hand_path + name for name in os.listdir(hand_path)]
		self.dataBuilder = dataBuilder

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		dict1 = self.dataBuilder.build([idx], self.files, self.hand_files, saveFlag=True, debug=0)[0]
		return dict1

