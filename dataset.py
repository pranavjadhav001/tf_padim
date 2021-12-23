import cv2
import glob
import numpy as np
import tensorflow as tf
import os

class MvtecDataGenerator(tf.keras.utils.Sequence):
	def __init__(self,dir_path,mask_path=None,batch_size=32):
		self.batch_size = batch_size
		self.dir_path = dir_path
		self.mask_path = mask_path
		self.image_paths,self.mask_paths = self.file_paths()
		self.indexes = np.arange(len(self.image_paths))

	def file_paths(self):
		image_paths = []
		mask_paths = []
		for path,subdirs,files in os.walk(self.dir_path):
			for name in files:
				image_paths.append(os.path.join(path,name))

		if self.mask_path is not None:
			for path,subdirs,files in os.walk(self.mask_path):
				for name in files:
					mask_paths.append(os.path.join(path,name))
			mask_paths = sorted(mask_paths)
		return sorted(image_paths),mask_paths

	def __len__(self):
		return int(np.ceil(len(self.image_paths)/self.batch_size))

	def __getitem__(self,index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		image_paths = [self.image_paths[k] for k in indexes]
		X,y = self.__data_generation(image_paths)
		return X,np.expand_dims(y, axis=-1)

	def __data_generation(self,image_paths):
		temp_image = []
		temp_mask = []
		for i in image_paths:
			img = cv2.imread(i)
			j = i.replace('test','ground_truth').split('.')[-2]+'_mask.png'
			if j in self.mask_paths:
				mask = cv2.imread(j,0)
			else:
				mask = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.int8)
			temp_image.append(img)
			temp_mask.append(mask)
		return temp_image,temp_mask