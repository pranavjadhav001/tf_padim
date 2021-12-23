import tensorflow as tf
import cv2
import numpy as np
import cv2
import os
import glob

def tf_embedding_concat_resize_method(x,y,resize_method='nearest'):
	b,h1,w1,c1 = x.shape
	b,h2,w2,c2 = y.shape
	new_img = tf.image.resize(y,(h1,w1),method=resize_method)
	final_img = tf.concat([x,new_img],axis=-1)
	return final_img

def model_input_image(image_path,width,height,preprocess_input):
	img = cv2.imread(image_path)
	img = cv2.resize(img, (width,height))
	img = preprocess_input(img)
	return img

def tf_embedding_concat_patch_method(l1,l2):
	bs,h1,w1,c1 = l1.shape
	_,h2,w2,c2 = l2.shape
	s = int(h1/h2)
	x = tf.compat.v1.extract_image_patches(l1,ksizes=[1,s,s,1],strides=[1,s,s,1],\
		rates=[1,1,1,1],padding='valid')
	x = tf.reshape(x,(bs,-1,h2,w2,c1))

	col_z = []
	for idx in range(x.shape[1]):
		col_z.append(tf.concat([x[:,idx,:,:,:],l2],axis=-1))
	z = tf.stack(col_z,axis=1)
	z = tf.reshape(z,(bs,h2,w2,-1))
	if s == 1:
		return z
	z = tf.nn.depth_to_space(z,block_size=s)
	return z