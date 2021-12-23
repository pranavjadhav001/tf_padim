import os
from classification_models.tfkeras import Classifiers
import tensorflow as tf
import numpy as np
import cv2
import glob
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers.experimental.preprocessing import Resizing,CenterCrop
from scipy.ndimage import gaussian_filter
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from utils import tf_embedding_concat_resize_method,tf_embedding_concat_patch_method,model_input_image
from dataset import MvtecDataGenerator
from args import get_args

random.seed(1024)
np.random.seed(1024)
tf.random.set_seed(1024)
args = get_args()

if args.device == 'cpu':
	config = tf.ConfigProto(
        device_count = {'GPU': 0}
   	 )
	sess = tf.Session(config=config)
	
if args.model == 'resnet18':
	Resnet18,preprocess_input = Classifiers.get('resnet18')
	model = Resnet18(tuple(args.center_size),weights='imagenet')
	output_layers = ['stage2_unit1_bn1','stage3_unit1_bn1','stage4_unit1_bn1']
	outputs = [model.get_layer(i).output for i in output_layers]
	new_model = Model(inputs=model.inputs,outputs=outputs)

elif args.model == 'resnet50':
	model = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_shape=tuple(args.center_size))
	output_layers = ['conv2_block3_3_bn','conv2_block3_3_bn','conv4_block6_3_bn']
	outputs = [model.get_layer(i).output for i in output_layers]
	new_model = Model(inputs=model.inputs,outputs=outputs)
	preprocess_input = tf.keras.applications.resnet.preprocess_input

data_augmentation= tf.keras.Sequential(
	[Resizing(*args.image_size[:2]),\
	 CenterCrop(*args.center_size[:2]),\
	 tf.keras.layers.Lambda(preprocess_input)])
mask_augmentation = tf.keras.Sequential(
	[Resizing(*args.image_size[:2]),\
	 CenterCrop(*args.center_size[:2])])

datagen = MvtecDataGenerator(dir_path=os.path.join(args.base_path,args.folder_path,'train'))
test_datagen = MvtecDataGenerator(dir_path=os.path.join(args.base_path,args.folder_path,'test'),\
	mask_path=os.path.join(args.base_path,args.folder_path,'ground_truth'))

layer_outputs = {'layer1':[],'layer2':[],'layer3':[]}
for i in range(len(datagen)):
	x,_ = datagen[i]
	x_aug = data_augmentation(np.array(x))
	img_outputs = new_model.predict(x_aug)
	for k,v in zip(layer_outputs.keys(),img_outputs):
		layer_outputs[k].extend(v)

conc1 = tf_embedding_concat_resize_method(np.asarray(layer_outputs['layer1']), np.asarray(layer_outputs['layer2']))
conc2 = tf_embedding_concat_resize_method(conc1, np.asarray(layer_outputs['layer3']))
idx = random.sample(range(0,conc2.shape[-1]), args.dim)
embedding_vec = tf.gather(conc2,indices=idx,axis=-1)
del conc1,conc2,layer_outputs
b,h,w,c = embedding_vec.shape
embedding_vec = tf.reshape(embedding_vec,(b,h*w,c))
mean = tf.reduce_mean(embedding_vec,axis=0)
cov = np.zeros(shape=(c,c,h*w))
I = np.identity(c)
for i in range(h*w):
	cov[:,:,i] = np.cov(embedding_vec[:,i,:],rowvar=False) + 0.01*I
cov_inv = np.linalg.inv(cov.T).T
train_outputs = [mean,cov_inv]
layer_outputs = {'layer1':[],'layer2':[],'layer3':[]}
masks = []
for i in range(len(test_datagen)):
	x,y = test_datagen[i]
	masks.extend(mask_augmentation(np.array(y)).numpy())
	x_aug = data_augmentation(np.array(x))
	img_outputs = new_model.predict(x_aug)
	for k,v in zip(layer_outputs.keys(),img_outputs):
		layer_outputs[k].extend(v)
conc1 = tf_embedding_concat_resize_method(np.asarray(layer_outputs['layer1']), np.asarray(layer_outputs['layer2']))
conc2 = tf_embedding_concat_resize_method(conc1, np.asarray(layer_outputs['layer3']))
embedding_vec = tf.gather(conc2,indices=idx,axis=-1)
del conc1,conc2,layer_outputs
b,h,w,c = embedding_vec.shape
embedding_vec = tf.reshape(embedding_vec,(b,h*w,c))

#einsum optimization inspired from
#https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/issues/8#issuecomment-850027147
delta = embedding_vec - np.expand_dims(mean, axis=0)
dist_list = np.sqrt(np.einsum('nlj,jkl,nlk->nl',delta,cov_inv,delta))
dist_list = np.array(dist_list).reshape(b,h,w)
dist_list = np.expand_dims(dist_list, axis=-1)
score_map = tf.image.resize(dist_list, size=(args.center_size[0],args.center_size[1]))
score_map = score_map.numpy()
for i in range(score_map.shape[0]):
	score_map[i] = gaussian_filter(score_map[i], sigma=4)
max_score = score_map.max()
min_score = score_map.min()
scores = (score_map -min_score)/(max_score-min_score)
total_roc_auc = []
masks = (np.array(masks)/255).astype(np.uint8)
img_scores = scores.reshape(scores.shape[0],-1).max(axis=1)
gt_list = np.array(masks).reshape(masks.shape[0],-1).max(axis=1)
gt_list = np.asarray(gt_list)
fpr,tpr,_ = roc_curve(gt_list,img_scores)
img_roc_auc = roc_auc_score(gt_list,img_scores)
total_roc_auc.append(img_roc_auc)
precision,recall,thresholds = precision_recall_curve(masks.astype(np.uint8).flatten(),scores.flatten())
a = 2*precision*recall
b = precision + recall
f1 = np.divide(a,b,out=np.zeros_like(a),where=b!=0)
threshold = thresholds[np.argmax(f1)]
fpr,tpr,_ = roc_curve(masks.astype(np.uint8).flatten(),scores.flatten())
per_pixel_rocauc = roc_auc_score(masks.astype(np.uint8).flatten(), scores.flatten())
print(args.folder_path,'image ROCAUC: %.3f' %(img_roc_auc),'pixel ROCAUC: %.3f' %(per_pixel_rocauc))







