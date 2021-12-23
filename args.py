import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device',type=str,help='cuda or cpu')
	parser.add_argument('--base_path',type=str,help='location to mvtec folder')
	parser.add_argument('--folder_path',type=str,help='which defect class')
	parser.add_argument('--model',type=str,help='resnet18/resnet50')
	parser.add_argument('--dim',  type=int,
	                    help='no of feature channels dimensions to take')
	parser.add_argument('--image_size',type=int,nargs='+',
	                    help='image size')
	parser.add_argument('--center_size',type=int,nargs='+',
	                    help='center crop size')
	args = parser.parse_args()
	return args