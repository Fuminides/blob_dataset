import argparse

def gen_parser():
	'''
	Generates the parser.
	'''	

	parser = argparse.ArgumentParser()
	parser.add_argument('--instances', type=int, default=1000, help='Number of images to generate.')
	parser.add_argument('--image_size', type=int, default=256, help='Size of the image.')
	parser.add_argument('--destination_path', type=str, default='data/', help='Path to save the images and the labels.')
	parser.add_argument('--seed', type=int, default=0, help='Random seed.')
	parser.add_argument('--model_destination', type=str, default='model/', help='Path to save/load the model.')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
	parser.add_argument('--dataset', type=str, default='blob', help='Dataset to use. Options: mnist, cifar10, cifar100, blob.')
	parser.add_argument('--grad_cam_output_path', type=str, default='grad_cam/', help='Path to save the grad cam images.')
	parser.add_argument('--dataset_path', type=str, default='../trials/', help='Path to the dataset.')
	return parser
