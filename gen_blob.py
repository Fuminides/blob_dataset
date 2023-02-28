import numpy as np
import pandas as pd
from PIL import Image


def random_image(size):
	'''
	Generate a random image with a specific size
	'''
	return np.random.rand(*size)


def gen_one_blob(image, size=.2, sample_size=10000):
	'''
	Adds one blob into an image
	'''
	pos_x = int(image.shape[0] * np.random.rand(1))
	pos_y = int(image.shape[1] * np.random.rand(1))

	mean = [pos_x, pos_y]
	cov  = [[size, 0], [0, size]]  
	aux  = np.random.multivariate_normal(mean, cov, sample_size).astype(int)

	coords_x = np.clip(aux[:, 0], 0, image.shape[0] - 1)
	coords_y = np.clip(aux[:, 1], 0, image.shape[1] - 1)

	for coord_x, coord_y in zip(coords_x, coords_y):
		image[coord_x, coord_y] += 0.2

	return np.clip(image, 0, 1)


def gen_big_blob(image, size_params):
	return gen_one_blob(image, size=size_params['big'] * np.random.rand(1) + 0.5, sample_size=size_params['sample_size'])


def gen_small_blob(image, size_params):
	return gen_one_blob(image, size=size_params['small'] * np.random.rand(1) + 0.5, sample_size=size_params['sample_size'])


def get_size_params(size):
	'''
	Return the adequate statistical parameters for each 

	:param size: image size to sample
	:return: hardcoded to 32x32 empirically tested.
	'''
	if np.equal(size, [32, 32]).all():
		return {'sample_size': 10000, 'small': 0.2, 'big': 5}
	elif np.equal(size, [256, 256]).all():
		return {'sample_size': 500000, 'small': 30, 'big': 300}



def gen_init_dataset(dataset_size, image_size):
	'''
	Gens the initial set of random images.
	'''
	return random_image([dataset_size] + image_size)


def simple_image(image, small=None):
	'''
	Creates a blob in the image. You can specify the size.
	'''
	params = get_size_params(image.shape)

	if small is None:
		small = np.random.rand(1) < 0.5

	if small:
		new_image = gen_small_blob(image, params)
	else:
		new_image = gen_big_blob(image, params)

	return new_image, small


def complex_image(image, n_blobs):
	'''
	Creates various blobs in the image. You can specify the number of blobs.
	'''
	if n_blobs > 9:
		small = True
	else:
		small = np.random.rand(1) < 0.5

	for i in range(n_blobs):
		image, _ = simple_image(image, small=small)
		assert image.max() <= 1.0

	return image, small


def gen_random_blob(image):
	if np.random.rand(1) < 0.5:
		new_image, small = simple_image(image)
		n_blobs = 1
	else:
		n_blobs = np.random.randint(10) + 1
		new_image, small = complex_image(image, n_blobs)

	return new_image, small, n_blobs


def gen_dataset(instances, image_size, destination_path):
	'''
	Generate a blob dataset.

	:param instance: number of images to generate.
	:param image_size: the size of each image.
	:param destination_path: path to save the images and the labels.
	'''
	if destination_path[-1] != '/':
		destination_path += '/'
	init_images = gen_init_dataset(instances, image_size)
	y_file = pd.DataFrame(np.zeros((instances, 2)), columns=['Blobs', 'Size'])

	for ix in range(instances):
		image = init_images[ix]
		new_image, small, n_blobs = gen_random_blob(image)

		y_file.iloc[ix, 0] = n_blobs
		y_file.iloc[ix, 1] = small
		img = Image.fromarray(np.uint8(new_image * 255))

		img.convert('RGB').save(destination_path + str(ix) + '.jpg')

	y_file.to_csv(destination_path + 'y_dataset.csv')


if __name__ == '__main__':
	gen_dataset(1000, [256, 256], './trials/')	








