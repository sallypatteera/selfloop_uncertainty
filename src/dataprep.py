import os
import random

import numpy as np
import cv2
from glob import glob

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

import albumentations as A

from hamming import hamming_set

H = 225
W = 225

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y): # x=image and y=mask
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path, split=0.2):
    images= sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth", "*.png")))

    test_size = int(len(images)*split)

    train_img, val_img = train_test_split(images, test_size=test_size, random_state=42)
    train_msk, val_msk = train_test_split(masks, test_size=test_size, random_state=42)

    train_img, test_img = train_test_split(train_img, test_size=test_size, random_state=42)
    train_msk, test_msk = train_test_split(train_msk, test_size=test_size, random_state=42)
    # return: ['path','path',...] for each
    return (train_img, train_msk), (val_img, val_msk), (test_img, test_msk)

def transform(image):
    transform = A.Compose(
        [
            A.Normalize(),
            A.ColorJitter(),
            A.Rotate([0,270]),
        ]
    )
    transformed_img = transform(image=image)['image']
    return transformed_img
# def read_image(path):
#     # path = path.decode() # main_segmentation uses this; main_pretext does not
#     img = cv2.imread(path, cv2.IMREAD_COLOR)        #(H, W, 3)
#     if img is None:
#         print("wrong path: {}".format(path))
#         return None
#     else:
#         img = cv2.resize(img, (225, 225))
#     img = img.astype(np.float32)
#     img = transform(img)
#     return img                                      #(512, 512, 3)
def read_image(path):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except TypeError as e:
        print("Error reading image:", e)
        print("Path type:", type(path))
        return None

    if img is None:
        print("Couldn't read image: {}".format(path))
        print("type(path) = {}; type(img) = {}".format(type(path), type(img)))
        return None

    img = cv2.resize(img, (225, 225))
    img = img.astype(np.float32)
    img = transform(img)
    print("type(path) = {}; type(img) = {}".format(type(path), type(img)))
    return img

import matplotlib.pyplot as plt
def visualize(path):
    img = read_image(path)
    plt.imshow(img)
    return img

def split_tiles(img_path):
    img = read_image(img_path)
    if img is None:
        return None
    h, w, channel = img.shape
    tile_size = h // 3
    num_tile = 3

    # ham_set = hamming_set()
    # ran_ham = random.choice(ham_set)
    # idx = [i for i in range(9)]

    split_image = []
    for row in range(num_tile):
        for col in range(num_tile):
            tile = img[row*tile_size:(row+1)*tile_size,
                   col*tile_size:(col+1)*tile_size, :]
            split_image.append(tile)
    # split_image = np.array([split_image])
    train_dataset = random.shuffle(split_image)
    target_dataset = split_image
    return target_dataset, train_dataset

def get_dataset(paths, batch):
    def func(i):
        i = i.numpy()
        target, train = split_tiles(paths[i])
        return target, train
    z = list(range(len(paths)))
    dataset = tf.data.Dataset.from_generator(lambda: z, tf.uint8)
    dataset = dataset.map(lambda i: tf.py_function(func=func,
                                                   inp=[i],
                                                   Tout=[tf.uint8, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    yield dataset


def split_dataset(paths, batch): # split_img = split_tiles(img_path)
    for path in paths:
        dataset = tf.data.Dataset.from_generator(
            split_tiles,
            args=[path],
            output_types=tf.float32
            )
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(10)
        yield dataset

def get_split_dataset(paths, batch):
    iterator = iter(paths)
    while True:
        try:
            path = next(iterator)
            print("\nprocessing path:", path)
        except StopIteration:
            print("\nNo more paths to process")
            break
        split_tiles_output = split_tiles(path)
        if split_tiles_output is None:
            print(">>>>> skipping path\n")
            continue
        split, index = split_tiles_output
        split_data = split_dataset((split, index), batch)
        yield split_data
    # iteratar = iter(paths)
    # for path in iterator:
    #     split_tiles_output = split_tiles(path)
    #     if split_tiles_output is None:
    #         print(">>>>> skipping path\n")
    #         continue
    #     split, index = split_tiles(path)
    #     # if split is None:
    #     #     continue
    #     split_data = split_dataset((split, index), batch)
    #     yield split_data
    #     try:
    #         next_path = next(iterator)
    #         print("\nprocessing next path:", next_path)
    #     except StopIteration:
    #         print("\nNo more paths to process")
    #         no_more_path = True
    #         if no_more_path:
    #             break

def read_mask(path):
    path = path.decode()
    msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #(H, W)
    msk = cv2.resize(msk, (W, H))
    msk = msk/225.0 # range = [0,1], pixel = 255
    msk = msk.astype(np.float32)                    #(256, 256)
    msk = np.expand_dims(msk, axis=-1)              #(256, 256, 1)
    return msk

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

# dataset_path = "/Users/Sally/PycharmProjects/SelfLoop_Uncertainty/dataset/dataset_isic18/"
# all_data = load_data(dataset_path)
# train_img, train_msk = all_data[0], all_data[1]
# print(f"Train: {len(train_img)}, {len(train_msk)}")
#
# og_train_dataset = split_tiles(train_img)
# og_zero = og_train_dataset[0]
# og_one = og_train_dataset[1]
#
# shuf_train_dataset = shuffling(og_train_dataset[0], og_train_dataset[1])
# shuf_zero = shuf_train_dataset[0]
# shuf_one = shuf_train_dataset[1]
#
# print("og_zero", len(og_zero))
# print("og_one", len(og_one))
# print("shuf_train_dataset", len(shuf_train_dataset))
#
# for i in range(len(og_train_dataset)):
#     print(f'Length of x[{i}]: {len(og_train_dataset[i])}, Length of y[{i}]: {len(shuf_train_dataset[i])}')