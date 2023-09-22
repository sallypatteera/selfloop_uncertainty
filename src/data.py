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

class JigsawDataset:
    def __init__(self, file_paths, image_size, mode):
        self.file_paths = file_paths
        self.image_size = image_size
        self.num_tiles = 9
        self.mode = mode #JUST ADDED

    def train_transform(self, image):
        transform = A.Compose([
            A.Normalize(),
            A.ColorJitter(),
            A.Rotate([0,270]),
            A.Resize(256, 256),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast()
        ])
        transformed = transform(image=image)['image']
        return transformed

    def test_transform(self, image):
        transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256)
        ])
        transformed = transform(image=image)['image']
        return transformed

    def load_images(self):
        # init
        tiles = []
        labels = []
        # shuffle file_paths on each time preparing data for training
        random.shuffle(self.file_paths)

        # read image
        for file_path in self.file_paths:
            img = cv2.imread(file_path)
            img = img.astype(np.float32)

            if self.mode == "train":
                img = self.train_transform(img)
            elif self.mode == "predict":
                img = self.test_transform(img)

            # split into tiles
            h, w, _ = img.shape
            tile_size = h // 3

            for row in range(3):
                for col in range(3):
                    tile = img[row * tile_size:(row + 1) * tile_size,
                           col * tile_size:(col + 1) * tile_size, :]
                    tiles.append(tile)
                    labels.append(row*3 + col)

        self.tiles = np.array(tiles)
        self.labels = np.array(labels)

        if self.mode == "predict":
            self.tiles, self.labels = shuffle(self.tiles, self.labels, random_state=42)
            return self.tiles, self.labels

    def prep_dataset(self, test_size=0.2, random_state=42):
        # split dataset into train and validation sets
        # x = tile; y = label
        x_train, x_val, y_train, y_val = train_test_split(self.tiles,
                                                          self.labels,
                                                          test_size=test_size,
                                                          random_state=random_state)
        # convert integer labels into a one-hot encoded categorical format
        y_train = tf.keras.utils.to_categorical(y_train,
                                                num_classes=self.num_tiles)
        y_val = tf.keras.utils.to_categorical(y_val,
                                              num_classes=self.num_tiles)
        return x_train, x_val, y_train, y_val

    def load_testset(self):
        tiles = []
        for file_path in self.file_paths:
            img = cv2.imread(file_path)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32)

            transform = A.Compose([
                A.Normalize(),
            ])
            img = transform(image=img)['image']

            h, w, _ = img.shape
            tile_size = h // 3

            for row in range(3):
                for col in range(3):
                    tile = img[row * tile_size:(row + 1) * tile_size,
                           col * tile_size:(col + 1) * tile_size, :]
                    tiles.append(tile)
        random.shuffle(tiles) # cannot shuffle WRONG syntax
        tiles = np.array(tiles)

        return tiles

class SegmentData:
    def __init__(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode

    def load_data(self, split=0.2):
        images = sorted(glob(os.path.join(self.dataset_path, "training_image", "*.jpg")))
        masks = sorted(glob(os.path.join(self.dataset_path, "training_mask", "*.png")))

        # shuffle the data
        images, masks = shuffle(images, masks, random_state=42)

        test_size = int(len(images) * split)

        train_img, val_img = train_test_split(images, test_size=test_size, random_state=42)
        train_msk, val_msk = train_test_split(masks, test_size=test_size, random_state=42)

        train_img, test_img = train_test_split(train_img, test_size=test_size, random_state=42)
        train_msk, test_msk = train_test_split(train_msk, test_size=test_size, random_state=42)
        # return: ['path','path',...] for each
        return (train_img, train_msk), (val_img, val_msk), (test_img, test_msk)

    def train_transform(self, image):
        transform = A.Compose([
            A.Normalize(),
            A.ColorJitter(),
            A.Rotate([0,270]),
            A.RandomCrop(256, 256),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast()
        ])
        transformed = transform(image=image)['image']
        return transformed

    def test_transform(self, image):
        transform = A.Compose([
            A.Normalize(),
            A.RandomCrop(256, 256)
        ])
        transformed = transform(image=image)['image']
        return transformed

    def read_img(self, path):
        path = path.decode()
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = img.astype(np.float32)

        # data augmentation
        if self.mode == "train":
            img = self.train_transform(img)
        elif self.mode == "predict":
            img = self.test_transform(img)

        return img

    def read_msk(self, path):
        path = path.decode()
        msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (256, 256))
        msk = msk/256.0
        msk = msk.astype(np.float32)
        msk = np.expand_dims(msk, axis=-1)
        return msk

    def tf_parse(self, x, y):
        def _parse(x, y):
            x = self.read_img(x)
            y = self.read_msk(y)
            return x, y

        x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
        x.set_shape([256, 256, 3])
        y.set_shape([256, 256, 1])
        return x, y

    def tf_dataset(self, X, Y, batch):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.prefetch(10)
        return dataset

