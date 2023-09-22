from keras.models import Model
from keras.layers import BatchNormalization, Dropout, MaxPool2D, Conv2D, Input, Flatten, Dense, GlobalAveragePooling2D, Reshape
from keras.applications import ResNet50

# class JigsawModel:
#     def __init__(self, tile_size, num_tiles):
#         self.tile_size = tile_size
#         self.num_tiles = num_tiles

def JigsawArch(tile_size, num_tiles):
    inputs = Input((tile_size, tile_size, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D((2,2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D((2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)
    flatten = Flatten()(pool3)

    dense1 = Dense(128, activation='relu')(flatten)
    dense1 = Dropout(0.25)(dense1)
    dense2 = Dense(num_tiles, activation='softmax')(dense1)

    model = Model(inputs, dense2)

    return model

    # define ResNet50
    # def ResnetArch(self):
    #     input_shape = (self.tile_size, self.tile_size, 3)
    #     inputs = Input(input_shape)
    #
    #     pretrained_resnet = ResNet50(include_top=False,
    #                                  weights='imagenet',
    #                                  input_shape=input_shape)
    #     for layer in pretrained_resnet.layers:
    #         layer.trainable = False
    #     pretrained_output = pretrained_resnet.output
    #     # pretrained_output = GlobalAveragePooling2D()(pretrained_output)
    #     reshaped_output = Reshape((-1, self.tile_size, self.tile_size, 3))(pretrained_output)
    #     Jigsaw_output = self.JigsawArch(reshaped_output)
    #
    #     model = Model(inputs=inputs,
    #                   outputs=Jigsaw_output)
    #
    #     return model