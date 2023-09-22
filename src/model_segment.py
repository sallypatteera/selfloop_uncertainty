import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D,UpSampling2D, Conv2DTranspose, Concatenate, Input, BatchNormalization, Flatten, Dense, Dropout
from keras.optimizers.legacy import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.python.keras.metrics import Recall, Precision
from metrics import dice_coef, iou
import matplotlib.pyplot as plt
from keras.models import Model

class unet_model:
    def __init__(self, input_shape, lr, model_path, csv_path, train_dataset, num_epoch, valid_dataset, train_steps, valid_steps):
        self.pretrained_jigsaw = tf.keras.models.load_model("files/model.h5")
        self.input_shape = input_shape
        self.lr = lr
        self.model_path = model_path
        self.csv_path = csv_path
        self.train_dataset = train_dataset
        self.num_epoch = num_epoch
        self.valid_dataset = valid_dataset
        self.train_steps = train_steps
        self.valid_steps = valid_steps


    def build_unet(self):
        """ Input """
        inputs = Input(self.input_shape)

        """ Encoder """
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1_1')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPool2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2_1')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPool2D((2, 2))(conv2)
        pool2 = Dropout(0.25)(pool2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', name='conv3_1')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPool2D((2, 2))(conv3)
        pool3 = Dropout(0.25)(pool3)

        """ Bottleneck """
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

        """ Decoder """
        up5 = UpSampling2D((2,2))(conv4)
        up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
        merge5 = Concatenate()([conv3, up5])
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

        up6 = UpSampling2D((2,2))(conv5)
        up6 = Conv2D(256, 2, activation='relu', padding='same')(up6)
        merge6 = Concatenate()([conv2, up6])
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

        up7 = UpSampling2D((2,2))(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same')(up7)
        merge7 = Concatenate()([conv1, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        """ Output """
        outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

        model = tf.keras.Model(inputs, outputs)
        return model

    def run_unet(self):
        unet = self.build_unet()
        # initializing weights from jigsaw model

        # for idx, layer in enumerate(self.pretrained_jigsaw.layers):
        #     if isinstance(unet[idx], Conv2D) or isinstance(unet[idx], BatchNormalization):
        #         unet.layes[idx].set_weights(layer.get_weights())
        # metrics = [dice_coef, iou, Recall(), Precision()]

        unet.compile(loss="binary_crossentropy", optimizer=Adam(self.lr), metrics=['accuracy'])
        unet.summary()

        callbacks = [
            ModelCheckpoint(self.model_path, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),# if 5 epoch loss doesn't reduce, model reduces the lr
            CSVLogger(self.csv_path),
            tf.keras.callbacks.TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False), #log the data while training# reduce overfitting and stop the training when val_loss stop improving for 20 epochs
        ]

        unet_history = unet.fit(
            self.train_dataset,
            epochs=self.num_epoch,
            validation_data=self.valid_dataset,
            steps_per_epoch=self.train_steps,
            validation_steps=self.valid_steps,
            callbacks=callbacks)

        return unet_history

    def visualize_unet(self):
        history = self.run_unet()
        acc = history.history['accuracy']  # new change for TF changes
        val_acc = history.history['val_accuracy']  # new change for TF changes

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.num_epoch)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
