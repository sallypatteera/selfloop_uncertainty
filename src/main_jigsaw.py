import numpy as np
import tensorflow as tf
import os
from glob import glob
from keras.optimizers.legacy import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.python.keras.metrics import Recall, Precision

from dataprep import create_dir
from data import JigsawDataset
from metrics import dice_coef, iou
from model_jigsaw import JigsawArch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Seeding allow the result to be reproducible"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """Folder for saving data"""
    create_dir("files")

    """Hyperparameter"""
    batch_size = 16
    lr = 1e-2
    num_epoch = 10
    model_path = 'files/jigsaw_model.h5'
    csv_path = "files/jigsaw_data.csv" # accuracy and loss data

    """Dataset: 60:20:20"""
    dataset_path = "/Users/Sally/PycharmProjects/SelfLoop_Uncertainty/dataset/dataset_isic_monuseg"
    images= sorted(glob(os.path.join(dataset_path, "training_image", "*.jpg")))
    image_size = (256, 256)

    dataset = JigsawDataset(images, image_size, "train")
    dataset.load_images()
    x_train, x_val, y_train, y_val = dataset.prep_dataset()

    # (train_img, train_msk), (val_img, val_msk), (test_img, test_msk) = load_data(dataset_path)

    print(f"Train: {len(x_train)}")
    print(f"Valid: {len(y_train)}")

    # train_steps = len(x_train)//batch_size
    # valid_steps = len(y_train) // batch_size
    #
    # if len(x_train) % batch_size != 0:
    #     train_steps += 1
    # if len(y_train) % batch_size != 0:
    #     valid_steps += 1

    print(">>>>> Next: load the model")
    """ Model """
    tile_size = 256 // 3
    num_tiles = 9
    # inputs = ((tile_size, tile_size, 3))
    # model1 = build_encoder((H, W, 3))
    # model2 = JigsawModel(tile_size, num_tiles)
    jigsaw_model = JigsawArch(tile_size, num_tiles)


    metrics = [dice_coef, iou, Recall(), Precision()]
    jigsaw_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                   optimizer=Adam(learning_rate=lr),
                   metrics=['accuracy'])
    jigsaw_model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1), # if 5 epoch loss doesn't reduce, model reduces the lr
        CSVLogger(csv_path), # log the data whil                                                                                                                                                                                                           e training
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
        # TensorBoard(log_dir="./src", write_images=True)# reduce overfitting and stop the training when val_loss stop improving for 20 epochs
    ]

    print(">>>>> start the training")

    history = jigsaw_model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        # steps_per_epoch=train_steps,
        # validation_steps=valid_steps,
        callbacks=callbacks,
    )


# plot the training
acc = history.history['accuracy'] # new change for TF changes
val_acc = history.history['val_accuracy'] # new change for TF changes

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epoch)

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