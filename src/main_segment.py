import numpy as np
import tensorflow as tf

from dataprep import create_dir
from data import SegmentData
from model_segment import unet_model

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving data """
    create_dir("seg_files")

    """ Hyperparameter """
    batch_size = 4
    lr = 1e-4
    num_epoch = 5
    model_path = "seg_files/seg_mogel.h5" # seg_model.h5
    csv_path = "seg_files/seg_data.csv"

    """ Dataset: 60:20:20 """
    dataset_path = "/Users/Sally/PycharmProjects/SelfLoop_Uncertainty/dataset/dataset_isic_monuseg"
    seg_data = SegmentData(dataset_path, "train")

    (train_img, train_msk), (val_img, val_msk), (test_img, test_msk) = seg_data.load_data()

    print(f"Train: {len(train_img)}, {len(train_msk)}")
    print(f"Valid: {len(val_img)}, {len(val_msk)}")
    print(f"Test: {len(test_img)}, {len(test_msk)}")

    train_dataset = seg_data.tf_dataset(train_img, train_msk, batch_size)
    valid_dataset = seg_data.tf_dataset(val_img, val_msk, batch_size)

    train_steps = len(train_img)//batch_size
    valid_steps = len(val_img)//batch_size

    if len(train_img) % batch_size != 0:
        train_steps += 1

    if len(val_img) % batch_size != 0:
        valid_steps += 1

    """ Model """
    unet = unet_model(input_shape=(256,256,3),
                      lr=lr,
                      model_path=model_path,
                      csv_path=csv_path,
                      train_dataset=train_dataset,
                      num_epoch=num_epoch,
                      valid_dataset=valid_dataset,
                      train_steps=train_steps,
                      valid_steps=valid_steps)
    unet.visualize_unet()