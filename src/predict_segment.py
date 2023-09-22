import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from dataprep import load_data, create_dir


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)        #(H, W, 3)
    img = cv2.resize(img, (256, 256))
    ori_img = img # original resized img
    img = img/255.0 # range = [0,1], pixel = 255
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return ori_img, img

W=256
H=256
def read_mask(path):
    msk = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #(H, W)
    msk = cv2.resize(msk, (W, H))
    ori_msk = msk
    msk = msk/255.0 # range = [0,1], pixel = 255
    msk = msk.astype(np.int32)                    #(256, 256)
    return ori_msk, msk

# save predicted images in comparison with original mask and input image
def save_results(ori_img, ori_msk, msk_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_msk = np.expand_dims(ori_msk, axis=-1) # (256, 256, 1)
    ori_msk = np.concatenate([ori_msk, ori_msk, ori_msk], axis=-1) # (256, 256, 3)

    msk_pred = np.expand_dims(msk_pred, axis=-1) # (256, 256, 1)
    msk_pred = np.concatenate([msk_pred, msk_pred, msk_pred], axis=-1) # (256, 256, 3)

    cat_images = np.concatenate([ori_img, line, ori_msk, line, msk_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """Seeding allow the result to be reproducible"""
    np.random.seed(42)
    tf.random.set_seed(42)

    """Folder for saving results"""
    create_dir("results")

    """ Load the model """
    with CustomObjectScope({'iou':iou, 'dice_coef':dice_coef}):
        model = tf.keras.models.load_model("seg_files/seg_mogel.h5")
        model.summary()

    """ Load the dataset """
    dataset_path = "/Users/Sally/PycharmProjects/SelfLoop_Uncertainty/dataset/dataset_isic18/"
    (train_img, train_msk), (val_img, val_msk), (test_img, test_msk) = load_data(dataset_path)

    SCORE = []
    # tqdm = progress bar
    for img, msk in tqdm(zip(test_img, test_msk), total=len(test_img)):
        """ Extracting the image name"""
        name = img.split("/")[-1]

        """ Read the iamge and mask """
        ori_img, img = read_image(img)
        ori_msk, msk = read_mask((msk))

        """ Predicting the mask """
        msk_pred = model.predict(img)[0] > 0.5 # first index from (1, 256, 256, 256); pred between 0-1 if pixel>0.5, treated as 1; pixel<0.5, treated as 0
        msk_pred = np.squeeze(msk_pred, axis=-1)
        msk_pred = msk_pred.astype(np.int32)  # (256, 256)

        """ Saving the predicted mask """
        save_image_path = f"results/{name}"
        save_results(ori_img, ori_msk, msk_pred, save_image_path)

        """ Flatten the array """
        msk = msk.flatten()
        msk_pred = msk_pred.flatten()

        """ Calculating metrics values """
        acc_value = accuracy_score(msk, msk_pred)
        f1_value = f1_score(msk, msk_pred, labels=[0,1], average='binary')
        recall_value = recall_score(msk, msk_pred, labels=[0,1], average='binary')
        precision_value = precision_score(msk, msk_pred, labels=[0,1], average='binary', zero_division=1)
        SCORE.append([name, acc_value, f1_value, recall_value, precision_value])

        """ Mean metrics values """
        score = [s[1:] for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Recall: {score[2]:0.5f}")
        print(f"Precision: {score[3]:0.5f}")

        df = pd.DataFrame(SCORE, columns=["Image Name", "Acc", "F1", "Recall", "Precision"])
        df.to_csv("seg_files/score.csv")