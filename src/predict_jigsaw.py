import numpy as np
import tensorflow as tf
import os
from glob import glob
from keras.utils import CustomObjectScope

from dataprep import create_dir
from data import JigsawDataset
from metrics import dice_coef, iou

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving results """
    create_dir("jigsaw_results")

    """ Load the model """
    with CustomObjectScope({'iou':iou,
                            'dice_coef':dice_coef}):
        model = tf.keras.models.load_model("files/jigsaw_model.h5")
        model.summary()

    """ load the dataset """
    dataset_path = "/Users/Sally/PycharmProjects/SelfLoop_Uncertainty/dataset/dataset_isic_monuseg"
    images= sorted(glob(os.path.join(dataset_path, "training_image", "*.jpg")))
    masks = sorted(glob(os.path.join(dataset_path, "training_mask", "*.png")))
    image_size = (256, 256)

    dataset = JigsawDataset(images, image_size, "predict")
    test_dataset = dataset.load_images()

    test_loss, test_acc =model.evaluate(test_dataset[0], test_dataset[1])
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)

    """ Make predictions """
    pred_idx = model.predict(test_dataset[0])
    # pred_arr = np.array(pred_idx)

    # """ Convert predictions to integers """
    # threshold = 0.5
    # int_pred = tf.clip_by_value(tf.cast(tf.floor(pred_idx * 9), tf.int32), 0, 8)
    #
    # """ convert true labels """
    # # int_pred = int_pred.tolist()
    #
    # """ Compute accuracy """
    # test_labels = test_dataset[1]
    # num_correct = tf.reduce_sum(tf.cast(tf.equal(int_pred, test_labels), tf.float32))
    # accuracy = num_correct.numpy() / len(test_labels)
    # print("Accuracy:", accuracy)
