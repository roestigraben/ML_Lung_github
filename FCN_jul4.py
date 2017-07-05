from __future__ import print_function

import numpy as np
import cv2
import math
from datetime import datetime
from collections import OrderedDict
import logging
from datetime import datetime
from six.moves import xrange
from scipy.io import loadmat
from tqdm import tqdm, trange

import tensorflow as tf
import TensorflowUtils as utils

# import the UNET model
import ModelLibrary as model

# import the pattern generator
import DatasetReader as dataset1

# import the medical images and ROI's
import BatchDatsetReader as dataset2
import read_HUGdata as HUGdata

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

############ SETTINGS ##############

NUM_OF_CLASSESS = 12                                # sets the number of different labels (classes)
LUNG_DATASET = False                                 # selects if the pattern generator or true lung data is used 
BATCH_SIZE = 4                                      # size of parallel processed samples
IMAGE_SIZE = 512                                    # size of sample in pixel x pixel
LEARNING_RATE = 1e-4                                # used for the optimizer
MAX_ITERATION = int(1e5 + 1)                        # overall count on training steps
DEBUG = False                                       # selects/deselects additional info printed
ROOT_LOGDIR = "logs"                                # directory from which we create log subdirectories for model and events
now = datetime.utcnow().strftime("%Y-%B-%d-%H-%M")
LOGS_DIR = "{}/run-{}/".format(ROOT_LOGDIR, now)    # directory where the events logs are stored for Tensorboard
DATA_DIR = "dataJul4/"                              # in case of real data, directory where data is available


# parameters for the UNET model 
n_class = NUM_OF_CLASSESS                           #
channels = 1                                        # greyscale (1) or RGB (3)
layers = 3                                          # number of convolutional layers in model 
features_root = 64                                  # initial number of kernels (leads to intial number of feature maps)
keep_prob = 0.85                                    # drop-out rate (0.85 means keep 85%)


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if DEBUG:
        # print(len(var_list))
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)
    return optimizer.apply_gradients(grads)

def main(argv=None):

    # define Tensorflow graph

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    logits, _, _ = model.create_conv_net(image, keep_prob, channels, n_class, layers, features_root, DEBUG)

    annotation_pred = tf.argmax(logits, dimension=3, name="prediction")

    pred_annotation = tf.expand_dims(annotation_pred, dim=3)

    # loss (or entropy)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, 
                                                                        squeeze_dims=[3]), name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)
    
    # Accuracy
    correct_prediction = tf.equal(annotation, tf.cast(pred_annotation, tf.int32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    acc_summary = tf.summary.scalar("accuracy", acc)

    # picture enhancing factor to visualize it better on Tensorboard
    x = tf.constant(80, tf.int8)
    # enhance picture saturation by factor x
    pred_annotation_enhanced = tf.cast(pred_annotation, tf.int8) * x
    annotation_enhanced = tf.cast(annotation, tf.int8) * x

    # write image files go tensorboard
    img1 = tf.summary.image("input_image", image, max_outputs=2)
    img2 = tf.summary.image("ground_truth", tf.cast(annotation_enhanced, tf.uint8), max_outputs=2)
    img3 = tf.summary.image("pred_annotation", tf.cast(pred_annotation_enhanced, tf.uint8), max_outputs=2)

    ### END OF GRAPH DEFINITION

    trainable_var = tf.trainable_variables()
    if DEBUG:
        for var in trainable_var:
            if var is not None:
                tf.summary.histogram(var.op.name, var)
                tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))
    train_op = train(loss, trainable_var)


    print("Setting up dataset reader") 

    if LUNG_DATASET: 
        print("Setting up image reader...")
        train_records, valid_records = HUGdata.read_dataset(DATA_DIR)
        print(len(train_records))
        print(len(valid_records))

        print("training dataset reader ....")
        image_options = {'resize': False, 'resize_size': IMAGE_SIZE, 'color': False}
        train_dataset_reader = dataset2.BatchDatset(train_records, image_options)
        print("validation dataset reader ....")
        validation_dataset_reader = dataset2.BatchDatset(valid_records, image_options)

    else:
        print("training dataset reader ....")
        train_dataset_reader = dataset1.Dataset_evo_2(IMAGE_SIZE, sample_size=200, shape_count = 30, r_min=1, r_max=10, noise=False, sigma_max=5)
        print("validation dataset reader ....")
        validation_dataset_reader = dataset1.Dataset_evo_2(IMAGE_SIZE, sample_size=20, shape_count = 30, r_min=1, r_max=10, noise=False, sigma_max=5)
 
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
    writer_1 = tf.summary.FileWriter(LOGS_DIR + "/training")
    writer_2 = tf.summary.FileWriter(LOGS_DIR + "/validation")
    writer_3 = tf.summary.FileWriter(LOGS_DIR + "/accuracy")

    image_writer = tf.summary.FileWriter(LOGS_DIR)


    print("Setting up Saver...")
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")


    
    for itr in xrange(MAX_ITERATION):

        train_images, train_annotations = train_dataset_reader.next_batch(BATCH_SIZE)
        
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: keep_prob}
        
        print("Step: {}".format(itr)) 
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 10 == 0:
            # training set 

            _feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 1.0}

            train_loss, summary  = sess.run([loss, loss_summary], feed_dict=_feed_dict)
            
            print("Step: %d, Train_loss:%g" % (itr, train_loss))

            writer_1.add_summary(summary, itr)
            writer_1.flush()
            

            # validation set 
            valid_images, valid_annotations = validation_dataset_reader.get_random_batch(BATCH_SIZE)
            
            feed_dict_validation = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
        
            validation_loss, summary, y_true, y_pred  = sess.run([loss, loss_summary, annotation, pred_annotation], feed_dict=feed_dict_validation)
        
            print("Step: %d, Validation_loss:%g" % (itr, validation_loss))

            writer_2.add_summary(summary, itr)
            writer_2.flush()

            
            # accuracy
            validation_accuracy, summary = sess.run([acc, acc_summary], feed_dict=feed_dict_validation)
        
            print("Step: %d, Validation_accuracy:%g" % (itr, validation_accuracy))

            writer_3.add_summary(summary, itr)
            writer_3.flush()

            # images

            i1, i2, i3, = sess.run([img1, img2, img3], feed_dict=feed_dict_validation)
            image_writer.add_summary(i1, itr)
            image_writer.add_summary(i2, itr)
            image_writer.add_summary(i3, itr)
            image_writer.flush()
            
        if itr % 100 == 0 and itr >= 1:

            yt = y_true.flatten()
            yp = y_pred.flatten()

            print("F1-Score ---------------------------")
            print(f1_score(yt, yp, average='weighted'))


            cm = confusion_matrix(yt, yp)
            print("confusion matrix -------------------")
            print("predicted classes -->")
            print(cm)

        if itr % 500 == 0:
            saver.save(sess, LOGS_DIR + "model.ckpt", itr)

    
    
    


if __name__ == "__main__":
    tf.app.run()
