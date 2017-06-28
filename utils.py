import pandas as pd
import numpy as np
import os
import matplotlib
import tensorflow as tf
from PIL import Image, ImageDraw

def load_dataset():
    if(not os.path.exists("./dataset/training.csv")):
        print("dataset does not exist")
        raise Exception

    #load dataset
    labeled_image = pd.read_csv("./dataset/training.csv")

    #preprocessing dataframe
    image = np.array(labeled_image["Image"].values).reshape(-1,1)
    image = np.apply_along_axis(lambda img: (img[0].split()),1,image)
    image = image.astype(np.int32) #because train_img elements are string before preprocessing
    image = image.reshape(-1,96*96) # data 96 * 96 size image

    label = labeled_image.values[:,:-1]
    label = label.astype(np.float32)

    #nan value to mean value
    col_mean = np.nanmean(label, axis=0)
    indices = np.where(np.isnan(label))
    label[indices] = np.take(col_mean, indices[1])

    return image, label

def split_data(image, label, train_rate=0.8, valid_rate=0.1, test_rate=0.1):
    if(train_rate + valid_rate + test_rate!= 1):
        print("train_rate plus valid_rate plus test_rate must be one")
        raise Exception
    #shuffle data
    arange_idx = np.arange(0, image.shape[0], 1)
    np.random.shuffle(arange_idx)

    train_idx = arange_idx[:int(train_rate*image.shape[0])]
    valid_idx = arange_idx[int(train_rate*image.shape[0]):int((train_rate + valid_rate)*image.shape[0])]
    test_idx = arange_idx[int((train_rate + valid_rate)*image.shape[0]):]

    train_X = image[train_idx]
    train_y = label[train_idx]
    valid_X = image[valid_idx]
    valid_y = label[valid_idx]
    test_X  = image[test_idx]
    test_y  = label[test_idx]

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)

def horizontal_flip(imgs, labels):
    if imgs.ndim != 2:
        print("Image dimension must be 2")
        raise Exception

    if imgs.shape[0] != labels.shape[0]:
        print("Images num and labels num must be equal")
        raise Exception

    #flip the img horizontally
    imgs = imgs.reshape(-1, 96, 96)
    imgs = np.flip(imgs, axis=2)
    imgs = imgs.reshape(-1, 96*96)

    #when flip the image horizontally, img's ypos does not change but xpos reflect on img's center pos
    result = np.copy(labels)

    for idx in range(labels.shape[0]):
        result[idx][::2] = 96 - result[idx][::2]

    return imgs, labels


def patch_pointing(fig, point_pos, size=0.2):
    if point_pos.size % 2 != 0:
        raise Exception

    for idx in range(point_pos.size//2):
        fig.gca().add_patch(matplotlib.patches.Circle((point_pos[2*idx], point_pos[2*idx+1]), size, ec='r', fc='none'))

    return fig

def brighten(imgs, labels):
    if imgs.ndim != 2:
        print("Image dimension must be 2")
        raise Exception

    if imgs.shape[0] != labels.shape[0]:
        print("Images num and labels num must be equal")
        raise Exception

    imgs = imgs + 5
    imgs = np.clip(imgs, 0, 255)

    return imgs, labels

def darken(imgs, labels):
    if imgs.ndim != 2:
        print("Image dimension must be 2")
        raise Exception

    if imgs.shape[0] != labels.shape[0]:
        print("Images num and labels num must be equal")
        raise Exception

    imgs = imgs - 5
    imgs = np.clip(imgs, 0, 255)

    return imgs, labels

def image_augmentation(image, label, horizon_flip=True, control_brightness=True):
    if image.ndim != 2:
        print("Image dimension must be 2")
        raise Exception

    if image.shape[0] != label.shape[0]:
        print("Images num and labels num must be equal")
        raise Exception


    if horizon_flip:
        flipped_img, flipped_label = horizontal_flip(image, label)
        image = np.concatenate((image, flipped_img))
        label = np.concatenate((label, flipped_label))

    if control_brightness:
        brightened_img, brightened_label = brighten(image, label)
        darkened_img, darkened_label = darken(image, label)

        image = np.concatenate((image, brightened_img, darkened_img))

        label = np.concatenate((label, brightened_label, darkened_label))

    image = image / 255.

    return image, label

def normalize(input_data):
    pass

def my_Model(input_x, input_dim, hidden_dim, output_dim , keep_prob=0.8 ,  name="MyModel"):

        W1 = tf.get_variable(name+"_w1", shape=[input_dim*input_dim, hidden_dim], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(input_dim)))
        b1 = tf.get_variable(name+"_b1", shape=[hidden_dim], initializer=tf.constant_initializer(0.))

        c1 = tf.matmul(input_x, W1) + b1
        c1 = tf.nn.relu(c1)
        c1 = tf.nn.dropout(c1, keep_prob)

        W2 = tf.get_variable(name+"_w2", shape=[hidden_dim, hidden_dim*2], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(hidden_dim)))
        b2 = tf.get_variable(name+"_b2", shape=[hidden_dim*2], initializer=tf.constant_initializer(0.))

        c2 = tf.matmul(c1, W2) + b2
        c2 = tf.nn.relu(c2)
        c2 = tf.nn.dropout(c2, keep_prob)

        W3 = tf.get_variable(name+"_w3", shape=[hidden_dim*2, hidden_dim*4], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(hidden_dim*2)))
        b3 = tf.get_variable(name+"_b3", shape=[hidden_dim*4], initializer=tf.constant_initializer(0.))

        c3 = tf.matmul(c2, W3) + b3
        c3 = tf.nn.relu(c3)
        c3 = tf.nn.dropout(c3, keep_prob)

        W4 = tf.get_variable(name+"_w4", shape=[hidden_dim*4, hidden_dim*8], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(hidden_dim*4)))
        b4 = tf.get_variable(name+"_b4", shape=[hidden_dim*8], initializer=tf.constant_initializer(0.))

        c4 = tf.matmul(c3, W4) + b4
        c4 = tf.nn.relu(c4)
        c4 = tf.nn.dropout(c4, keep_prob)

        W5 = tf.get_variable(name+"_w5", shape=[hidden_dim*8, output_dim], initializer=tf.truncated_normal_initializer(stddev=2/np.sqrt(hidden_dim*8)))
        b5 = tf.get_variable(name+"_b5", shape=[output_dim], initializer=tf.constant_initializer(0.))

        logits = tf.matmul(c4, W5) + b5

        return logits

def draw_features_point_on_image(image, features, src_width, src_height, rgba=(50,255,50,255), depth=3, point_thickness=3):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    for feature in features:
        for x, y in feature.reshape(-1,2):
            draw.ellipse([x, y, x+point_thickness, y+point_thickness], fill=rgba)

    return np.array(img.getdata(), np.uint8).reshape(src_height, src_width, depth)
