import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import cv2 as cv
from keras.utils import multi_gpu_model
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import keras.backend.tensorflow_backend as tfback

# print("tf.__version__ is", tf.__version__)
# print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


def binary_focal_loss(gamma=2, alpha=0.99):
    """
    Binary form of focal loss.
         Focal loss for binary classification problems

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def _signed_boundary_distance_batch(masks, normalize=True):
    """
    numpy helper (wrapped by tf.py_func).

    For every mask in the batch build a signed distance map ``phi`` w.r.t. the
    foreground boundary, using ``cv2.distanceTransform``:
        phi(x) < 0  inside  the foreground object
        phi(x) > 0  outside the foreground object
        phi(x) = 0  on the object boundary
    Tiles that contain a single class (no boundary) contribute nothing.

    masks: float array, shape (N, H, W, 1), values in [0, 1].
    returns: float32 array, same shape.
    """
    masks = (np.asarray(masks) > 0.5).astype(np.uint8)
    phi = np.zeros(masks.shape, dtype=np.float32)
    for i in range(masks.shape[0]):
        m = masks[i, :, :, 0]
        has_fg = m.any()
        has_bg = (m == 0).any()
        if not has_fg or not has_bg:
            continue
        dist_in = cv.distanceTransform(m, cv.DIST_L2, 3)
        dist_out = cv.distanceTransform(1 - m, cv.DIST_L2, 3)
        d = dist_out - dist_in
        if normalize:
            max_abs = np.max(np.abs(d))
            if max_abs > 0:
                d = d / max_abs
        phi[i, :, :, 0] = d
    return phi


def binary_focal_boundary_loss(gamma=2, alpha=0.99, boundary_weight=1.0,
                               normalize_distance=True):
    """
    Combined loss = binary focal loss + ``boundary_weight`` * boundary loss.

    The focal term is identical to :func:`binary_focal_loss`.

    The boundary term follows Kervadec et al., "Boundary loss for highly
    unbalanced segmentation" (https://arxiv.org/abs/1812.07032):
        L_boundary = mean( y_pred * phi(y_true) )
    where ``phi`` is the signed distance transform of the ground-truth
    boundary (negative inside the object, positive outside), computed with
    ``cv2.distanceTransform`` via ``tf.py_func``. Minimising it pushes the
    prediction to 1 inside the object and to 0 outside, with the pressure
    growing with the distance from the true contour.

    Set ``boundary_weight=0`` to recover the plain focal loss.

    Usage:
        loss = binary_focal_boundary_loss(alpha=0.2, boundary_weight=1.0)
        model.compile(loss=loss, optimizer=adam)
        # loading later:
        load_model(path, custom_objects={
            'binary_focal_boundary_loss_fixed': loss})
    """
    alpha_c = tf.constant(alpha, dtype=tf.float32)
    gamma_c = tf.constant(gamma, dtype=tf.float32)
    boundary_weight_c = tf.constant(boundary_weight, dtype=tf.float32)

    def binary_focal_boundary_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # ----- focal term (same as binary_focal_loss_fixed) -----------------
        alpha_t = y_true * alpha_c + (K.ones_like(y_true) - y_true) * (1 - alpha_c)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * \
            (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma_c) * K.log(p_t)
        focal_loss = K.mean(focal)

        # ----- boundary term ----------------------------------------------------
        phi = tf.py_func(
            lambda m: _signed_boundary_distance_batch(m, normalize_distance),
            [y_true], tf.float32, stateful=False)
        phi.set_shape(y_pred.get_shape())
        phi = tf.stop_gradient(phi)
        boundary_loss = K.mean(y_pred * phi)

        return focal_loss + boundary_weight_c * boundary_loss

    return binary_focal_boundary_loss_fixed


def unet(pretrained_weights=None, input_size=(512, 512, 1), loss_func='binary_crossentropy'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-6), loss=loss_func, metrics=['accuracy'])
    
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



def unet16(input_size=(512, 512, 3), loss_func='binary_crossentropy'):

    # Get back the convolutional part of a VGG network trained on ImageNet
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
    # vgg16 = VGG16(weights=None, include_top=False, input_shape=input_size)
    # vgg16.summary()

    # Create your own input format
    input = Input(input_size, name='image_input')

    # Use the generated model
    block1_conv1 = vgg16.get_layer("block1_conv1")(input)
    block1_conv2 = vgg16.get_layer("block1_conv2")(block1_conv1)
    block1_pool = vgg16.get_layer("block1_pool")(block1_conv2)

    block2_conv1 = vgg16.get_layer("block2_conv1")(block1_pool)
    block2_conv2 = vgg16.get_layer("block2_conv2")(block2_conv1)
    block2_pool = vgg16.get_layer("block2_pool")(block2_conv2)

    block3_conv1 = vgg16.get_layer("block3_conv1")(block2_pool)
    block3_conv2 = vgg16.get_layer("block3_conv2")(block3_conv1)
    block3_conv3 = vgg16.get_layer("block3_conv3")(block3_conv2)
    block3_pool = vgg16.get_layer("block3_pool")(block3_conv3)

    block4_conv1 = vgg16.get_layer("block4_conv1")(block3_pool)
    block4_conv2 = vgg16.get_layer("block4_conv2")(block4_conv1)
    block4_conv3 = vgg16.get_layer("block4_conv3")(block4_conv2)
    block4_pool = vgg16.get_layer("block4_pool")(block4_conv3)

    block5_conv1 = vgg16.get_layer("block5_conv1")(block4_pool)
    block5_conv2 = vgg16.get_layer("block5_conv2")(block5_conv1)
    block5_conv3 = vgg16.get_layer("block5_conv3")(block5_conv2)
    # block5_pool = vgg16.get_layer("block5_pool")(block5_conv3)

    drop5 = Dropout(0.5)(block5_conv3)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([block4_conv3, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([block3_conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([block2_conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')\
        (UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([block1_conv2, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)
    # model.summary()

    model.compile(optimizer=Adam(lr=1e-6), loss=loss_func, metrics=['accuracy'])
    return model

