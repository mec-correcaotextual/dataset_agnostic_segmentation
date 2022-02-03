"""
Various tensorflow utilities
"""

import tensorflow as tf


def weighted_sigmoid(predictions, raw_labels, w_pos, w_neg):
    with tf.compat.v1.name_scope('weighted_sigmoid'):
        lab_shape = tf.compat.v1.shape(raw_labels)
        w_pos = tf.compat.v1.fill(lab_shape, w_pos)
        w_neg = tf.compat.v1.fill(lab_shape, w_neg)
        loss_weights = tf.compat.v1.where(tf.compat.v1.greater(raw_labels, 0.), w_pos, w_neg)
        labels = tf.compat.v1.cast(raw_labels, tf.compat.v1.float32)
        loss = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions, name='unpacked')
        loss = tf.compat.v1.multiply(loss_weights, loss, name='weighted_loss')
    return loss


def weighted_l1(predictions, images):
    with tf.compat.v1.name_scope('weighted_sigmoid'):
        loss = tf.compat.v1.compat.v1.losses.absolute_difference(images, predictions, weights=1.0, scope='weighted_loss')
    return loss


def weighted_xent_with_reshape(predictions, raw_labels, w_pos, w_neg, with_border=False):
    with tf.compat.v1.name_scope('weighted_xent_with_reshape'):
        raw_labels = tf.compat.v1.reshape(raw_labels, [-1])
        lab_shape = tf.compat.v1.shape(raw_labels)
        w_pos = tf.compat.v1.fill(lab_shape, w_pos)
        w_neg = tf.compat.v1.fill(lab_shape, w_neg)
        loss_weights = tf.compat.v1.where(tf.compat.v1.greater(raw_labels, 0.), w_pos, w_neg)
        logits = tf.compat.v1.reshape(predictions, [-1, 2]) if not with_border else tf.compat.v1.reshape(predictions, [-1, 3])
        labels = tf.compat.v1.cast(raw_labels, tf.compat.v1.int32)
        xent = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='unpacked_cls_loss')
        xent = tf.compat.v1.multiply(loss_weights, xent, name='weighted_loss')
    return xent


def smooth_l1(x, y):
    with tf.compat.v1.name_scope('smooth_l1'):
        diff = tf.compat.v1.subtract(x, y, name='diff')
        l1 = tf.compat.v1.abs(x - y, name='abs_diff')
        l1_smooth = tf.compat.v1.where(tf.compat.v1.greater(l1, 1), tf.compat.v1.subtract(l1, 0.5), tf.compat.v1.multiply(0.5, tf.compat.v1.square(diff)), name='loss')
    return l1_smooth


def weighted_loss_with_reshape(loss_vector, pos_neg_indicator, w_pos, w_neg):
    with tf.compat.v1.name_scope('weighted_loss_with_reshape'):
        pos_neg_indicator = tf.compat.v1.reshape(pos_neg_indicator, [-1])
        lv_shape = tf.compat.v1.shape(loss_vector)
        w_pos = tf.compat.v1.fill(lv_shape, w_pos)
        w_neg = tf.compat.v1.fill(lv_shape, w_neg)
        loss_weights = tf.compat.v1.where(tf.compat.v1.greater(pos_neg_indicator, 0.), w_pos, w_neg)
        weighted_loss_vector = tf.compat.v1.multiply(loss_weights, loss_vector, name='weighted_loss')
    return weighted_loss_vector
