# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"Functions for building neural networks with Tensorflow"

import logging

import tensorflow as tf
from tensorflow.contrib import slim

from .framework import assert_shape


LOG = logging.getLogger('main')


@slim.add_arg_scope
def gaussian_noise(inputs, scale, is_training, name=None):
    with tf.name_scope(name, 'gaussian_noise', [inputs, scale, is_training]) as scope:
        def do_add():
            noise = tf.random_normal(tf.shape(inputs))
            return inputs + noise * scale
        return tf.cond(is_training, do_add, lambda: inputs, name=scope)


@slim.add_arg_scope
def flip_randomly(inputs, horizontally, vertically, is_training, name=None):
    """Flip images randomly. Make separate flipping decision for each image.

    Args:
        inputs (4-D tensor): Input images (batch size, height, width, channels).
        horizontally (bool): If True, flip horizontally with 50% probability. Otherwise, don't.
        vertically (bool): If True, flip vertically with 50% probability. Otherwise, don't.
        is_training (bool): If False, no flip is performed.
        scope: A name for the operation.
    """
    with tf.name_scope(name, "flip_randomly") as scope:
        batch_size, height, width, _ = tf.unstack(tf.shape(inputs))
        vertical_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                            tf.to_int32(vertically) *
                            tf.to_int32(is_training))
        horizontal_choices = (tf.random_uniform([batch_size], 0, 2, tf.int32) *
                              tf.to_int32(horizontally) *
                              tf.to_int32(is_training))
        vertically_flipped = tf.reverse_sequence(inputs, vertical_choices * height, 1)
        both_flipped = tf.reverse_sequence(vertically_flipped, horizontal_choices * width, 2)
        return tf.identity(both_flipped, name=scope)



@slim.add_arg_scope
def random_translate(inputs, scale, is_training,
                     padding_mode='REFLECT', name='random_translate'):
    """Translate images by a random number of pixels
    The dimensions of the image tensor remain the same. Padding is added where necessary, and the
    pixels outside image area are cropped off.
    For performance reasons, the offset values need to be integers and not Tensors.
    Args:
        inputs (4-D tensor): Input images (batch size, height, width, channels).
        scale (integer): Maximum translation in pixels. For each image on the batch, a random
            2-D translation is picked uniformly from ([-scale, scale], [-scale, scale]).
        is_training (bool): If False, no translation is performed.
        padding_mode (string): Either 'CONSTANT', 'SYMMETRIC', or 'REFLECT'. What values to use for
            pixels that are translated from outside the original image. This parameter is passed
            directly to tensorflow.pad fuction.
        scope: A name for the operation.
    """
    assert isinstance(scale, int)

    with tf.name_scope(name) as scope:
        def random_offsets(batch_size, minval, inclusive_maxval, name='random_offsets'):
            with tf.name_scope(name) as scope:
                return tf.random_uniform([batch_size],
                                         minval=minval, maxval=inclusive_maxval + 1,
                                         dtype=tf.int32, name=scope)

        def do_translate(name='do_translate'):
            with tf.name_scope(name) as scope:
                batch_size = tf.shape(inputs)[0]
                offset_heights = random_offsets(batch_size, -scale, scale, 'offset_heights')
                offset_widths = random_offsets(batch_size, -scale, scale, 'offset_widths')
                return translate(inputs, offset_heights, offset_widths,
                                 scale, padding_mode, name=scope)

        return tf.cond(is_training, do_translate, lambda: inputs, name=scope)


def translate(inputs, vertical_offsets, horizontal_offsets, scale, padding_mode, name='translate'):
    """Translate images

    The dimensions of the image remain the same. Padding is added where necessary, and the
    pixels outside image area are cropped off.

    Args:
        inputs (4-D tensor): Input images (batch size, height, width, channels).
        vertical_offsets (1-D tensor of integers): Vertical translation in pixels for each image.
        horizontal offsets (1-D tensor of integers): Horizontal translation in pixels.
        scale (integer): Maximum absolute offset (needed for performance reasons).
        padding_mode (string): Either 'CONSTANT', 'SYMMETRIC', or 'REFLECT'. What values to use for
            pixels that are translated from outside the original image. This parameter is passed
            directly to tensorflow.pad fuction.
    """
    assert isinstance(scale, int)
    kernel_size = 1 + 2 * scale
    batch_size, inp_height, inp_width, channels = inputs.get_shape().as_list()

    def one_hots(offsets, name='one_hots'):
        with tf.name_scope(name) as scope:
            with tf.control_dependencies([tf.assert_less_equal(tf.abs(offsets), scale)]):
                result = tf.expand_dims(tf.one_hot(scale - offsets, kernel_size), 1, name=scope)
                assert_shape(result, [batch_size, 1, kernel_size])
                return result

    def assert_equal_first_dim(tensor_a, tensor_b, name='assert_equal_first_dim'):
        with tf.name_scope(name) as scope:
            first_dims = tf.shape(tensor_a)[0], tf.shape(tensor_b)[0]
            return tf.Assert(tf.equal(*first_dims), first_dims, name=scope)

    with tf.name_scope(name) as scope:
        with tf.control_dependencies([
            assert_equal_first_dim(inputs, vertical_offsets, "assert_height"),
            assert_equal_first_dim(inputs, horizontal_offsets, "assert_width")
        ]):
            filters = tf.matmul(one_hots(vertical_offsets),
                                one_hots(horizontal_offsets),
                                adjoint_a=True)
            assert_shape(filters, [batch_size, kernel_size, kernel_size])

            padding_sizes = [[0, 0], [scale, scale], [scale, scale], [0, 0]]
            padded_inp = tf.pad(inputs, padding_sizes, mode=padding_mode)
            assert_shape(padded_inp,
                         [batch_size, inp_height + 2 * scale, inp_width + 2 * scale, channels])

            depthwise_inp = tf.transpose(padded_inp, perm=[3, 1, 2, 0])
            assert_shape(depthwise_inp,
                         [channels, inp_height + 2 * scale, inp_width + 2 * scale, batch_size])

            depthwise_filters = tf.expand_dims(tf.transpose(filters, [1, 2, 0]), -1)
            assert_shape(depthwise_filters, [kernel_size, kernel_size, batch_size, 1])

            convoluted = tf.nn.depthwise_conv2d_native(depthwise_inp, depthwise_filters,
                                                       strides=[1, 1, 1, 1], padding='VALID')
            assert_shape(convoluted, [channels, inp_height, inp_width, batch_size])

            result = tf.transpose(convoluted, (3, 1, 2, 0), name=scope)
            assert_shape(result, [batch_size, inp_height, inp_width, channels])

            return result


def lrelu(inputs, leak=0.1, name=None):
    with tf.name_scope(name, 'lrelu') as scope:
        return tf.maximum(inputs, leak * inputs, name=scope)


def adam_optimizer(cost, global_step,
                   learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                   name=None):
    with tf.name_scope(name, "adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        return optimizer.minimize(cost, global_step=global_step, name=scope)
