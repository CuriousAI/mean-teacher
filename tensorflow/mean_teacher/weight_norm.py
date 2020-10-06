# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Layers with weight normalization and mean-only batch normalization

See https://arxiv.org/abs/1602.07868 (Salimans & Kingma, 2016)

The code is adapted from
https://github.com/openai/pixel-cnn/blob/fc86dbce1d508fa79f8e9a7d1942d229249a5366/pixel_cnn_pp/nn.py
"""

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


@add_arg_scope
def fully_connected(inputs, num_outputs,
                    activation_fn=None, init_scale=1., init=False,
                    eval_mean_ema_decay=0.999, is_training=None, scope=None):
  
    with tf.variable_scope(scope, "fully_connected"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V',
                                [int(inputs.get_shape()[1]), num_outputs],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(inputs, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros_like(m_init), trainable=True)
            x_init = tf.reshape(
                scale_init, [1, num_outputs]) * (x_init - tf.reshape(m_init, [1, num_outputs]))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, g, b = [tf.get_variable(var_name) for var_name in ['V', 'g', 'b']]

            # use weight normalization (Salimans & Kingma, 2016)
            inputs = tf.matmul(inputs, V)
            training_mean = tf.reduce_mean(inputs, [0])

            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            inputs = tf.reshape(scaler, [1, num_outputs]) * \
                inputs + tf.reshape(b, [1, num_outputs])

            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs


@add_arg_scope
def conv2d(inputs, num_outputs,
           kernel_size=[3, 3], stride=[1, 1], padding='SAME',
           activation_fn=None, init_scale=1., init=False,
           eval_mean_ema_decay=0.999, is_training=None, scope=None):
  
    with tf.variable_scope(scope, "conv2d"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', kernel_size + [int(inputs.get_shape()[-1]), num_outputs],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(inputs, V_norm, [1] + stride + [1], padding)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros_like(m_init), trainable=True)
            x_init = (tf.reshape(scale_init, [1, 1, 1, num_outputs]) *
                      (x_init - tf.reshape(m_init, [1, 1, 1, num_outputs])))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init

        else:
            V, g, b = [tf.get_variable(var_name) for var_name in ['V', 'g', 'b']]

            # use weight normalization (Salimans & Kingma, 2016)
            W = (tf.reshape(g, [1, 1, 1, num_outputs]) *
                 tf.nn.l2_normalize(V, [0, 1, 2]))

            # calculate convolutional layer output
            inputs = tf.nn.conv2d(inputs, W, [1] + stride + [1], padding)
            training_mean = tf.reduce_mean(inputs, [0, 1, 2])

            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean

            inputs = tf.nn.bias_add(inputs, b)

            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs
