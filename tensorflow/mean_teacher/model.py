"Mean teacher model"

import logging
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from . import weight_norm as wn
from .framework import ema_variable_scope, name_variable_scope, assert_shape, HyperparamVariables


LOG = logging.getLogger('main')


class Model:
    DEFAULT_HYPERPARAMS = {
        # Consistency hyperparameters
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_coefficient': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,
    }

    #pylint: disable=too-many-instance-attributes
    def __init__(self, result_dir):
        self.checkpoint_dir = os.path.join(result_dir, 'checkpoints')
        self.summary_dir = os.path.join(result_dir, 'summaries')
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.summary_dir)

        with tf.name_scope("placeholders"):
            self.images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='images')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFAULT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_coefficient'],
                                                name='consistency_coefficient')

            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')

        self.logits_1, self.logits_2, self.logits_ema = inference(
            self.images,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'])

        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.logits_ema, self.labels)

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.logits_1, self.labels)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.logits_ema, self.labels)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_pi, self.cons_costs_pi = consistency_costs(
                self.logits_1, self.logits_2, self.cons_coefficient, consistency_mask)
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.logits_1, self.logits_ema, self.cons_coefficient, consistency_mask)

            self.mean_total_cost_pi, self.total_costs_pi = total_costs(
                self.class_costs_1, self.cons_costs_pi)
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt)

            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])

        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])

        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/pi": self.mean_cons_cost_pi,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/total_cost/pi": self.mean_total_cost_pi,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),

                # Note that the evaluation costs are not directly comparable
                # to the training costs. Training batches contain unlabeled
                # samples but the evaluation batches do not. Because
                # classification cost is zero for unlabeled samples, the
                # training costs are smaller than evaluation costs when
                # doing semi-supervised learning.
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
                "eval/cons_cost/pi": streaming_mean(self.cons_costs_pi),
                "eval/cons_cost/mt": streaming_mean(self.cons_costs_mt),
                "eval/total_cost/pi": streaming_mean(self.total_costs_pi),
                "eval/total_cost/mt": streaming_mean(self.total_costs_mt)
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.run(self.init_init_op)

    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn):
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            results, _ = self.run([self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch))
            step_control = self.get_training_control()
            if step_control['time_to_print']:
                LOG.info("step %d: %r", step_control['step'], {**results, **step_control})
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        LOG.info("step %d validation: %r", step, {**results, 'step': step})

    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.images: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def save_checkpoint(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")
        path = self.saver.save(self.session, checkpoint_prefix, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.summary_dir)
        writer.add_graph(self.session.graph)
        return writer.get_logdir()


Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])


def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }


def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")


def inference(inputs, is_training, ema_decay, normalize_input, flip_horizontally):
    with tf.variable_scope("initialization") as var_scope:
        _ = tower(inputs,
                  is_training,
                  normalize_input=normalize_input,
                  flip_horizontally=flip_horizontally,
                  is_initialization=True)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        output1 = tower(inputs,
                        is_training,
                        normalize_input=normalize_input,
                        flip_horizontally=flip_horizontally,
                        name=name_scope)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        output2 = tower(inputs,
                        is_training,
                        normalize_input=normalize_input,
                        flip_horizontally=flip_horizontally,
                        name=name_scope)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        ema_output = tower(inputs,
                           is_training,
                           normalize_input=normalize_input,
                           flip_horizontally=flip_horizontally)
    return output1, output2, tf.stop_gradient(ema_output)


def tower(inputs,
          is_training,
          normalize_input,
          flip_horizontally,
          is_initialization=False,
          name=None):
    with tf.name_scope(name, "tower"):
        default_conv_args = dict(
            padding='SAME',
            kernel_size=[3, 3],
            activation_fn=nn.lrelu,
            init=is_initialization
        )
        training_mode_funcs = [
            nn.random_translate, nn.flip_randomly, nn.gaussian_noise, slim.dropout,
            wn.fully_connected, wn.conv2d
        ]
        training_args = dict(
            is_training=is_training
        )

        with \
        slim.arg_scope([wn.conv2d], **default_conv_args), \
        slim.arg_scope(training_mode_funcs, **training_args):
            #pylint: disable=no-value-for-parameter
            net = inputs
            assert_shape(net, [None, 32, 32, 3])

            net = tf.cond(normalize_input,
                          lambda: slim.layer_norm(net,
                                                  scale=False,
                                                  center=False,
                                                  scope='normalize_inputs'),
                          lambda: net)
            assert_shape(net, [None, 32, 32, 3])

            net = nn.flip_randomly(net,
                                   horizontally=flip_horizontally,
                                   vertically=False,
                                   name='random_flip')
            net = nn.random_translate(net, scale=2, name='random_translate')
            net = nn.gaussian_noise(net, scale=0.15, name='gaussian_noise')

            net = wn.conv2d(net, 128, scope="conv_1_1")
            net = wn.conv2d(net, 128, scope="conv_1_2")
            net = wn.conv2d(net, 128, scope="conv_1_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_1')
            net = slim.dropout(net, 0.5, scope='dropout_1')
            assert_shape(net, [None, 16, 16, 128])

            net = wn.conv2d(net, 256, scope="conv_2_1")
            net = wn.conv2d(net, 256, scope="conv_2_2")
            net = wn.conv2d(net, 256, scope="conv_2_3")
            net = slim.max_pool2d(net, [2, 2], scope='max_pool_2')
            net = slim.dropout(net, 0.5, scope='dropout_2')
            assert_shape(net, [None, 8, 8, 256])

            net = wn.conv2d(net, 512, padding='VALID', scope="conv_3_1")
            assert_shape(net, [None, 6, 6, 512])
            net = wn.conv2d(net, 256, kernel_size=[1, 1], scope="conv_3_2")
            net = wn.conv2d(net, 128, kernel_size=[1, 1], scope="conv_3_3")
            net = slim.avg_pool2d(net, [6, 6], scope='avg_pool')
            assert_shape(net, [None, 1, 1, 128])

            net = slim.flatten(net)
            assert_shape(net, [None, 128])
            net = wn.fully_connected(net, 10, init=is_initialization, scope="fully_connected")
            assert_shape(net, [None, 10])
            return net


def errors(logits, labels, name=None):
    """Compute error mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over unlabeled examples.
    Mean error is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, name=None):
    with tf.name_scope(name, "consistency_costs") as scope:
        assert_shape(logits1, [None, 10])
        assert_shape(logits2, [None, 10])
        assert_shape(cons_coefficient, [])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)
        costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1) * cons_coefficient
        costs = costs * tf.to_float(mask)
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(class_costs, cons_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        assert_shape(class_costs, [None])
        assert_shape(cons_costs, [None])
        costs = class_costs + cons_costs
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs
