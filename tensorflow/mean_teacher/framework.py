# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"Tools for building Tensorflow graphs"

from contextlib import contextmanager

import tensorflow as tf


class HyperparamVariables:
    def __init__(self, hyperparams, name_or_scope=None):
        self.variables = {}
        self.placeholders = {}
        self.assign_ops = {}

        with tf.variable_scope(name_or_scope, "hyperparams"):
            for name, default in hyperparams.items():
                variable = tf.Variable(default, name=name, trainable=False)
                tf.add_to_collection("hyperparams", variable)
                placeholder = tf.placeholder(dtype=variable.dtype,
                                             shape=variable.get_shape(),
                                             name=(name + "/placeholder"))
                assign_op = tf.assign(variable, placeholder, name=(name + "/assign"))

                assert name not in self.variables
                self.variables[name] = variable
                self.placeholders[name] = placeholder
                self.assign_ops[name] = assign_op

    def __getitem__(self, name):
        "Get the TF tensor representing the hyperparameter"
        return self.variables[name]

    def get(self, session, name):
        "Get the current value of the given hyperparameter in the given session"
        return session.run(self.variables[name])

    def assign(self, session, name, value):
        "Change the value of the given hyperparameter in the given session"
        return session.run(self.assign_ops[name], {self.placeholders[name]: value})


@contextmanager
def name_variable_scope(name_scope_name,
                        var_scope_or_var_scope_name,
                        *var_scope_args,
                        **var_scope_kwargs):
    """A combination of name_scope and variable_scope with different names

    The tf.variable_scope function creates both a name_scope and a variable_scope
    with identical names. But the naming would often be clearer if the names
    of operations didn't inherit the scope name of the (reused) variables.
    So use this function to make shorter and more logical scope names in these cases.
    """
    with tf.name_scope(name_scope_name) as outer_name_scope:
        with tf.variable_scope(var_scope_or_var_scope_name,
                               *var_scope_args,
                               **var_scope_kwargs) as var_scope:
            with tf.name_scope(outer_name_scope) as inner_name_scope:
                yield inner_name_scope, var_scope


@contextmanager
def ema_variable_scope(name_scope_name, var_scope, decay=0.999):
    """Scope that replaces trainable variables with their exponential moving averages

    We capture only trainable variables. There's no reason we couldn't support
    other types of variables, but the assumed use case is for trainable variables.
    """
    with tf.name_scope(name_scope_name + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)
        }
        ema = tf.train.ExponentialMovingAverage(decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def use_ema_variables(getter, name, *_, **__):

        assert name in original_trainable_vars, "Unknown variable {}.".format(name)
        return ema.average(original_trainable_vars[name])

    with name_variable_scope(name_scope_name,
                             var_scope,
                             custom_getter=use_ema_variables) as (name_scope, var_scope):
        yield name_scope, var_scope


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.get_shape().as_list()
    error_message = "tensor {name} shape {actual} != {expected}"
    assert tensor_shape == expected_shape, error_message.format(
        name=tensor.name, actual=tensor_shape, expected=expected_shape)
