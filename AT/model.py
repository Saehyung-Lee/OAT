# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
  """Wide ResNet model."""

  def __init__(self, num_classes):
    """ResNet constructor.
    """
    self.num_classes = num_classes
    self._build_model()

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return (stride, stride)

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      self.x_input = tf.placeholder(
        tf.float32,
        shape=[None, 32, 32, 3])

      self.y_input = tf.placeholder(tf.float32, shape=[None, self.num_classes])
      self.scaler = tf.placeholder(tf.float32, shape=None)
      self.is_training = tf.placeholder(tf.bool, shape=None)


      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      input_standardized = tf.transpose(input_standardized, [0, 3, 1, 2])
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    #filters = [16, 128, 256, 512]
    filters = [16, 160, 320, 640]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, self.num_classes)

    self.single_label = tf.cast(tf.argmax(self.y_input, axis=1), tf.int64)
    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.single_label)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    self.tr_correct_prediction, _ = tf.split(self.correct_prediction, 2, axis=0)
    self.tr_num_correct = tf.reduce_sum(
        tf.cast(self.tr_correct_prediction, tf.int64))
    self.tr_accuracy = tf.reduce_mean(
        tf.cast(self.tr_correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent_before_scale = tf.nn.softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.y_xent = self.y_xent_before_scale * self.scaler
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.layers.batch_normalization(
          inputs=x,
          momentum=.9,
          epsilon=1e-5,
          center=True,
          scale=True,
          axis=1,
          training=self.is_training)

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, self._stride_arr(1))

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._avg_pool(orig_x, stride, stride)
        orig_x = tf.pad(
            orig_x, [[0, 0], [(out_filter-in_filter)//2, (out_filter-in_filter)//2], [0, 0], [0, 0]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if 'kernel' in var.name:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      init = tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
      layer = tf.layers.Conv2D(
          out_filters,
          kernel_size=filter_size,
          strides=strides,
          padding='same',
          data_format='channels_first',
          dilation_rate=(1,1),
          use_bias=False,
          kernel_initializer=init)
      return layer.apply(x)

  def _avg_pool(self, x, size, strides):
      return tf.layers.average_pooling2d(x, size, strides, 'valid', data_format='channels_first')


  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _batch_flatten(self, x):
      """
      Flatten the tensor except the first dimension.
      """
      shape = x.get_shape().as_list()[1:]
      if None not in shape:
          return tf.reshape(x, [-1, int(np.prod(shape))])
      return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

  def _fully_connected(self, x, out_dim):
      """FullyConnected layer for final output."""
      inputs = self._batch_flatten(x)
      init = tf.uniform_unit_scaling_initializer(factor=1.0)
      layer = tf.layers.Dense(
          units=out_dim,
          use_bias=True,
          kernel_initializer=init,
          bias_initializer=tf.constant_initializer())
      return layer.apply(inputs)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [2, 3])
