""" Class for house data preparation, feature engineering etc."""
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

class Cifar10Data(object):
  """ Cifar10Data class
    Description: 
      A lot of code is taken from 
      models/tutorials/image/cifar10_estimator/
  """
  def __init__(self, batch_size, use_convnet=False, tfrecords_path='./tfrecords'):
    self.batch_size = batch_size
    self.use_convnet = use_convnet
    self.tfrecords_path = tfrecords_path
    self.image_height = 32
    self.image_width = 32
    self.image_num_channels = 3
    self.train_data_size = 40000 # TODO fix hard coding
    self.eval_data_size = 10000 # TODO fix hard coding
    self.test_data_size = 10000 # TODO fix hard coding

    # Create datasets
    self.train_dataset = self.create_datasets(is_train_or_eval_or_test='train')
    self.eval_dataset = self.create_datasets(is_train_or_eval_or_test='eval')
    self.test_dataset = self.create_datasets(is_train_or_eval_or_test='test')

    # self.feature_normalization()

    # Create reinitializable iterator and use for train, eval and test
    iterator = tf.data.Iterator.from_structure(
        self.train_dataset.output_types,
        self.train_dataset.output_shapes)

    # features, labels from iter.get_net()
    self.features, self.labels = iterator.get_next()

    self.train_iter_op = iterator.make_initializer(self.train_dataset)
    self.eval_iter_op = iterator.make_initializer(self.eval_dataset)
    self.test_iter_op = iterator.make_initializer(self.test_dataset)

  def feature_normalization(self):

    def normalization1(self, tensor_in):
      return tensor_in / 255.

    self.train_dataset = self.train_dataset.map(normalization1)
    self.eval_dataset = self.eval_dataset.map(normalization1)
    self.test_dataset = self.test_dataset.map(normalization1)

    def _batch_normalization(tensor_in, epsilon=.0001):
      mean, variance = tf.nn.moments(tensor_in, axes=[0])
      print(mean)
      scale = tf.Variable(tf.ones([3072]))
      beta = tf.Variable(tf.zeros([3072]))
      tensor_normalized = tf.nn.batch_normalization(
          tensor_in, mean, variance, scale, beta, epsilon)
      return tensor_normalized

    self.train_dataset = self.train_dataset.map(_batch_normalization)

  def create_datasets(self, is_train_or_eval_or_test='train'):
    """ function to create trai
    """
    if is_train_or_eval_or_test is 'train':
      tfrecords_files = glob.glob(self.tfrecords_path + '/train/*')
    elif is_train_or_eval_or_test is 'eval':
      tfrecords_files = glob.glob(self.tfrecords_path + '/eval/*')
    elif is_train_or_eval_or_test is 'test':
      tfrecords_files = glob.glob(self.tfrecords_path + '/test/*')

    # Read dataset from tfrecords
    dataset = tf.data.TFRecordDataset(tfrecords_files)

    # Decode/parse 
    dataset = dataset.map(self._parse_and_decode, num_parallel_calls=self.batch_size)

    if is_train_or_eval_or_test is 'train':
      # For training dataset, do a shuffle and repeat
      dataset = dataset.shuffle(10000).repeat().batch(self.batch_size)
    else:
      # Just create batches for eval and test
      dataset = dataset.batch(self.batch_size)
    return dataset

  def _parse_and_decode(self, serialized_example):
    features = tf.parse_single_example(
        serialized_example, 
        features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          }
        )
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    # image = image / 255.
    image.set_shape([self.image_num_channels * self.image_height * self.image_width])

    if self.use_convnet is True:
      # reshape to channels, height, width and 
      # transpose to have width, height, and channels
      image = tf.cast(
          tf.transpose(tf.reshape(
            image, [self.image_num_channels, self.image_height, self.image_width]), 
            [1, 2, 0]), tf.float32)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 10)

    return image, label

def main(argv):
  """ main routing to use LinearRegression model """
  if argv is not None:
    print('argv: {}'.format(argv))

  batch_size = 128
  tfrecords_path = './tfrecords'
  # cifar_data = Cifar10Data(batch_size, True, tfrecords_path)
  cifar_data = Cifar10Data(batch_size, False, tfrecords_path)

  images, classes = cifar_data.train_dataset.make_one_shot_iterator().get_next()

  with tf.Session() as sess:
    np_images = sess.run(images)
    print('shape %', np_images.shape)
    np_classes = sess.run(classes)
    print('shape %', np_classes.shape)
    import ipdb; ipdb.set_trace()
    plt.imshow(np.resize(np_images[0], [32, 32, 3]), interpolation='nearest')
    plt.show()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
