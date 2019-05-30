"""Example to illustrate dataset creation from raw images
"""
import os
import sys
import tensorflow as tf

def train_prep_func(img_path, label):
  """Function to preprocess data
    Args:
      img_path(str): Image path
      label: Image label
    Returns:
      feat: Image features after preprocessing
      label: Image label
  """
  img_data = tf.read_file(img_path)
  feat = tf.image.decode_jpeg(img_data, channels=3)
  feat = tf.image.convert_image_dtype(feat, tf.float32)
  return feat, label, img_path

def get_label(img_path):
  """Extract label from image file name
    Args:
      img_path(str): Image path
    Returns:
      label(int): Label value
  """
  if isinstance(img_path, bytes):
    img_path = img_path.decode(sys.getdefaultencoding())
  fn = os.path.basename(img_path)
  cl = fn.split('_')[0]
  if cl == 'cat':
    label = 0
  else:
    label = 1

  return label


def read_img_file(img_path):
  """Function to pass into tf.py_func
    Args:
      img_path(str): Image path
    Returns:
      img_path(str): Image path
      label(int): Image label
  """
  label = get_label(img_path)
  return img_path, label

def create_dataset_from_images():
  """Example of map function
  """
  file_pattern = ["./curated_data/images/*.jpeg",
                  "./curated_data/images/*.jpg"]
  list_files = tf.gfile.Glob(file_pattern)
  labels = []
  for img_path in list_files:
    labels.append(get_label(img_path))

  dataset = tf.data.Dataset.from_tensor_slices(
    (list_files,labels)).map(train_prep_func)
  # Iterate
