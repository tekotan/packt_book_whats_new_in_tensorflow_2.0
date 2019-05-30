"""Examples to demonstrate Tensorflow Dataset API
"""
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

def construct_dataset_pandas():
  """Construct Dataset with from_tensor_slices
  using pandas DataFrame
  """
  csv_file = "./text_data/train.csv"
  df = pd.read_csv(csv_file,
                   header=None)
  df.columns = ["square_ft", "house_type", "price"]
  dataset = tf.data.Dataset.from_tensor_slices(dict(df))

  it = dataset.make_initializable_iterator()
  ne = it.get_next()
  sess = tf.Session()
  sess.run(it.initializer)
  print(dataset.output_types)
  print(dataset.output_shapes)
  print(sess.run(ne))

