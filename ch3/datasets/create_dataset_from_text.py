""" Example code to read csv, build dataset and iterate"""
import tensorflow as tf

def train_decode_line(row):
  """ function to decode a row from csv file
  Args:
    row (str): line from csv
  Returns:
    dict of tensor, tensor: features_columns, label
  """
  cols = tf.decode_csv(row, record_defaults=[[0.], ['house'], [0.]])
  myfeatures = {'sq_footage':cols[0], 'type':cols[1]}
  mylabel = cols[2] #price
  return myfeatures, mylabel

def predict_decode_line(row):
  """ function to decode a row from csv file
  Args:
    row (str): line from csv
  Returns:
    dict of tensor: features_columns
  """
  cols = tf.decode_csv(row, record_defaults=[[0.], ['house']])
  myfeatures = {'sq_footage':cols[0], 'type':cols[1]}
  return myfeatures

train_dataset = tf.data.\
    TextLineDataset('text_data/train.csv').map(train_decode_line)
train_dataset = train_dataset.batch(3)
train_dataset_iter = train_dataset.make_one_shot_iterator()

predict_dataset = tf.data.\
    TextLineDataset('text_data/predict.csv').map(
		predict_decode_line)
predict_dataset = predict_dataset.batch(3)
predict_dataset_iter = predict_dataset.make_one_shot_iterator()

with tf.Session() as sess:
  # training
  print('Training data...')
  features, label = train_dataset_iter.get_next()
  print(sess.run([features, label]))
  features, label = train_dataset_iter.get_next()
  print(sess.run([features, label]))

  # predict
  print('\n\nPredict data...')
  features = predict_dataset_iter.get_next()
  print(sess.run(features))
  features = predict_dataset_iter.get_next()
  print(sess.run(features))
