import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv("/Users/harrysong/Downloads/cardio/cardio_train.csv",sep=";")


print(df.head())
target = df.pop('cardio')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))


for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

train_dataset = dataset.shuffle(len(df)).batch(1)
