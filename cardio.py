import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv("/Users/harrysong/Downloads/cardio/cardio_train.csv",sep=";")


target = df.pop('cardio')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))



train_dataset = dataset.shuffle(len(df)).batch(1)

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='RMSprop',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)
