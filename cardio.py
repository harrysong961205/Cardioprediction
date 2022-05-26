# random forest 모듈을 윈도우와 맥에서는 사용할 수 없다! 실패!
import numpy as np
import pandas as pd
import tensorflow as tf


df = pd.read_csv("/Users/harrysong/Downloads/cardio/cardio_train.csv",sep=";")



print(df.info())

