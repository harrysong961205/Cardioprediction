import numpy as np
import pandas as pd

dataset = pd.read_csv("/Users/harrysong/Downloads/cardio/cardio_train.csv",sep=";")



# 3. X/y 나누기
X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]



print(X.head(10))
print(y.head(10))


print(X.shape)
print(y.shape)

# 4. Train set, Test set 나누기
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=9)

X_val,X_test,y_val,y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)

print(X_test.shape)
print(y_test.shape)

 
# 5. Keras 패키지 가져오기
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

print(keras.__version__)

# 6. MLP 모델 생성
model = Sequential()

model.add(Dense(20, input_dim=11, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# 7. Compile - Optimizer, Loss function 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 6. 학습시키기

batch_size = 200
epochs = 500

history = model.fit(X_train, y_train, epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, y_val), shuffle=True, verbose=1)
                    
                    
# 7. 모델 평가하기
train_accuracy = model.evaluate(X_train, y_train)
test_accuracy = model.evaluate(X_test, y_test)

print(train_accuracy)
print(test_accuracy)

 
# 10. 학습 시각화하기
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('epoch')
plt.xlabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('epoch')
plt.xlabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
