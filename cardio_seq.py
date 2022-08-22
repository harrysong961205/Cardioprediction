import numpy as np
import pandas as pd

dataset = pd.read_csv(
    "/Users/harrysong/Downloads/cardio/cardio_train.csv", sep=";")


# 3. X/y 나누기
# 3. X/y 나누기
train_list = []
epoch_level = [500]
batch_level = [200]
acc_list = []
for a in range(0, len(batch_level)):
    dataset_select = 70000-70000 % batch_level[a]
    X = dataset.iloc[:dataset_select, 1:-1]
    y = dataset.iloc[:dataset_select, -1]
    from sklearn.preprocessing import MinMaxScaler

    print(X.head(10))
    print(y.head(10))

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    print(X[:5])
    print(y[:5])

    print(X.shape)
    print(y.shape)

    # 4. Train set, Test set 나누기
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=9)

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=123)

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
    drop_rate = 0.03
    model.add(Dense(64, input_dim=11, activation='relu'))
    # model.add(keras.layers.Dropout(drop_rate))
    for b in range(10):
        model.add(Dense(512, activation='relu'))

    # model.add(keras.layers.Dropout(drop_rate))
    # model.add(keras.layers.Dropout(drop_rate))

    model.add(Dense(1, activation='sigmoid'))

    # 6-2 . early_stopping
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    mc = ModelCheckpoint('model', monitor='accuracy',
                         mode='max', save_best_only=True)
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0003, patience=50, mode='auto')

    print(model.summary())

    # 7. Compile - Optimizer, Loss function 설정
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # 8. 학습시키기

    batch_size = batch_level[a]
    epochs = epoch_level[a]

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val), shuffle=True, verbose=1,
                        callbacks=[earlystopping])

    # 9. 모델 평가하기a
    train_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)
    print("batch_zie: ", batch_size, "epochs: ", epochs)
    print(train_accuracy)
    print(test_accuracy)

    acc_list.append(a)
    acc_list.append(train_accuracy)
    acc_list.append(test_accuracy)

    # 10. 학습 시각화하기
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
