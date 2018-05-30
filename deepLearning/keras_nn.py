#coding:utf-8
'''
模块设计目的：使用keras 快速构建网络结构
1.简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
2.支持CNN和RNN，或二者的结合
3.无缝CPU和GPU切换

@author:Jeeker
'''

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.utils import to_categorical

#使用Keras库下搭建的DNN神经网络模型
def Kmodel_1(x_train, y_train, x_test,y_test):
    from sklearn import preprocessing
    #将y对应的标签需要转换为y矩阵
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    print(y_train.shape,type(y_train))
    #定义DNN模型的网络结构
    model=Sequential()
    model.add(Dense(units=1024,input_dim=x_train.shape[1]))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    import numpy as np
    numClass=y_train.shape[1]
    print('numClass:',numClass)
    model.add(Dense(numClass,activation='softmax'))
    #model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
    #训练模型
    model.fit(x_train,y_train,epochs=5,batch_size=1000,validation_data=(x_test,y_test))

    #hist = model.fit(X, y, validation_split=0.2)
    #print(hist.history)
    from keras.models import load_model
    #保存模型
    model.save('my_model.h5')

    #loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    #classes = model.predict(x_test, batch_size=128)
    #model = load_model('my_model.h5')
    #可以测试针对测试集，模型的准确率
    note_prediction = list(model.predict(x_test, batch_size=256))
    print('*'*20,model.predict(x_test, batch_size=256),'*'*20)

    return model

#增加了卷积操作的模型
def Kmodel_2(X,y):
    #转换成图像处理一样 ，40000 -> 200*200
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

    #自定义网络结构
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(100, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    #训练模型
    model.fit(X, y, batch_size=16, epochs=10)
    #score = model.evaluate(x_test, y_test, batch_size=16)

    return model
