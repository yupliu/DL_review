import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical


(train_x,train_y),(test_x,test_y) = keras.datasets.mnist.load_data()

train_x = train_x/255.0
test_x = test_x/255.0

train_x_reshape = train_x.reshape(train_x.shape[0],-1)
test_x_reshape = test_x.reshape(test_x.shape[0],-1)

nb_class = 10
train_y_reshape = to_categorical(train_y,nb_class)
test_y_reshape = to_categorical(test_y,nb_class)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32,input_dim=28*28,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
print(model.summary())
model.fit(train_x_reshape,train_y_reshape,epochs=10,batch_size=100)
score = model.evaluate(test_x_reshape,test_y_reshape,verbose=0)
print('Test score:', score)


