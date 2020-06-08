import numpy as np
import keras, sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import shutil
from sklearn.model_selection import train_test_split
import pickle

data_dir_quakes='/media/peter/data/earthquakenz/data/bloksnorm1/'
data_dir_noquakes='/media/peter/data/earthquakenz/data/noquakes/bloksnorm1/'
data_dir_all='/media/peter/data/earthquakenz/data/bloks_all/'
filenames=np.load(data_dir_all+'filenames.npy')
labels=np.load(data_dir_all+'labels.npy')
# labels=keras.utils.to_categorical(labels)

filenames_shuffled, labels_shuffled = sklearn.utils.shuffle(filenames,labels)
X_train_filenames, X_val_filenames, y_train, y_val=train_test_split(filenames_shuffled, labels_shuffled,
                                                                   test_size=0.2,random_state=1)

class blok_Generator(keras.utils.Sequence) :

    def __init__(self, blok_filenames, labels, batch_size) :
        self.blok_filenames = blok_filenames
        self.labels = labels
        self.batch_size = batch_size


    def __len__(self) :
        # print('calling len')
        return (np.ceil(len(self.blok_filenames) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_x = self.blok_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        data_dir_all='/media/peter/data/earthquakenz/data/bloks_all/'

        return np.array([np.reshape(np.load(data_dir_all+file_name),(3000,3*58,1)) for file_name in batch_x]),np.array(batch_y)

#     return np.array([np.load(data_dir_all+file_name) for file_name in batch_x]),np.array(batch_y)
#     return np.array([
#             resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
#                for file_name in batch_x])/255.0, np.array(batch_y)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,SeparableConv2D,DepthwiseConv2D,Conv1D

batch_size=16

training_batch_generator=blok_Generator(X_train_filenames,y_train,batch_size)
validation_batch_generator=blok_Generator(X_val_filenames,y_val,batch_size)
# input_shape=(batch_size,3000,3,58) #np.load(data_dir_all+noquake_filenames[0]).shape
input_shape=(3000,58*3,1)

model = Sequential()



model.add(Conv2D(filters=12,kernel_size=(3,3),strides=(1,3),input_shape=input_shape,activation ='relu'
                          ,padding='valid'))
# model.add(Conv2D(filters=12,kernel_size=(10,1),strides=(4,1),input_shape=input_shape,activation ='relu'
#                           ,padding='valid'))
model.add(MaxPooling2D(pool_size=(3,1)))

model.add(Conv2D(filters=10,kernel_size=(15,1),strides=(5,1),activation ='relu'
                          ,padding='valid'))
model.add(MaxPooling2D(pool_size=(12,1))) #used to be 12
model.add(BatchNormalization(axis=3))

# model.add(Conv2D(filters=20,kernel_size=(6,1),strides=(3,1)
#                           ,activation ='relu',padding='valid'))

# model.add(SeparableConv2D(filters=2,kernel_size=(10,1),padding='valid',activation ='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))

# model.add(SeparableConv2D(filters=32,kernel_size=(10,3),padding='valid',activation ='relu'))
#,input_shape=input_shape,activation ='relu'))
# model.add(BatchNormalization(axis=3))
# model.add(SeparableConv2D(filters=32,kernel_size=(10,3),padding='valid'))
#,input_shape=input_shape,activation ='relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(128, activation = "relu")) #Fully connected layer
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(128, activation = "relu")) #Fully connected layer
model.add(Dense(128, activation = "relu")) #Fully connected layer

model.add(Dense(128, activation = "relu")) #Fully connected layer
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(128, activation = "relu")) #Fully connected layer
# model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation = "relu")) #Fully connected layer
# model.add(BatchNormalization())
model.add(Dropout(0.2))
# model.add(Dense(64, activation = "relu")) #Fully connected layer
# model.add(BatchNormalization())
# model.add(Dropout(0.4))


opt = keras.optimizers.Adam(learning_rate=0.003)
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model.add(Dense(12,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
print('Model Compiled')

# binary_accuracy=tf.keras.metrics.BinaryAccuracy(
#     name="binary_accuracy", dtype=None, threshold=0.5
# )
# model=keras.models.load_model('model_save12_4_checkpoint',custom_objects={'BinaryAccuracy':binary_accuracy})
model.load_weights('model_save12_4_checkpoint')
model.summary()

checkpoint = keras.callbacks.ModelCheckpoint('model_save12_4_checkpoint', monitor='loss', verbose=1, save_best_only=True, mode='min')
csv_logger = keras.callbacks.CSVLogger("model_history_log.csv", append=True)
callbacks_list = [checkpoint,csv_logger]

model_history=model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch=int(1523//batch_size),
                    epochs=150,
                    verbose=1,
                    validation_data=validation_batch_generator,
                    validation_steps=int(381//batch_size),
                    use_multiprocessing=True,
                    workers=8,
                    callbacks=callbacks_list)

model.save("model_save12_4")

with open('model_history_save12_4.pkl', 'wb') as file_pi:
    pickle.dump(model_history.history, file_pi)
