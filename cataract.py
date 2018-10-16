import keras

import numpy as np

from PIL import Image




model = Sequential()

model.add(Convolutional2D(32,3,3, input_shape(img_width, img_height,3)))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(convolutional2D(32,3,3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(convolutional2D(64,3,3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])

model.fit_generator(training_data,
samples_per_epoch = 2048, 
nb_epoch = 30,
validation_data=validation_data,nb_val_samples=832)

model.save('models/simple_CNN.h5')

'''