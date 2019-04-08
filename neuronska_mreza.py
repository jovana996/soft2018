# keras
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Dropout, Flatten, MaxPooling2D
import numpy as np



def create_model():
    model = Sequential()
    model.add(Convolution2D(28, (3, 3), padding='same', activation='relu',input_shape=(28, 28, 1)))
    model.add(Convolution2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model




