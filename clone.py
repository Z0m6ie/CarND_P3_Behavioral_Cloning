import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Lambda, MaxPooling2D
from keras.layers import Dropout
from keras.layers import Cropping2D
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint
from random import shuffle
import matplotlib.pyplot as plt

# Read in the csv log from the training data. I manually removed the headders
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Use train test split from sklearn to create a validation set of 20%
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Use a generator to reduce the amount of info the computer need to store in mem


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            # create 2 lists of images and stearing angles to be the input and output
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                lname = './data/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(lname)
                rname = './data/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(rname)

                # opencv aparantly reads in images as BGR, our images are RGB
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                #temp = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
                # Compute a random brightness value and apply to the image
                #brightness = 0.25 + np.random.uniform()
                #temp[:, :, 2] = temp[:, :, 2] * brightness
                #center_image = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + 0.25
                right_angle = float(batch_sample[3]) - 0.25
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320

# Create the model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))  # Crop image
model.add(Conv2D(24, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(36, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(48, 5, 5, subsample=(2, 2),activation='relu', W_regularizer=l2(0.001)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.25))
model.add(Dense(10, activation='relu', W_regularizer=l2(0.001)))
#model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath='model{epoch:02d}.h5')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=validation_generator, nb_val_samples=len(validation_samples), verbose=1, nb_epoch=5, callbacks=[checkpointer])
# 5 epochs

print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
