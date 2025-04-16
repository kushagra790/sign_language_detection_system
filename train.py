# Importing the necessary libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setting CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setting image size
sz = 48

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# Adding convolutional layers and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers with dropout
classifier.add(Dense(units=128, activation='relu'))
#classifier.add(Dropout(0.30))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # Assuming 27 classes for sign language detection

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data and training the model

# Setting up data augmentation for the training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Setting up data augmentation for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading training and test data from directories
training_set = train_datagen.flow_from_directory('data2/train',
                                                 target_size=(sz, sz),
                                                 batch_size=15,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data2/test',
                                            target_size=(sz , sz),
                                            batch_size=15,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Training the model
classifier.fit(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=20,
        validation_data=test_set,
        validation_steps=len(test_set))

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.weights.h5')
print('Weights saved')
