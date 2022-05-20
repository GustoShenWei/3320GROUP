# Convolutional Neural Network
# 5/18-2022
# This neural network is trained to differentiate the characteristics
# between Pigeon and Turtledove. To test, add an image of a Pigeon or a Turtledove to the
# single prediction folder and execute the entrie code (besides the fit generator function)
# with the path of the new image you want to use. 
#

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 

#Initialise the CNN
classifier = Sequential() 

# 1 - Build the convolution

# 32 filters or a 3x3 grid, since these pictures are colorful, color needs a axis.
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))

# 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Second layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# 3 - Flattening 
classifier.add(Flatten())

# 4 - Full Connection, making an ANN

classifier.add(Dense(128, activation = 'relu'))

#Binary outcome so sigmoid is being used
classifier.add(Dense(1, activation = 'sigmoid'))

## Compiling the NN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the neural network for the images
# Make these images to be a same size, not too large or small

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


###### EXECUTE THIS TO TRAIN MODEL ##############
#classifier.fit(training_set,
 #               steps_per_epoch=30,
  #                epochs=20,
   #                 validation_data=test_set,
    #                validation_steps=1500)
            


#--------- New Prediction -------------

import numpy as np
from keras.preprocessing import image

#Load the image
test_image = image.load_img('dataset/single_prediction/1.jpeg',target_size=(64, 64))  #Change the name of this line to read other picture
#Change to a 3 Dimensional array because it is a colour image
test_image = image.img_to_array(test_image)
#add a forth dimension
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

#treshold of 50% to classify the image
if result[0][0] > 0.5:
    prediction = 'Pigeon'
    print(result[0][0], " certainty of being a", prediction)
else:
    prediction = 'Turtledove'
    temp = 1.0 - result[0][0]
    print(temp, " certainty of being a", prediction)
        
 




