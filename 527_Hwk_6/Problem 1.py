from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import time
#import statements

end = 0


def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
#Method to load cifar10 dataset and store to 4 variables


def prepare_images(train, test):
    train_scaling = train.astype('float32')
    test_scaling = test.astype('float32')
    train_scaling = train_scaling / 255.0
    test_scaling = test_scaling / 255.0
    return train_scaling, test_scaling
#Scale the images to have values between 0 and 1, instead of 0 and 255
#This is to stop weights from being really big and having an unstable model


def define_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#Defining the model architecture. This was taken from an online introductory cifar10 machine learning
#tutorial


def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.show()
#Function to graph a bunch of performance metrics after every epoch, and see what the ideal
#number of epochs is and if the model is over training


def run_test():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prepare_images(trainX, testX)
    model = define_model()
    history = model.fit(trainX, trainY, epochs=64, batch_size=64, validation_data=(testX, testY), verbose=0)
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))
    global end
    end = time.time()
    summarize_diagnostics(history)
#Testing the model training and outputting performance metrics


if __name__ == "__main__":
    start = time.time()
    run_test()
    print("Execution time in seconds:", end-start)