# Author: Thai Quoc Hoang
# Email: quocthai9120@gmail.com / qthai912@uw.edu
# GitHub: https://github.com/quocthai9120

# Program Description: This program trains the Convolutional Neural Network
# model for CIFAR_10 dataset.

from keras.utils import to_categorical
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, \
                         Flatten, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def standardize_data(data):
    '''
    Pre : gives a np array represents color images.

    Post: returns a new np array represents color images after
    normalization by subtracting mean then dividing by the standard
    deviation.
    '''
    data_de_mean = data - np.mean(data, axis=0)
    return data_de_mean / np.std(data_de_mean, axis=0)


def initialize_model(x_train, num_classes):
    '''
    Pre : gives a 4 dimensional numpy array represents the training images
    and an integer represents the number of classes.

    Post: returns a CNN model as Keras Sequential model.
    '''
    model = Sequential()

    model.add(Conv2D(
        filters=32, kernel_size=(3, 3), input_shape=x_train.shape[1:]
    ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def save_model(model, name, direction):
    '''
    Pre : gives a Kereas Sequential model, a file name as string, and
    a direction from current directory as string

    Post: saves the model to the given path
    '''
    save_dir = os.path.join(os.getcwd(), direction)
    model_path = os.path.join(save_dir, name)
    model.save(model_path)


def visualize_images(data, labels, label_names, predict=None, channels=3,
                     start=0, cols=4, rows=4, size=10, fontsize=10):
    '''
    Pre : gives a numpy array represents data of color images (4 dimensions
    array), a numpy array represents labels for the corresponding images, a
    numpy array represents the label names, a numpy array represents the
    prediction for the given images, an integer represents number of channels
    of the given images with default = 3, an integer represents start index
    for visualization with default = 0, an integer represents number of
    columns with default = 4, an integer represents number of columns with
    default = 4, an integer represents size of image with default = 10, an
    integer represents size of title's font with default = 10.

    Post: plots predicted images and save the plot to 'CNN predictions.png'.
    '''
    if (channels != 3):
        data = data[:, :, :, 0]
    fig = plt.figure(figsize=(size, size))
    plt.subplots_adjust(bottom=.05, top=.95, hspace=.9)

    cols = cols
    rows = rows
    for i in range(1, cols * rows + 1):
        img = data[start + i - 1]
        fig.add_subplot(rows, cols, i)
        if (channels != 3):
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        if predict is not None:
            pred = label_names[predict[start + i - 1]]
        else:
            pred = 'NaN'
        real = label_names[int(np.where(labels[start + i - 1] == 1)[0])]
        plt.title('Predict: ' + pred + '\n Real: ' + real, fontsize=fontsize)
        plt.axis('off')
        plt.savefig('CNN predictions.png')
    plt.show()


def check_predictions_probability(model, x_test, y_test, label_names,
                                  number_instances=10):
    '''
    Pre : gives a Kereas Sequential model, a numpy array represents
    x_test with 4 dimensions (color images), a numpy array represents
    all labels for corresponding testing images, a numpy array of label
    names, and an integer represents number of testing instances with
    default = 10.

    Post: plots several predictions with probabilities of each label
    and save as png files.
    '''
    print('Use several examples to see how the model classify:')
    y_pred = model.predict(x_test)

    for i in range(number_instances):
        scores = y_pred[i]
        data = pd.DataFrame({'Labels': label_names, 'Scores': scores})
        sns.catplot(
                x='Labels',
                y='Scores',
                kind='bar',
                data=data
            )
        plt.xticks(rotation=-45)
        plt.title('Testing Instance ID: ' + str(i) + '\n'
                  'Predicted:' + str(label_names[np.argmax(scores)])
                  + '\nReal Label:'
                  + str(label_names[int(np.where(y_test[i] == 1)[0])]))
        plt.savefig('CNN prediction probability instance ' + str(i) + '.png',
                    bbox_inches='tight')
    plt.show()


def main():
    # load data
    x_train = np.load('Models/x_train.npy').astype('float32') / 255
    x_val = np.load('Models/x_val.npy').astype('float32') / 255
    x_test = np.load('Models/x_test.npy').astype('float32') / 255
    y_train = np.load('Models/y_train.npy')
    y_val = np.load('Models/y_val.npy')
    y_test = np.load('Models/y_test.npy')
    label_names = np.load('Models/label_names.npy')
    num_classes = len(label_names)

    # put y_train, y_val, and y_test to categories
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # standardize data
    x_train_standardization = standardize_data(x_train)
    x_val_standardization = standardize_data(x_val)
    x_test_standardization = standardize_data(x_test)

    # initialize model
    model = initialize_model(x_train, num_classes)

    # initialize data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(x_train_standardization)

    # compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # setup Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                   patience=10, verbose=2, mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)

    # setup model checkpoint
    checkpointer = ModelCheckpoint(filepath='Models/weights.hdf5', verbose=1,
                                   save_best_only=True)

    # setup reduce learning rate when the metric stops improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                  verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=0, min_lr=0)

    # fit the model with training data and validate with validation data
    history = model.fit_generator(
        datagen.flow(x_train_standardization, y_train, batch_size=32),
        steps_per_epoch=len(x_train_standardization) / 32, epochs=100,
        validation_data=(x_val_standardization, y_val),
        shuffle=True,
        callbacks=[early_stopping, checkpointer, reduce_lr]
    )

    # plot the model's validation loss for further hyperparameters tuning
    val_loss = history.history['val_loss']
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.xlabel('epoch')
    plt.ylabel('validation loss')
    plt.title('CNN model validation loss history')
    plt.show()

    # predict testing data
    print()
    print('Evaluating testing data:')
    scores = model.evaluate(
        x_test_standardization,
        y_test,
        verbose=1
    )
    print('   Model accuracy for testing data:', str(scores[1] * 100) + '%')
    print()

    # visualize predictions
    y_pred = model.predict_classes(x_test_standardization)
    visualize_images(x_test, y_test, label_names, y_pred,
                     cols=8, rows=8, fontsize=8)

    # check predictions probability
    sns.set()
    check_predictions_probability(model, x_test_standardization,
                                  y_test, label_names)

    # export model
    save_model(model, name='cnn_model_improved', direction='Models')


if __name__ == "__main__":
    main()
