# Author: Thai Quoc Hoang
# Email: quocthai9120@gmail.com / qthai912@uw.edu
# GitHub: https://github.com/quocthai9120

# Program Description: This program initializes data for the
# CIFAR_10 dataset then export to use the data later.


import numpy as np
from sklearn.model_selection import train_test_split
import os


def unpickle(file):
    """
    Pre : gives a file name contains data of cifar_10.

    Post: returns a dictionary with keys are categories of dataset
    and values are corresponding information for each key.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    # unpack all training data and testing data
    data = []
    for i in range(1, 6):
        data.append(unpickle('data_batch_' + str(i)))
    test = unpickle('test_batch')

    # divide training data to x (training inputs) and y (labels)
    label_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    x = []
    y = []

    for i in range(5):
        d = data[i][b'data'].reshape(10000, 3, 32, 32)
        labels = data[i][b'labels']

        for j in range(10000):
            x.append(np.transpose(d[j], (1, 2, 0)))
            y.append(labels[j])

    x = np.array(x)
    y = np.array(y)

    # divide testing data to x (testing inputs) and y (labels)
    d = test[b'data'].reshape(10000, 3, 32, 32)
    labels = test[b'labels']

    x_test = []
    y_test = []

    for i in range(10000):
        x_test.append(np.transpose(d[i], (1, 2, 0)))
        y_test.append(labels[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # divide training data to smaller training data and validation data
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.1, random_state=91)

    # export data
    os.mkdir('Models')
    save_dir = os.path.join(os.getcwd(), 'Models')
    np.save(save_dir + '/x_train', x_train)
    np.save(save_dir + '/x_val', x_val)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_val', y_val)
    np.save(save_dir + '/x_test', x_test)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/label_names', np.array(label_names))


if __name__ == "__main__":
    main()
