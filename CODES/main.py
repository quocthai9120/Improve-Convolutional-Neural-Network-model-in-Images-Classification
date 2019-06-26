# Author: Thai Quoc Hoang
# Email: quocthai9120@gmail.com / qthai912@uw.edu
# GitHub: https://github.com/quocthai9120

# Program Description: This program calls all other files to initialize data
# then runs the CNN model for the CIFAR_10 dataset.


import cifar_10_data_preprocessing
import cifar_10_conv_net


def main():
    cifar_10_data_preprocessing.main()
    cifar_10_conv_net.main()


if __name__ == "__main__":
    main()
