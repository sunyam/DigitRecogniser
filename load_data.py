# Download the MNIST Data from Yann Lecun's website

import gzip
import numpy as np

def load_X_images(file):

    with gzip.open(file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

        # Data is in the form of 1-D array, need to convert it to 4-D where 1st dimension is the total number of images, 2nd is channels (here, it is monochrome i.e 1), 3rd and 4th is the image pixels (28*28)
        data = data.reshape(-1, 1, 28, 28)

        # Convert bytes to float32 in the range [0,255]
        return data/np.float32(256)

#print load_X_images('/Users/sunyambagga/GitHub/DigitRecogniser/train-images-idx3-ubyte.gz')

def load_Y_labels(file):

    with gzip.open(file, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data

#print load_Y_labels('/Users/sunyambagga/GitHub/DigitRecogniser/train-labels-idx1-ubyte.gz')