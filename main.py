from load_data import load_X_images, load_Y_labels

x_train = load_X_images('/Users/sunyambagga/GitHub/DigitRecogniser/train-images-idx3-ubyte.gz')

y_train = load_Y_labels('/Users/sunyambagga/GitHub/DigitRecogniser/train-labels-idx1-ubyte.gz')

print x_train
print "\n\n\n\n"
print y_train