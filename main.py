from load_data import load_X_images, load_Y_labels

x_train = load_X_images('/Users/sunyambagga/Kaggle/Digit_Recognizer/train-images-idx3-ubyte.gz')

y_train = load_Y_labels('/Users/sunyambagga/Kaggle/Digit_Recognizer/train-labels-idx1-ubyte.gz')

#print x_train
#print y_train

# Using Theano and Lasagne
import lasagne
import theano
import theano.tensor as T

def creat_neural_network(input_var=None):

    input_layer = lasagne.layers.InputLayer(shape=(None,1,28,28), input_var=input_var)

    # Dropout to avoid overfitting
    d_input_layer = lasagne.layers.DropoutLayer(input_layer, p=0.2)

    hidden_layer_1 = lasagne.layers.DenseLayer(d_input_layer, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    d_hidden_layer_1 = lasagne.layers.DropoutLayer(hidden_layer_1, p=0.5)

    hidden_layer_2 = lasagne.layers.DenseLayer(d_hidden_layer_1, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())

    d_hidden_layer_2 = lasagne.layers.DropoutLayer(hidden_layer_2, p=0.5)

    output_layer = lasagne.layers.DenseLayer(d_hidden_layer_2, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return output_layer

#creat_neural_network()
