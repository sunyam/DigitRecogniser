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
input_var = T.tensor4('input')
target_var = T.ivector('target')

nn = creat_neural_network(input_var)

# Error Function
prediction = lasagne.layers.get_output(nn)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Update weights
parameters = lasagne.layers.get_all_params(nn, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, parameters, learning_rate=0.15, momentum=0.9)

# Creating a theano function for a single training step
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# TRAINING THE NEURAL NET
# Note: I trained it for 200 epochs; can take a few hours
num_iterations = 10

for i in range(num_iterations):
    err = train_fn(x_train, y_train)
    print "Current iteration" + str(i)


# Making predictions
test_prediction = lasagne.layers.get_output(nn, deterministic=True)

val_fn = theano.function([input_var], T.argmax(test_prediction, axis=1)[0])

# Download Kaggle Test Data: "test_MNIST.csv"
import csv
import numpy as np

X_test = []
data = []
with open("/Users/sunyambagga/Kaggle/Digit_Recognizer/test_MNIST.csv") as file:
    lineReader = csv.reader(file, delimiter=',', quotechar="\"")
    lineNum = 1
    for row in lineReader:
        if lineNum == 1:
            lineNum = 9
        else:
            data.append(row)

data = np.array(data, dtype='f')
data = data/np.float32(256)
X_test = data.reshape(-1, 1, 28, 28)


# Writing results to csv
for i in range(len(X_test)):
    with open('results.csv', 'a') as f:
        # Just to see progress
        if i%1000==0:
            print "Writing File", i
        
        f.write(str(i+1) + ',' + '"' + str(val_fn([X_test[i]])) + '"' + '\n')

# For Kaggle Submission, include "ImageId","Label" as the first row

# NOTE: With the current results.csv, you will get a 83% accuracy; Try out different parameters, specially num_iterations=200 to get more 99+% accuracy