# import numpy, tensorflow, input file & other required libraries
from __future__ import division, print_function, absolute_import
import inp_file_train
import tensorflow as tf
import numpy as np

#no of training datasets in for each speed
num_train =30 
# total no of datasets = num_train * no of classes(0,1,2,3,) * no of axes (x,y,z)
n_train = num_train*4*3 

# Training Parameters
batch_size = 15 

# Network Parameters
num_input = 784 # acceleration data input size (28*28)
n_classes = 4 # acceleration total classes , 0,1,2,3 (speeds)

num_classes=n_classes*3 # n_classes * no of axes

# Getting input data for training
data = inp_file_train.read_inp(n_train,num_classes,one_hot=False)

# Calling tensorflow session
session = tf.Session() 
# input placeholder
x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name='x')

# placeholders for labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# no of channels in convolution process
num_channels=1 

## Network graph params
# filter size for first convolution layer
filter_size_conv1 = 5 
# no of filters in first convolution layer
num_filters_conv1 = 32

# filter size for second convolution layer
filter_size_conv2 = 3
# no of filters in second convolution layer
num_filters_conv2 = 64

# fully conneted layer size
fc_layer_size = 1024

# function to create weights of given shape
def create_weights(shape):
    #initialize and return truncated normal random values with s.d 0.05
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# function to create biases of given size
def create_biases(size):
    #initialize and return constant values
    return tf.Variable(tf.constant(0.05, shape=[size]))

# function to create conv layer & perform convolution, max-pooling and ReLU activation
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):     
    # create weights   
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # create biases
    biases = create_biases(num_filters)

    # Create conv layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    layer += biases

    # max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
                            
    # Output of pooling is fed to Relu, the activation function
    layer = tf.nn.relu(layer)

    #return the resulting layer
    return layer

# function to flatten the input layer
def create_flatten_layer(layer):
    # getting shape from the previous layer.
    layer_shape = layer.get_shape()

    # No. of features = height * width* channels
    num_features = layer_shape[1:4].num_elements()

    # Flatten the layer
    layer = tf.reshape(layer, [-1, num_features])

    #return the resulting layer
    return layer

# function to create fully connected layer
def create_fc_layer(input,
             num_inputs,  
             num_outputs,
             use_relu=True):
    # create weights 
    weights = create_weights(shape=[num_inputs, num_outputs])
    # create biases 
    biases = create_biases(num_outputs)

    # For FC layer i/p=x and o/p=wx+b
    layer = tf.matmul(input, weights) + biases
    # activation
    if use_relu:
        layer = tf.nn.relu(layer)

    #return the layer
    return layer

# Creation of conv layer 1
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

# Creation of conv layer 2
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)
               
# Creation of flattened layer          
layer_flat = create_flatten_layer(layer_conv2)

# Creation of fully connected layer 1
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

# Creation of fully connected layer 2
layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

# get predictions ( probabilities )
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

# get the predicted class ( class with maximum probability )
y_pred_cls = tf.argmax(y_pred, dimension=1)

# initiaize global variables
session.run(tf.global_variables_initializer())
# find cross entropy after applying softmax fn
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
# find cost, the mean of cross entropy foe different classes
cost = tf.reduce_mean(cross_entropy)
# Define Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
# Calculate no of correct predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# Calculate accuracy of training
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer()) 

# Function to show progress of training on terminal window
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # training accuracy
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    # Validation accuracy
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    # Message to be printed
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    # Print the message
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

# incremental variable while training
total_iterations = 0
# Define saver
saver = tf.train.Saver()

# function to train the network
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        # Input data for training and validation
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch = data.validation.next_batch(batch_size)

        # Reshape to 28*28 
        x_batch1=np.reshape(x_batch, [-1,28,28,1])
        x_valid_batch1=np.reshape(x_valid_batch, [-1,28,28,1])
        
        # Create feed dictionary
        feed_dict_tr = {x: x_batch1,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch1,
                              y_true: y_valid_batch}
        
        # Optimize the model
        session.run(optimizer, feed_dict=feed_dict_tr)

        # Show progress of training process at regular intervals
        if i % int(data.train._num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train._num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            # Save the model at regular intervals ( overwrite )
            saver.save(session, './cnn-model') 


    total_iterations += num_iteration

# Train the model 
train(num_iteration=1600)

# Display the completion of training
print('\n..Training completed..\n ..The model has been saved..\n')
