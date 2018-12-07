import tensorflow as tf
import numpy as np
import inp_file_pred
import os

# Network Parameters
num_input = 784 # accel data input (28*28)
num_classes = 12 # accel total classes , 3*3

# no of prediction datasets
num_pred=1
# total no of datasets = num_pred * no of axes (x,y,z)
n_pred =num_pred*3 

#authenticating to the project
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="schrocken.json"

#Getting inp data
data = inp_file_pred.read_inp(n_pred,num_classes,one_hot=False)
# input data set
x_set = data.pred._acdata

# function to return position of largest number of a list
def largest(a):
	max = a[0]
	pos = 0
	for i in range(num_classes):
		if(a[i]>max):
			max = a[i]
			pos = i
	return pos
print('\n..Reconstructing the Tensorflow graph..\n')

# Funcion to predict the class
def predict(x_set):

	#The input is of shape [None, height,width, num_channels]
	x_batch = np.reshape(x_set,[1,28,28,1])
	# Restoring the saved model 
	sess = tf.Session()
	# Recreating the network graph
	saver = tf.train.import_meta_graph('cnn-model.meta')
	# Loading the weights
	saver.restore(sess, tf.train.latest_checkpoint('./'))

	# Accessing the restored default graph
	graph = tf.get_default_graph()

	# o/p tensor in original graph
	y_pred = graph.get_tensor_by_name("y_pred:0")

	# Feeding inputs to the input placeholders
	x= graph.get_tensor_by_name("x:0") 
	y_true = graph.get_tensor_by_name("y_true:0") 
	y_test_images = np.zeros((1, num_classes)) 

	# Creating feed_dict & running the session to get 'result'
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)

    # return the predicted class
	return largest(result[0])

#Prediction
preds=np.zeros(n_pred)
for i in range(n_pred):
    curr_pred = predict (x_set[i])   
    preds[i]=curr_pred #preds stores all the predictions

preds_edit = np.zeros(n_pred)
#convert to real test labels (0,1,2,3)
for i in range(n_pred):
    for j in range(4):
        if preds[i]==3*j+0 or preds[i]==3*j+1 or preds[i]==3*j+2:
            preds_edit[i] = j

preds_final = np.zeros(num_pred)
cnt=0
# Get the final prediction (mean of three axes)
for i in range(0,n_pred,3):
    avg = 0.0
    avg= (preds_edit[i] + preds_edit[i+1] + preds_edit[i+2] )/3
    avg = int(round(avg))
    preds_final[cnt] = avg
    cnt=cnt+1

#Display the predicted speed
print('\nPredicted Speed:', int(preds_final[0]),'\n')
