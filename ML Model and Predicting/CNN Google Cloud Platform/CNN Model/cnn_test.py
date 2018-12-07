# import numpy, tensorflow, input file & other required libraries
import tensorflow as tf
import numpy as np
import inp_file_test

# no of test datasets
num_test = 20 
# total no of datasets = num_test * no of axes (x,y,z)
n_test = num_test*3

# Network Parameters
num_input = 784 # accelertion data input size (28*28)
num_classes = 12 # acceleration total classes , 4*3

# Getting input data for testing
data = inp_file_test.read_inp(n_test,num_classes,one_hot=False)

# input data set
x_set = data.test._acdata
# Reshape to 28*28 
x_set = np.reshape(x_set,[n_test,28,28,1])

# function to return position of largest number of a list
def largest(a):
	max = a[0]
	pos = 0
	for i in range(num_classes):
		if(a[i]>max):
			max = a[i]
			pos = i
	return pos

# Funcion to test the model
def test_fn(x_set):

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

# Testing
preds=np.zeros(n_test)  
for i in range(n_test):
    curr_pred = test_fn (x_set[i])   
    preds[i]=curr_pred  #preds stores all the predictions

preds_edit = np.zeros(n_test)
#convert to real labels (0,1,2,3)
for i in range(n_test):
    for j in range(4):
	    if preds[i]==3*j+0 or preds[i]==3*j+1 or preds[i]==3*j+2:
	    	preds_edit[i] = j 

preds_final = np.zeros(num_test)
cnt=0
# Get the final predictions (mean of three axes)
for i in range(0,n_test,3):
    avg = 0.0
    avg= (preds_edit[i] + preds_edit[i+1] + preds_edit[i+2] )/3
    avg = int(round(avg))
    preds_final[cnt] = avg
    cnt=cnt+1

#Display the predicted classes against the input test datasets
for i in range(num_test):
    print('Test data:',i+1, "   Predicted Speed:", int(preds_final[i]))

test_labels = np.zeros(20)
cnt1=0
# actual test labels
for i in range(4):
    for j in range(5):        
	    test_labels[cnt1] = i
	    cnt1=cnt1+1

#Finding accuracy
adn=0.0
for i in range(num_test):
    if preds_final[i]==test_labels[i] :
	    adn=adn+1.0
		    
acc=adn/num_test*100

# Print the accuracy of testing
print('\nAccuracy of testing :',acc,'%')
