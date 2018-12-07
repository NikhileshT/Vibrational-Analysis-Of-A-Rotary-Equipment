window_size=2 # in minutes -- dataflow ..
num_iter=1000000

import tensorflow as tf
import numpy as np
import inp_file_pred
import os
from google.cloud import storage
import datetime
import time

st=time.time() #start time

# Network Parameters
num_input = 784 # accel data input (28*28)
num_classes = 12 # accel total classes , 3*3
num_pred=1
n_pred =num_pred*3

#authenticating to the project
os.environ['GOOGLE_APPLICATION_CREDENTIALS']="schrocken.json"

def largest(a):
	max = a[0]
	pos = 0
	for i in range(num_classes):
		if(a[i]>max):
			max = a[i]
			pos = i
	return pos

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

client = storage.Client()
bucket = client.get_bucket('my-bucks-data')

def predict(x_set):
	#The input is of shape [None, height,width, num_channels]
	x_batch = np.reshape(x_set,[1,28,28,1])

	# Creating feed_dict & running the session to get 'result'
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)

	return largest(result[0])
		

def main():
    print('checking for file')
    flag_dwld=0 # to check if downloading happened
    flag_pred=0 # to check if prediction happened
    full_time=str(datetime.datetime.utcnow())
    date=full_time[:10]    
    hr=full_time[11:-13]
    minut=full_time[14:-10]
    try: 
        window_start_minut= int(minut) - window_size
        if window_start_minut<0:
            window_start_minut=window_start_minut+60
            window_start_hr=int(hr)-1
        else:
            window_start_hr=int(hr)
        if window_start_minut<10:
            window_start_minut='0'+str(window_start_minut)
        else:
            window_start_minut=str(window_start_minut)
        if window_start_hr<10:
            window_start_hr='0'+str(window_start_hr)
        else:
            window_start_hr=str(window_start_hr)
        current_filename=date+'T'+str(window_start_hr)+':'+str(window_start_minut)+':00.000Z-'+date+'T'+hr+':'+minut+':00.000Z-pane-0-last-00000-of-00001'
        blob = storage.Blob('output/'+current_filename, bucket)
            
        #print('trying to dwnld ')
        with open('data.csv', 'wb') as file_obj:
            blob.download_to_file(file_obj)
        print('..obtained data from cloud storage..')
            
            
        flag_dwld=1 # dwld happened
        data = inp_file_pred.read_inp(n_pred,num_classes,one_hot=False)
        x_set = data.pred._acdata
        #Prediction after download
        preds=np.zeros(n_pred)            
                        
        for i in range(n_pred):
            curr_pred = predict (x_set[i])   
            preds[i]=curr_pred
        flag_pred=1 # pred happened
            
            
            
        #assumption: only accel values are used (x,y,z)
        preds_edit = np.zeros(n_pred)
        #convert to real test labels
        for i in range(n_pred):
            for j in range(4):
                if preds[i]==3*j+0 or preds[i]==3*j+1 or preds[i]==3*j+2:
                    preds_edit[i] = j
            
        preds_final = np.zeros(num_pred)
        cnt=0
        for i in range(0,n_pred,3):
            avg = 0.0
            avg= (preds_edit[i] + preds_edit[i+1] + preds_edit[i+2] )/3
            avg = int(round(avg))
            preds_final[cnt] = avg
            cnt=cnt+1

        #Display the predicted classes
        print('PREDICTED SPEED:', int(preds_final[0]))
        print('Predicted at :',datetime.datetime.now(),'\n\n')
        ### if pred happens successfully
        min_done=int(minut) + window_size -1
        time.sleep(window_size*60-50)
        
    except:
    
        if flag_dwld==0:
            time.sleep(4)
            print('not found')
        elif flag_dwld==1 and flag_pred==0:
            print('\nERROR: - no. of data points retrieved are insufficient for prediction\n\n')
            ### if dwld happens successfully
            min_done=int(minut) + window_size -1
            time.sleep(window_size*60-50)

print('\nProcess started\n')        
for i in range(num_iter):
    main()

print('Process finished')

#improper file
#wait
