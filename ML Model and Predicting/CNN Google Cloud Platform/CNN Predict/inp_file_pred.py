# import numpy, csv, fft
import numpy as np
import csv
from scipy.fftpack import fft
from google.cloud import storage

# Create a class for the datasets
class DataSet(object):
    def __init__(self, acdata, labels):
        # Assert dimmension-1 of data and labels are the same 
        assert acdata.shape[0] == labels.shape[0], (
                "acdata.shape: %s labels.shape: %s" % (acdata.shape,labels.shape))
                
        # Assert dimmension-3 of data is 1 
        assert acdata.shape[3] == 1
        # Convert acdata into 2-d list
        acdata = acdata.reshape(acdata.shape[0],
                                    acdata.shape[1] * acdata.shape[2])
        acdata = acdata.astype(np.float32)
        # no of datasets per batch
        self._num_examples = acdata.shape[0]
        self._acdata = acdata
        self._labels = labels
        # no of epochs completed
        self._epochs_done = 0       
        self._index_in_epoch = 0

    @property
    def acdata(self):
        return self._acdata

    @property
    def labels(self):
        return self._labels
    
    # Fucnction to return next batch of datasets
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_done += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._acdata = self._acdata[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # return the batch
        return self._acdata[start:end], self._labels[start:end]

# Function to download data from google cloud storage
def download_from_cloud():
    # cloud storage client
    client = storage.Client()
    # bucket in which streamed sensor data are stored
    bucket = client.get_bucket('my-bucks-data')
    # dataflow output path
    blob = storage.Blob('output/out', bucket)

    # download from storage to local file
    with open('data.csv', 'wb') as file_obj:
        blob.download_to_file(file_obj)

# Function to sort the downloaded data based on time of streaming
# (dataflow ouput is not in order )
def bubble_sort():
    with open("data.csv") as file:
        reader=csv.reader(file)
        da=list(reader) # d is list format of data.csv
    
    data=np.array(da)
    # shape of data
    shp=data.shape
    rw=shp[0]
    cl=shp[1]
    # print no of data points in received data ( must be at least 3140
    print('  ..no of data points=',rw,'\n')
    
    #converting to float values
    global data_sorted
    data_sorted=np.zeros(shp)
    for i in range(rw):
        for j in range(cl):
            data_sorted[i][j]=float(data[i][j])

    #bubble sort based on time
    for i in range(rw):
    #Last i elements are already in place   
        for j in range(rw-1-i): 
            if (data_sorted[j][0]+0.0 > data_sorted[j+1][0]+0.0):
                tmp=data_sorted[j][0]
                data_sorted[j][0]=data_sorted[j+1][0]
                data_sorted[j+1][0] =tmp

# Function to generate the data signals for prediction
def getData(colno):     # colno = column to be read
    ac_sig = np.zeros(3139)
    # normalize the signal
    for i in range(3139):
        ac_sig[i] = float(data_sorted[i][colno]) / 2.38
    
    # Sampling with sliding window 5 long, step size 2 ( mean )
    ac_smpld = np.zeros(1568)
    for m in range(1568):
        adn = 0.0
        for n in range(5):
            adn = adn + float(ac_sig[m*2 + n]) # sum 
            ac_smpld[m] = adn / 5 #average

    # Hanning window
    han_wind=np.hanning(1568)
    # Multiply the sampled signal with hanning window
    ac_han=np.multiply(ac_smpld,han_wind)

    # get fft of ac_han ; taking absolute value as fft is symmetric
    ac_fft = abs(fft(ac_han))
    # list to store the final result : the prediction data
    ac_data = np.zeros(784) 

    #finding rms of fft bands
    for i in range(784):
        sq_sum = 0.0
        for j in range(2):
            # squared sum
            sq_sum = sq_sum + ac_fft[i*2 + j] * ac_fft[i*2 + j]  
            # mean of squared sum
            sq_sum = sq_sum /2  
            # root of mean of squared sum = rms
            ac_data[i] = np.sqrt(sq_sum) 
    # return prediction data
    return ac_data

# Function to get the final datasets for training
def read_inp(n_pred,num_classes,one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    # download
    print('\n..Downloading data from cloud..\n')
    download_from_cloud()
    # sort
    print('\n..Sorting..\n')
    bubble_sort()
    
    print('\n..Reading data input for prediction..\n')

    # prediction data
    pred_acdata = np.zeros((n_pred,28,28,1))
    for i in range(n_pred):
        count=0
        acdat=getData(i)
        for j in range(28):
            for k in range(28):
                pred_acdata[i,j,k,0]=acdat[count] 
                count+=1
  
    # labels ( random )
    ext_lab = np.zeros((n_pred,))
    # Create prediction dataset
    data_sets.pred = DataSet(pred_acdata, ext_lab)
    # return the final dataset for prediction
    return data_sets
