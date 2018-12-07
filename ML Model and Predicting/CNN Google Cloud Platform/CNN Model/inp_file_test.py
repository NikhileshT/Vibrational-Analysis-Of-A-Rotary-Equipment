# import numpy, csv, fft
import numpy as np
import csv
from scipy.fftpack import fft

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


# 784*2=1568, 1567*2 + 5 = 3139
# open the file with testing data
with open("test_sig1.csv") as file:
    # read the csv file
    reader=csv.reader(file)
    #assign the values to a list
    ts_sig=list(reader) #ts_sig is the list format of csv file test_sig

# Function to generate the data signals for testing
def getData(colno):     # colno = column to be read
    ac_sig = np.zeros(3139)
    # normalize the signal
    for i in range(3139):
        ac_sig[i] = float(ts_sig[i+1][colno]) / 2.38
    
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
    # list to store the final result : the test data
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
    # return test data
    return ac_data

# Function to get the final datasets for training
def read_inp(n_test,num_classes,one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    # Print on terminal
    print('\n..Reading data inputs for testing..\n')

    # test data
    test_acdata = np.zeros((n_test,28,28,1))
    for i in range(n_test):
        count=0
        acdat=getData(i)
        for j in range(28):
            for k in range(28):
                test_acdata[i,j,k,0]=acdat[count] 
                count+=1
    
    # labels ( random )
    ext_lab = np.zeros((n_test,))
    # Create test dataset
    data_sets.test = DataSet(test_acdata, ext_lab)
    # return the final test dataset
    return data_sets
