#######################################
# This contains function to load different datatypes and in general data handling
#######################################
import numpy as np
import urllib
import gzip
import cPickle
import os
import matplotlib.pyplot as plt

def load_mnist(f_source=None,file_name='mnist.pkl.gz',f_saved_path=None,num_training=50000, num_val=10000, num_test=10000):
    """
     This function load MNIST dataset from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
     Parameters:
         input: 
           mnist_file_path: mnist file location
           file_name: file name to be saved 
           f_source: file's url
           num_training: To specifiy how many training samples to be returned, should be =<50000
           num_val:   To specifiy how many validation samples to be returned, should be =<10000
           num_test:  To specifiy how many validation samples to be returned, should be =<10000
           
         output:
           X_train,y_train ==> training data
           X_val,y_val     ==> validation data
           X_test, Y_test  ==> test data
    """
    if num_training > 50000 or  num_training <1 or  num_val > 10000 or num_val < 1 or num_test > 10000 or num_test <1:
         raise ValueError('Number of num_training/num_test/num_val must be between [1,50000]/[1,10000]/[1,10000] ')

    ##############################################
    # Step 1: download the file
    ##############################################
    if f_source is None:
        f_source='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if f_saved_path is None:
        f_name=file_name
    else:
        f_name=os.path.join(f_saved_path,file_name)

    if os.path.exists(f_name):
        print ('The %s exits' % f_name)
    else:
        print('Start downloading %s from %s' %(file_name,f_source))
        urllib.urlretrieve(f_source,file_name)
        print('Done witd data downloading.')
        
    ##############################################
    # Step 2: Load the file and preporcess it
    ############################################## 
    f_data = gzip.open(file_name, 'rb')
    training_set, validation_set, test_set = cPickle.load(f_data)
    f_data.close()
    
    X_train,y_train=training_set
    X_val,y_val=validation_set
    X_test,y_test=test_set
    
    print ('Training ====> images %s and lables %s' %(X_train.shape,y_train.shape))
    print ('Validation ==> images %s and lables %s' %(X_val.shape,y_val.shape))
    print ('Test ========> images %s and lables %s' %(X_test.shape,y_test.shape))
    print ('Done witd data loading.')
    return X_train[:num_training],y_train[:num_training],X_val[:num_val],y_val[:num_val],X_test[:num_test],y_test[:num_test]


def vis_data(input_data,labels,cls_names,exp_per_class,img_dim):
    """
     This function shows some examples per classes
     input_data should be N*D where N is number of samples and D is data dimension 
    """
    for cls,cls_name in enumerate(cls_names):
        
        # find the index of given classes
        idx_set=np.where(labels ==cls)[0]
        idx_set=np.random.choice(idx_set,exp_per_class,replace=False)
       
        for i,idx in enumerate(idx_set):
        
            # create a subplot and show the images
            plt.subplot(exp_per_class,len(cls_names),1+i*len(cls_names)+cls)
            plt.imshow(input_data[idx].reshape(img_dim,img_dim))
            plt.axis('off')
            if i==0:
                plt.title(cls_name)
    plt.show()        
    plt.close()



