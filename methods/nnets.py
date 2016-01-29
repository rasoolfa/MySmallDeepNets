import numpy as np

def init_model(input_dim, n_hiddens, ouput_size,r_seed=1234):
        """
        Initialize the weights and biases for a two-layer fully connected neural
        network. 

        Returns:
        A dictionary mapping parameter names to arrays of parameter values. It has
        the following keys:
        - W1: First layer weights; has shape (D, H)
        - b1: First layer biases; has shape (H,)
        - W2: Second layer weights; has shape (H,output_size)
        - b2: Second layer biases; has shape (output_size,)

        """

        rng=np.random.RandomState(r_seed) 
        
        # initialize a model
        model = {}
        model['W1'] = 0.01 * rng.randn(input_dim,n_hiddens)
        model['b1'] = np.zeros(n_hiddens)
        model['W2'] = 0.01 * rng.randn(n_hiddens, ouput_size)
        model['b2'] = np.zeros(ouput_size)

        return model

def NN_net_two_layer(X, model,y=None, reg=0.0,p=1.0):
        
        """
         In this function, the gradient is computed for two-layers NN based on the backpropagation 
         X: is input data which is N,D
         y: is label sets, N*1
         reg_val: is L2 regularization multiplier
         models: is all model params
         p: is dropout rate, p=1 means no dropout. P should be 0< p <=1 
        """
        
        ##################
        # Step 0: Extract some info about data and model parameters
        ##################
        
        N=float(X.shape[0])
        W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2']

        ##################
        # Step 1: Do feedforward pass 
        ##################
        dot1=np.dot(X,W1)+b1    #dot1 is N*H
        H1=reLU(dot1)  # H1 is  N*H
        
        #dropout out step (implemented based on inverted dropout)
        mask_drop=(np.random.rand(*H1.shape)<p)/p
        out_drop = H1*mask_drop # drop!

        dot2=np.dot(out_drop,W2)+b2  #dot2 is N*C
        score=dot2      #dot2 is N*C
        
        # return the socre for each classes.
        #It would useful for prediction time. 

        if y is None:
            return score
        
        ##################
        # Step 2: Now calculate the loss and gradient
        ##################

        #compute loss
        loss=None
        loss,ddot2=softmax_loss_grad(dot2,y)
        loss=loss+0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))
        
        ##################
        # Step 2: Now it is time to propagate the error and 
        # calculate the gradient using chain rule (i.e. backprogation)
        ##################
        dout_drop=np.dot(ddot2,W2.T)
        dW2=np.dot(out_drop.T,ddot2)
        db2=np.sum(ddot2,axis=0)

        dH1= dout_drop*mask_drop

        ddot1=dreLU(dot1)*dH1
        dW1=np.dot(X.T,ddot1)
        db1=np.sum(ddot1,axis=0)

        ##################
        # Step 3: Now we have the grads, it is time to save them and add regs if there is any 
        ##################
        grads={}
        grads['W2']=dW2/N+(reg*W2)
        grads['b2']=db2/N
        grads['W1']=dW1/N+ (reg*W1)
        grads['b1']=db1/N

        return loss, grads
def reLU(U):
    """
    This function computes ReLU scores which is the max(0, XW+b)dot is N*H
    Instad of np.maximum(0,dot), I use U * (U > 0)    
    Refer:https://groups.google.com/forum/#!msg/theano-users/ifA36zXtBiI/JefZ_F6rkyAJ
    """
    return U * (U > 0) # would be N*H

def dreLU(f):
    """
    Calculate the derivative of Relu func
    """
    return f > 0.0
  
def softmax_loss_grad(f,y):
    """
    This function compute the softmax loss-function and grad f is score and N*C 
    """

    num_classes=f.shape[1]

    f=f.T
    f-=np.max(f,axis=0) # to deal with numerical stability 
    F =np.exp(f) # size f is C*N

    mask_y = np.zeros((num_classes, y.shape[0])) # is C*N
    mask_y[y, np.arange(y.shape[0])] = 1.0   #mask_y: in each column only the corresponding row is 1 other zeros, for example if y[0] is class=2 only row[2] is 1 and others are zeros
  
    p_0=(F)/np.sum(F,axis=0)  #p_0 is C*N
    p_m=p_0*mask_y       #p_m is C*N, it only contains value for y[i]
    p=np.sum(p_m,axis=0) # convert to 1*N
    loss=sum(-np.log(p)) / y.shape[0]
  
    #print 'Nice...',p_0
    df= p_0-mask_y 
  
    return loss,df.T  

def euclidean_loss_grad(f,y):
    """
     This function compute the loss-function and grad 
    """ 
    N=float(f.shape[0])
    diff=f-y
    loss=0.5*np.sum((diff)**2)/N
    return loss,diff
 
