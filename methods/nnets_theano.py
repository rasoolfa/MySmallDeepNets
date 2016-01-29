import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict


def get_random_start_seed():
     """ 
       set random seed to make result reproducible
     """
     random_seed = 1234
     return np.random.RandomState(random_seed) 

def numpy_floatX(data):
    """
      Helper function
    """ 
    return np.asarray(data, dtype=config.floatX)

def reLU(U):
    """
    This function computes ReLU scores which is the max(0, XW+b)dot is N*H
    Instad of np.maximum(0,dot), I use U * (U > 0)    
    Refer:https://groups.google.com/forum/#!msg/theano-users/ifA36zXtBiI/JefZ_F6rkyAJ
    """
    return U * (U > 0) # would be N*H

def init_model(input_dim, n_hiddens, ouput_size):
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
        rng = get_random_start_seed()  

        # initialize a model
        model = OrderedDict()
        model['W1'] = theano.shared(0.01 * rng.randn(input_dim,n_hiddens).astype(config.floatX),name ='W1')
        model['b1'] = theano.shared(np.zeros(n_hiddens).astype(config.floatX),name ='b1')
        model['W2'] = theano.shared(0.01 * rng.randn(n_hiddens, ouput_size).astype(config.floatX),name ='W2')
        model['b2'] = theano.shared(np.zeros(ouput_size).astype(config.floatX),name ='b2')
        
        return model

def dropout(H,use_dropout,dropout_prob):
    '''
      Implementation of inverted dropout 
      use_dropout is a flag that indicates whether apply dropout or not
      dropout_prob is dropout rate
    '''
    rng = get_random_start_seed()
    srng = RandomStreams(rng.randint(999999))
    mask=srng.binomial(H.shape, p=dropout_prob,n=1, dtype=config.floatX)/(dropout_prob)
    mask=T.cast(mask,config.floatX)
    U=T.switch(T.eq(use_dropout,1.0),mask,1.0)
    U=H*U
    return U

def build_model(params,drop_p=1.0,reg=0.0): 
    '''
      In this function define the forward pass. 
    '''
    
    ###############################
    # Step 0: Extract some info about data and model parameters
    ###############################

    # Used for dropout.
    use_dropout   = theano.shared(numpy_floatX(0.))
    drop_prop     = theano.shared(numpy_floatX(drop_p), name='drop_prop')

    #get the parameters
    W1,b1,W2,b2= params['W1'],params['b1'],params['W2'],params['b2']
    
    # get the regularization multiplier   
    beta= theano.shared(numpy_floatX(reg), name='beta')
    
    #define Xt theano.matrix and yt theano.ivector 
    Xt = T.matrix('Xt',dtype=config.floatX)
    yt = T.vector('yt',dtype='int32') 

    ##################
    # Step 1: Network architecture
    ##################
    # layer One
    dot1= T.dot(Xt,W1)+b1    #dot1 is N*H
    H1= reLU(dot1)  # H1 is  N*H
    out_drop= dropout(H1,use_dropout,drop_prop)  #dropout
    
    # layer Two
    dot2= T.dot(out_drop,W2)+b2  #dot2 is N*C
    score= dot2      #dot2 is N*C
    
    ##################
    # Step 2: Now define the loss function
    ##################
    N= Xt.shape[0]

    # use hing loss as cost function   
    f= dot2.T
    correct_cls_scores= f[yt[T.arange(N)],T.arange(N)]
    margins = reLU(f - correct_cls_scores + 1.0) # the relu here is used to do the T.maximum
    margins= T.set_subtensor(margins[yt, T.arange(N)],0.0)
    
    # calculate the cost
    cost= 0.5*beta*(T.sum(W1*W1)+T.sum(W2*W2))+ T.sum( margins ) / N
    
    return Xt,yt,cost,score,use_dropout

def sgd(lr, model, grads, xt, yt, cost):
    """
       Stochastic Gradient Descent
    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in model.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([xt,yt], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(model.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, model, grads, xt, yt, cost,decay_rate=0.95):
    """
    
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    xt: Theano variable
        Model inputs
    yt: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    """
    
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in model.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([xt, yt], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    ''' 
    step_cache =step_cache*decay_rate+(1.0-decay_rate)*(grads[p]**2)
    -1.0*lr* grads/(np.sqrt(step_cache+1e-8))
    
    '''
    
    pup=[]
    
    for p, g in zip(model.values(), gshared):
        
        step_cache = theano.shared(p.get_value() * np.float32(0.))
        step_cache_t = step_cache*decay_rate+(1.0-decay_rate)*(g**2)
        p_t=p-(lr* g)/(T.sqrt(step_cache_t+1e-8))
        pup.append((step_cache, step_cache_t))
        pup.append((p, p_t))
        
    f_update = theano.function([lr], [], updates=pup,on_unused_input='ignore', name='rmsprop_f_update')


    return f_grad_shared, f_update

def train_net_two_layer(X,y,model,X_val,y_val,reg=0,b_size=70,lr=1e-2,n_epochs=40,
                        lr_decay=0.95,sample_batches=True,optimizer=sgd,
                        r_seed=1234,momentum=0.9,drop_rate=1.0):

    """
         lr_rate: is learning rate
         lr_decay: is learning rate decay
         optimizer: Type of parameters update, can be sgd, adam, rmsprop,adagrad
         b_size: is batach size
         n_epochs:  is number of training iterations
         r_seed: set random seed to make result reproducible
         sample_batches: if true will use mini-batches, if it is false, it uses full batch  
         momentum: is used for momentum's parameter update
         reg: is L2 regularization multiplier
         drop_rate: is dropout rate, p=1 mean no dropout. P should be 0< p <=1 
    """
    
    loss_hist= []
    N = X.shape[0]
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    val_acc_history = []
    
    X= X.astype(theano.config.floatX)
    X_val= X_val.astype(theano.config.floatX)

    y= y.astype('int32')
    y_val= y_val.astype('int32')
    lr_rate= float(lr)
        
    ########################
    # Step 0: Do some preprocess, params inits, and Build model, i.e feedforwad step
    ########################

    Xt,yt,cost,prediction,use_dropout= build_model(model,drop_rate,reg)
    f_cost = theano.function([Xt,yt], cost, name='f_cost')
    f_pred = theano.function([Xt],prediction,name='f_pred')

    
    '''
     get the gradients
    '''
    grads = T.grad(cost, wrt=list(model.values()))
    f_grad = theano.function([Xt,yt], grads, name='f_grad')
    
    '''
     now optimization 
    '''
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, model, grads,
                                        Xt, yt, cost)
    
    N = X.shape[0]
    
    if sample_batches:
        iter_per_epoch = int(N / b_size) # using SGD
    else:
        iter_per_epoch=1

    num_iters = n_epochs * iter_per_epoch
    rng = get_random_start_seed()   
    
    ########################
    # Start training main loop
    ########################
    for it in range(num_iters):
        
        if it % 400 == 0:  
            print('starting iteration ', it)

        if sample_batches:
            batch_mask = rng.choice(N,  b_size)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]
        else:
            # no SGD used, full batch 
            X_batch = X
            y_batch = y

        
        # evaluate cost and gradient
        use_dropout.set_value(1.0) #dropout is enabled 
        cost_loss= f_grad_shared(X_batch,y_batch)
        f_update(lr_rate)
        loss_hist.append(cost_loss)
        

        # Check if loss is not inf or Nan
        if np.isnan(cost_loss) or np.isinf(cost_loss):
            print ('Break epoch %d / %d: cost %f, lr %e'% (epoch, n_epochs, cost_loss, lr_rate))
            break

        # every epoch perform an evaluation on the validation set
        first_it = (it == 0)
        epoch_end = (it + 1) % iter_per_epoch == 0
        
        if first_it or epoch_end:
            
            if it > 0 and epoch_end:
                
                # decay the learning rate
                lr *= lr_decay
                epoch += 1
            
            # evaluate val accuracy
            use_dropout.set_value(0.0)  #disable dropout
            scores_val = f_pred(X_val)
            pred_class= np.argmax(scores_val,axis=1)
            val_acc= np.mean(y_val==pred_class)
            val_acc_history.append(val_acc)
            

            # keep track of the best model based on validation accuracy
            if val_acc > best_val_acc:
                
                # make a copy of the model
                best_val_acc = val_acc
                best_model = {}
                
                for p in model:
                    
                    best_model[p] = model[p].copy()

            print ('Finished epoch %d / %d: cost %f, val %f, lr %e'% (epoch, n_epochs, cost_loss, val_acc, lr_rate))

    print('Finished optimization. Best val acc: %f \n' % (best_val_acc, ) )
    
    return loss_hist,best_val_acc,best_model


  

 
