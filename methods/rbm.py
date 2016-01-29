import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
    
class RBM():
    def __init__(self, lr=1e-3, lr_decay=0.95, optimizer='sgd', 
                 b_size=100, numhids=10, n_epochs=10,CDK=1, r_seed=1234,
                 sample_batches=True,momentum=0.9,reg=0,img_dim=28, show_negdata=True):
        """
         lr: is learning rate
         lr_decay: is learning rate decay
         optimizer: Type of parameters update, can be sgd, adam, rmsprop,adagrad
         b_size: is batach size
         numhids: is number of hidden units
         n_epochs:  is number of training iterations
         CDk: is number of of Gibbs sampling steps
         rand_seed: set random seed to make result reproducible
         sample_batches: if true will use mini-batches, if it is false, it uses full batch  
         momentum: is used for momentum's parameter update
         reg: is L2 regularization multiplier
         show_negdata: is used to show images in negative phase 
        """
        self.lr=lr
        self.lr_decay=lr_decay
        self.optimizer=optimizer
        self.b_size=b_size
        self.numhids=numhids
        self.n_epochs=n_epochs
        self.CDK=CDK
        self.r_seed=r_seed
        self.rng=np.random.RandomState(r_seed) 
        self.sample_batches=sample_batches
        self.momentum=momentum
        self.reg=reg
        self.img_dim=img_dim
        self.show_negdata=show_negdata
    
    def sigmoid(self,z):
        """
         this functions computes sigmoid which is 1 /(1+np.exp(-x));
        """
        return 1.0 /(1.0+np.exp(-z))
    
    
    def train_with_CD(self,X):
        """
         This function train the RBM based on Contrastive Divergence
         X: is the training data that has N*D shape.
        """
        ##################################
        #Step 0: Some preprocessing
        ##################################
        N = X.shape[0]    # number of inputs
        numvis=X.shape[1] # input dimension or number of visible units
        epoch=0
        
        
        if self.sample_batches:
            iter_per_epoch = int(N / self.b_size) # using SGD
        else:
            iter_per_epoch=1 # using full batch
            
        num_iters = self.n_epochs * iter_per_epoch
        
        ##################################
        #Step 1: Paramters init
        ##################################
        
        #W is D*K matrix for visible-hidden weights
        W=0.02*self.rng.randn(numvis,self.numhids)
        b=np.zeros((self.numhids))         # b is K*1 matrix for hidden bases
        c=np.zeros((numvis,))              # c is D*1 matrix for visible bases 

        
        #W is D*K matrix for visible-hidden weights
        W_cache=np.zeros((numvis,self.numhids)) 
        b_cache=np.zeros((self.numhids))         # b is K*1 matrix for hidden bases
        c_cache=np.zeros((numvis,))              # c is D*1 matrix for visible bases 
        
        
        
        ##################################
        #Step 1: Start training with Contrastive Divergence
        ##################################
        for it in range(num_iters):
            
            if it % 400 == 0:
                print('starting iteration ', it)

            ## gets the minibatches    
            if self.sample_batches:
                batch_mask = np.random.choice(N, self.b_size)
                X_batch = X[batch_mask]
                
            else:
                # no SGD used, full gradient descent
                X_batch = X
                
            ###############
            ## Positive step: 
            #  1) samples h ~ p(h|v)
            #  2) then p(hj=1|V) > U[0,1] (sample)
            ############### 
            poshidprob=self.sigmoid( np.dot(X_batch,W)+ b )
            poshidstates=np.double(self.rng.rand(*poshidprob.shape) < poshidprob)
            
            ###############
            ## Negative step: 
            #  samples v' from p(v'|h) and h'~ p(h'|v')
            ############### 
            neghidstates=poshidstates.copy()
            for cd in range(self.CDK):
                # negative data from h->v'
                negdata=self.sigmoid( np.dot(neghidstates,W.T)+ c )

                # from v'-->h' (hidden unit inference)
                neghidprob=self.sigmoid(np.dot(negdata,W)+ b);
                # then use p(h'j=1|v') > U[0,1](samples)
                neghidstates=np.double(self.rng.rand(*neghidstates.shape) < neghidstates)
                
            ###############    
            # Gradient(approximate) computation step
            ###############
            dW=(np.dot(X_batch.T,poshidprob)-np.dot(negdata.T,neghidprob))/float(X_batch.shape[0])
            db=np.mean(poshidprob)-np.mean(neghidprob)
            dc=np.mean(X_batch)-np.mean(negdata)
            
            #monitor reconstruction error
            recon=self.sigmoid( np.dot(poshidprob,W.T)+ c )
            recon_error=np.linalg.norm(X_batch-recon)/float(X_batch.shape[0])
            
            ###############
            # Paramater Update
            ###############
            if self.optimizer=='momentum':
                W_cache=self.momentum*W_cache+self.lr*(dW-self.reg*W)
                b_cache=self.momentum*b_cache+self.lr*db
                c_cache=self.momentum*c_cache+self.lr*dc
                W+=W_cache
                b+=b_cache
                c+=c_cache
            else:
                raise ValueError('Optimizer type is not supported!')

            ###############
            # Print and show progress 
            ###############
                
            first_it = (it == 0)
            epoch_end = (it + 1) % iter_per_epoch == 0
            if first_it or epoch_end:
                if it > 0 and epoch_end:
                    self.lr *= self.lr_decay
                    epoch += 1               
                print ('Finished epoch %d / %d: reconError %f, lr_rate %e'% (epoch, self.n_epochs, recon_error,self.lr)) 
                
                #show images per 3 epochs
                if self.show_negdata==True and epoch %2==0: 
                    show_reconstructed_img(recon,self.img_dim)
        
        
        # Keep the parameters in the dic
        model={}
        model['W']=W
        model['b']=b
        model['c']=c

        return  model
 

def show_reconstructed_img(input_data,img_dim):
    """
     This function shows reconstructed images for the RBM 
     input_data should be N*D where N is number of samples and D is data dimension 
     N should be larger of 20
     This functions 
    """
    N=input_data.shape[0]
    num_row= 5
    num_col= 5    

    if  N < num_row*num_col :
        raise ValueError('The batch size should be larger than %d  if this features is wanted otherwise it can be disabled by setting show_negdata=False\n'% (num_row*num_col))
        
    num_samples=num_row*num_col
    samples_idx=np.random.choice(N,num_samples,replace=False)    

    
    plt.figure(figsize = (num_row,num_col))
    gs1 = gridspec.GridSpec(num_row,num_col)
    gs1.update(wspace=0.01, hspace=0.01) # set the spacing between axes. 
    
    for i,idx in enumerate(samples_idx):
        plt.subplot(num_row,num_col,i+1)
        plt.imshow(input_data[idx].reshape(img_dim,img_dim))
        plt.axis('off')
    plt.show()
    plt.close()
