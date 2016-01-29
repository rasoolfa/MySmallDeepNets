import numpy as np

class Train_nnets:

    def train(self,X,y,model,loss_func,X_val,y_val,reg=0,b_size=100,lr=1e-2, lr_decay=0.95, 
                 n_epochs=40,sample_batches=True,optimizer='sgd',r_seed=1234,momentum=0.9,drop_rate=1.0):
        """
         lr: is learning rate
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
         
        ########################
        # Step 0: Do some preprocess and params inits
        ########################
            
        loss_hist=[]
        step_cache = {} 
        N = X.shape[0]
        epoch = 0
        best_val_acc = 0.0
        best_model = {}
        val_acc_history = []
        best_model={}
        rng=np.random.RandomState(r_seed) 

    
        if sample_batches:
            iter_per_epoch = int(N / b_size) # using SGD
        else:
            iter_per_epoch=1

        num_iters = n_epochs * iter_per_epoch
        print optimizer
        ########################
        # Start training main loop
        ########################
    
        for it in range(num_iters):
        
            if it % 400 == 0:  
                print('starting iteration ', it)

            if sample_batches:
                batch_mask = rng.choice(N, b_size)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
            else:
                # no SGD used, full batch
                X_batch = X
                y_batch = y

        
            # evaluate cost and gradient
            cost,grads=loss_func(X_batch,model,y_batch,reg=reg,p=drop_rate)
            loss_hist.append(cost)


            ##################
            # Perform parameters update 
            ##################
            for param in model:
                
                if (optimizer=='sgd'):
                    
                    #####################
                    # SGD params update
                    #####################
                    dW = -1.0*lr * grads[param]
                    
                elif(optimizer=='adagrad'):
                    
                    #####################
                    # adagrad params update
                    #####################
                    if not param in step_cache:
                        step_cache[param] = np.zeros(grads[param].shape)
                    step_cache[param] =step_cache[param]+ (grads[param]**2)
                    dW = -lr * grads[param]  / np.sqrt(step_cache[param] + 1e-8)
                    
                elif(optimizer=='rmsprop'):
                    
                    #####################
                    # rmsprop params update
                    #####################
                    decay_rate = 0.99
                    if not param in step_cache:
                        step_cache[param] = np.zeros(grads[param].shape)
                    step_cache[param] =step_cache[param]*decay_rate+(1.0-decay_rate)*(grads[param]**2)
                    dW= -lr * grads[param]/(np.sqrt(step_cache[param]+1e-8))
                    
                else:
                    
                    raise ValueError('%s is not supported'%optimizer)

                model[param] += dW 

            ###############
            # Print and show progress 
            ###############  
        
            # every epoch perform an evaluation on the validation set
            first_it = (it == 0)
            epoch_end = (it + 1) % iter_per_epoch == 0
            if first_it or epoch_end:
            
            	    if it > 0 and epoch_end:
                
                	# decay the learning rate
               		lr *= lr_decay
                	epoch += 1
            
		    # evaluate val accuracy
		    scores_val = loss_func(X_val, model)
		    pred_class=np.argmax(scores_val,axis=1)
		    val_acc=np.mean(y_val==pred_class)
		    val_acc_history.append(val_acc)
		    
		    # keep track of the best model based on validation accuracy
		    if val_acc > best_val_acc:
		        
		        # make a copy of the model
		        best_val_acc = val_acc
		        best_model = {}
		        
		        for p in model:
		            
		            best_model[p] = model[p].copy()
		            
		    print ('Finished epoch %d / %d: cost %f, val %f, lr %e'% (epoch, n_epochs, cost, val_acc, lr))

            
        
        print('Finished optimization. Best val acc: %f \n' % (best_val_acc, ) )
        
        return loss_hist,best_val_acc,best_model

