def linearCCA(H1,H2,dim=300,rcov1=0.1,rcov2=0.01):
    '''
     H1 and H2 are NxD matrices containing samples rowwise.
     dim is the desired dimensionality of CCA space.
     r is the regularization of autocovariance for computing the correlation.
     A and B are the transformation matrix for view 1 and view 2.
     m1 and m2 are the mean for view 1 and view 2.
     D is the vector of singular values.
   '''
    N,d1=np.shape(H1)
    _,d2=np.shape(H2)
    
    ### remove mean
    m1=np.mean(H1,axis=0)
    H1=H1-m1
    
    m2=np.mean(H2,axis=0)
    H2=H2-m2
    
    ## Estimate covariances, with regularization
    S11=np.dot(H1.T,H1)/(N-1.0)+rcov1*np.identity(d1)
    S12=np.dot(H1.T,H2)/(N-1.0)
    S22=np.dot(H2.T,H2)/(N-1.0)+rcov2*np.identity(d2)
    
    #For numerical stability.   
    D1,V1=scipy.linalg.eig(S11)
    D2,V2=scipy.linalg.eig(S22)
    D1 = np.real(D1)
    idx1 = D1>1e-12
    D1 = D1[idx1] 
    V1 = V1[:,idx1]
    
    D2 = np.real(D2)
    idx2 = D2>1e-12
    D2 = D2[idx2]
    V2 = V2[:,idx2]
   
    ## Form normalized covariance matrix
    #singular value decomposition
    K11 = np.dot(np.dot(V1,np.diag(D1**(-1/2.0))),V1.T)
    K22 = np.dot(np.dot(V2,np.diag(D2**(-1/2.0))),V2.T)
    T =  np.dot(np.dot(K11,S12),K22)
    U, D, V= np.linalg.svd(T, full_matrices= False)
    
    #####################################################
    # important the [U,D,V]=svd(T) in matlab is same as python
    # as follow:
    # Um=U
    # Dm=D
    # Vm=V.T
    V=V.T
    #####################################################
    
    A = np.dot(K11,U[:,0:dim])
    B = np.dot(K22,V[:,0:dim])
    D = D[0:dim]
    
    return A,B,m1,m2,D

