import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
'''
Each algorithm proceeds in 4 steps:
    1.Generate truths and noise level for each worker
    2.add inherent noise
    3.decides privacy level and adds privacy noise
    4.perform CRH on sensory data to obtain RMSE and return
'''

def CRH(sensory_data,M,N,threshold):
    weights=np.ones(N)
    discovered_truths=np.zeros(M)
    last_truths=np.zeros(M)
    while True:
        last_truths=np.copy(discovered_truths)
        weights_sum=np.sum(weights)
        for i in range(M):
            discovered_truths[i]=np.sum(sensory_data[:,i]*weights)/weights_sum
        dis_sum=0
        for i in range(N):
            dis_sum+=np.sum((sensory_data[i]-discovered_truths)**2)
        for i in range(N):
            weights[i]=np.log(dis_sum/np.sum((sensory_data[i]-discovered_truths)**2))
        if(np.sum(np.abs(discovered_truths-last_truths))<threshold):
            break
    return discovered_truths


def RMSE(discovered_truths, truths,M):
    return np.sqrt(np.sum((discovered_truths-truths)**2)/M)

def standard(M,N,LOW,HIGH,threshold):
    # M,N: number of objects and workers
    # LOW, HIGH: range of truths
    # lambdae: hyper parameter for sampling noise level
    # threshold: threshold for CRH to stop iteration
    #------------------------------------------------------------------------------
    # generate random data
    np.random.seed(30)
    truths=np.random.randint(LOW,HIGH,M)
    # generate noise level with exponential distribution
    noise_level=np.random.randint(0,100,N)
    # np.random.seed(2000)
    sensory_data=np.zeros((N,M)) # id first objects then
    for i in range(N):
        sensory_data[i]=truths+np.random.normal(0,noise_level[i],M)
    #-------------------------------------------------------------------------------
    # perform CRH and return RMSE results
    discovered_truths=CRH(sensory_data,M,N,threshold)
    return RMSE(discovered_truths,truths,M)

def ICDCS2020(M,N,LOW,HIGH,lambda1,lambda2,threshold):
    # M,N: number of objects and workers
    # LOW, HIGH: range of truths
    # lambda1: hyper parameter for sampling noise level
    # lambda2: hyper parameter for sampling privacy level
    # threshold: threshold for CRH to stop iteration
    #------------------------------------------------------------------------------
    # generate random data
    np.random.seed(100)
    truths=np.random.randint(LOW,HIGH,M)
    # notice that ICDCS2020 assumes a hyper-parameter lambda1 to generate noise level
    # thus resulting in different sensory data
    noise_level=np.random.exponential(1/lambda1,size=N)
    sensory_data=np.zeros((N,M)) # id first objects then
    for i in range(N):
        sensory_data[i]=truths+np.random.normal(0,noise_level[i],M)
    discovered_truths=CRH(sensory_data,M,N,threshold)
    rmse1=RMSE(discovered_truths,truths,M)
    #-------------------------------------------------------------------------------
    # after obtaining random sensory data, it requires to add privacy noise
    privacy_level=np.random.exponential(1/lambda2,N)
    # each user samples Gaussian noise according to his sampled variance
    for i in range(N):
        sensory_data[i]+=np.random.normal(0,privacy_level[i],M)
    #------------------------------------------------------------------------------
    # after perturbation, perform CRH and return MSE results
    discovered_truths=CRH(sensory_data,M,N,threshold)
    rmse2=RMSE(discovered_truths,truths,M)
    return rmse1,rmse2


def ICDCS2020_v1(M,N,LOW,HIGH,nLOW,nHIGH,lambda2,threshold):
    # This version 1 replaces the way of generating noise level
    # M,N: number of objects and workers
    # LOW, HIGH: range of truths
    # lambda1: hyper parameter for sampling noise level
    # lambda2: hyper parameter for sampling privacy level
    # threshold: threshold for CRH to stop iteration
    #------------------------------------------------------------------------------
    # generate random data
    np.random.seed(10)
    truths=np.random.randint(LOW,HIGH,M)
    # generate noise level same as others
    noise_level=np.random.randint(nLOW,nHIGH,N)
    np.random.seed(20)
    sensory_data=np.zeros((N,M)) # id first objects then
    for i in range(N):
        sensory_data[i]=truths+np.random.normal(0,noise_level[i],M)
    #-------------------------------------------------------------------------------
    # after obtaining random sensory data, it requires to add privacy noise
    np.random.seed(30)
    privacy_level=np.random.exponential(lambda2,N)
    # each user samples Gaussian noise according to his sampled variance
    for i in range(N):
        sensory_data[i]+=np.random.normal(0,privacy_level[i],M)
    #------------------------------------------------------------------------------
    # after perturbation, perform CRH and return MSE results
    discovered_truths=CRH(sensory_data,M,N,threshold)
    return RMSE(discovered_truths,truths,M)

def TMC2021CDP(M,N,LOW,HIGH,lambdae,budget,threshold):
    # M,N: number of objects and workers
    # LOW, HIGH: range of truths
    # budget: privacy budget for Laplace mechanism (sensitivity can be computed from LOW and HIGH)
    # threshold: threshold for CRH to stop iteration
    #-----------------------------------------------------------------------------
    # generate random data
    np.random.seed(10)
    truths=np.random.randint(LOW,HIGH,M)
    # noise generated from np.random.randint is much larger than ICDCS2020
    noise_level=np.random.exponential(lambdae,N)
    np.random.seed(20)
    sensory_data=np.zeros((N,M)) # id first objects then
    for i in range(N):
        sensory_data[i]=truths+np.random.normal(0,noise_level[i],M)
    #----------------------------------------------------------------------------
    # add privacy noise
    np.random.seed(30)
    # each user samples Gaussian noise according to his sampled variance
    for i in range(N):
        sensory_data[i]+=np.random.laplace(0,(HIGH-LOW)/budget[i],M)
    #----------------------------------------------------------------------------
    discovered_truths=CRH(sensory_data,M,N,threshold)
    return RMSE(discovered_truths,truths)

def standard_test_scale_lambda(M,N,LOW,HIGH,threshold):
    rmsel1=[]
    rmsel2=[]
    for lambdae in range(1,300):
        rmse=standard(M,N,LOW,HIGH,lambdae/10,threshold)
        rmsel1.append(rmse[0])
        rmsel2.append(rmse[1])
    plt.plot(np.arange(0.1,30,0.1),rmsel1,'blue',label='Discovered truths')
    plt.plot(np.arange(0.1,30,0.1),rmsel2,'red',label='Sensory data')
    plt.xlabel("Scale of $\lambda$")
    plt.ylabel("RMSE (the lower, the better)")
    plt.legend()
    plt.show()

def standard_test_number_workers(M,LOW,HIGH,threshold):
    rmsel=[]
    for n in range(M//10,10*M,M//10):
        rmsel.append(standard(M,n,LOW,HIGH,threshold))
    return rmsel

def TMC2021_test(M,N,LOW,HIGH,nLOW,nHIGH,threshold):
    # TMC2021CDP(M,N,LOW,HIGH,nLOW,nHIGH,budget,threshold)
    rmsel=[]
    for b in np.arange(1,30,0.1):
        budget=np.zeros(N)+b
        rmsel.append(TMC2021CDP(M,N,LOW,HIGH,10,budget,threshold))
    plt.plot(np.arange(1,30,0.1),rmsel)
    plt.show()

def ICDCS2020_test_scale_lambda(M,N,LOW,HIGH,threshold):
    rmsel1=[]
    rmsel2=[]
    # fix c at 2.0
    for lambdae in range(1,300):
        rmse=ICDCS2020(M,N,LOW,HIGH,lambdae/10,2*lambdae/10,threshold)
        rmsel1.append(rmse[0])
        rmsel2.append(rmse[1])
    plt.plot(np.arange(0.1,30,0.1),rmsel1,'red',linewidth=0.5,label='Discovered truths before privacy')
    plt.plot(np.arange(0.1,30,0.1),rmsel2,'black',linewidth=0.5,label='Discovered truths before privacy')
    plt.xlabel("Scale of $\lambda$")
    plt.ylabel("RMSE (the lower, the better)")
    plt.legend()
    plt.show()



if __name__=='__main__':
    #---------------------------------------
    # default parameters
    M=100
    N=100
    threshold=1e-6
    LOW=-1e3
    HIGH=1e3-1
    lambdae=10
    #----------------------------------------
    # standard test
    # standard_test_scale_lambda(M,N,LOW,HIGH,threshold)
    rmsel=[]
    for m in range(100,210,10):
        rmsel=standard_test_number_workers(m,LOW,HIGH,threshold)
        plt.plot(np.arange(0.1,10,0.1),rmsel,label="M=%d"%m)
    plt.xlabel("Number of workers / number of objects")
    plt.ylabel("RMSE(the lower, the better)")
    plt.legend()
    plt.show()


    
    #------------------------------------------
    # ICDCS2020 test
    # ICDCS2020_test_scale_lambda(M,N,LOW,HIGH,threshold)
    #------------------------------------------
    # TMC2021 test
    # TMC2021_test(M,N,LOW,HIGH,nLOW,nHIGH,threshold)
    