import numpy as np
import matplotlib.pyplot as plt

def RMSE(discovered_truths, truths,M):
    return np.sqrt(np.sum((discovered_truths-truths)**2)/M)
def CRH(sensory_data,M,N,threshold):
    weights=np.ones(N)
    discovered_truths=np.zeros(M)
    last_truths=np.zeros(M)
    ite_count=0
    while True:
        ite_count+=1
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
    return ite_count


def sequence_CRH(M,N,T,threshold,LOW,HIGH,lambdae):
    # generate truths
    truths=np.random.randint(LOW,HIGH,size=(T,M))
    # generate noise level with exponential distribution
    noise_level=np.random.exponential(lambdae,size=N)
    # np.random.seed(2000)
    sensory_data=np.zeros(size=(T,N,M)) # time epoch first, id then objects finally
    for i in range(T):
        for j in range(N):
            sensory_data[i,j]+=truths[i]+np.random.normal(0,noise_level[j],size=M)
    ite_count=[]
    for i in range(T):
        pass
    


if __name__=='__main__':
    M=100
    N=100
    T=100
    threshold=1e-6
    LOW=-1e3
    HIGH=1e3-1
    lambdae=10
