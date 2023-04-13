import numpy as np
import matplotlib.pyplot as plt


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
    return discovered_truths,weights

def modified_CRH(sensory_data,M,N,threshold,weight_threshold):
    # filter out those whose weights is quite low
    discovered_truths,weights=CRH(sensory_data,M,N,threshold)
    filtered_weights_indices=np.where(weights>np.sum(weights)/N*weight_threshold)
    filtered_sensory_data=sensory_data[filtered_weights_indices]
    filtered_discovered_truths,filtered_weights=CRH(filtered_sensory_data,M,filtered_sensory_data.shape[0],threshold)
    return filtered_discovered_truths,filtered_weights

def RMSE(discovered_truths, truths,M):
    return np.sqrt(np.sum((discovered_truths-truths)**2)/M)

def outliers(LOW,HIGH,lambdae,M,N,scala,worker_ids=[],position={}):
    # the worker_ids is a 1D vector, position is a 2D matrix
    # the function returns the sensory data,truths,noise_level

    truths=np.random.randint(LOW,HIGH,M)
    # generate noise level with exponential distribution
    noise_level=np.random.exponential(lambdae,size=N)
    # np.random.seed(2000)
    sensory_data=np.zeros((N,M)) # id first objects then

    if worker_ids==[]:
        for i in range(N):
            sensory_data[i]=truths+np.random.normal(0,noise_level[i],M)
    else:
        for i in range(N):
            for j in range(M):
                if i in worker_ids and j in position[i]:
                    sensory_data[i,j]=truths[j]+np.random.normal(0,noise_level[i])+scala
                else:
                    sensory_data[i,j]=truths[j]+np.random.normal(0,noise_level[i])
    return truths,sensory_data

def one_worker_all_element(LOW,HIGH,M,N,scala,threshold,lambdae,switch=0,weight_threshold=0):
    rmsel=[]
    if switch==0:
        for i in range(100):
            ids=list(np.random.choice(N,size=1))
            position={}
            position[ids[0]]=list(np.arange(M))
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=CRH(sensory_data,M,N,threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    else:
        for i in range(100):
            ids=list(np.random.choice(N,size=1))
            position={}
            position[ids[0]]=list(np.arange(M))
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=modified_CRH(sensory_data,M,N,threshold,weight_threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    return rmsel

def one_worker_random_element(LOW,HIGH,M,N,scala,threshold,lambdae,switch=0,weight_threshold=0):
    rmsel=[]
    if switch==0:
        for i in range(100):
            ids=list(np.random.choice(N,size=1,replace=False))
            position={}
            position[ids[0]]=np.random.choice(M,size=1,replace=False)
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=CRH(sensory_data,M,N,threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    else:
        for i in range(100):
            ids=list(np.random.choice(N,size=1,replace=False))
            position={}
            position[ids[0]]=np.random.choice(M,size=1,replace=False)
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=modified_CRH(sensory_data,M,N,threshold,weight_threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    return rmsel

def random_worker_random_elements(LOW,HIGH,M,N,scala,threshold,lambdae,switch=0,weight_threshold=0):
    rmsel=[]
    f1=0.05
    f2=0.1
    if switch==0:
        for i in range(100):
            ids=list(np.random.choice(N,size=int(np.ceil(f1*N)),replace=False))
            position={}
            for j in ids:
                position[j]=list(np.random.choice(M,size=int(np.ceil(f2*M)),replace=False))
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=CRH(sensory_data,M,N,threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    else:
        for i in range(100):
            ids=list(np.random.choice(N,size=int(np.ceil(f1*N)),replace=False))
            position={}
            for j in ids:
                position[j]=list(np.random.choice(M,size=int(np.ceil(f2*M)),replace=False))
            truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
            discovered_truths,weights=modified_CRH(sensory_data,M,N,threshold,weight_threshold)
            rmsel.append(RMSE(truths, discovered_truths,M))
    return rmsel

def no_ouliers(LOW,HIGH,M,N,scala,threshold,lambdae):
    rmsel=[]
    for i in range(100):
        truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala)
        discovered_truths,weights=CRH(sensory_data,M,N,threshold)
        rmsel.append(RMSE(truths, discovered_truths,M))
    return rmsel

def effect_of_outliers(LOW,HIGH,M,N,scala,threshold,lambdae,switch=0,weight_threshold=0):
    # situation 1: one worker, all elements
    rmsel1=one_worker_all_element(LOW,HIGH,M,N,scala,threshold,lambdae,switch,weight_threshold)
    
    # situation 2: one worker, one random elements
    rmsel2=one_worker_random_element(LOW,HIGH,M,N,scala,threshold,lambdae,switch,weight_threshold)
    
    # situation 3: random workers, random elements
    rmsel3=random_worker_random_elements(LOW,HIGH,M,N,scala,threshold,lambdae,switch,weight_threshold)
    
    # situation 4: No outliers
    rmsel4=no_ouliers(LOW,HIGH,M,N,scala,threshold,lambdae)

    plt.scatter(list(range(100)),rmsel1,color="blue")
    plt.plot(list(range(100)),np.zeros(100)+np.sum(rmsel1)/100,color="blue",label="one worker, all elements")
    plt.scatter(list(range(100)),rmsel2,color="red")
    plt.plot(list(range(100)),np.zeros(100)+np.sum(rmsel2)/100,color="red",label="one worker, one random elements")
    plt.scatter(list(range(100)),rmsel3,color="green")
    plt.plot(list(range(100)),np.zeros(100)+np.sum(rmsel3)/100,color="green",label="random workers, random elements (limited fraction)")
    plt.scatter(list(range(100)),rmsel4,color="black")
    plt.plot(list(range(100)),np.zeros(100)+np.sum(rmsel4)/100,color="black",label="No scala")
    plt.legend()
    plt.show()
def discover_outlier_from_truths(LOW,HIGH,M,N,scala,threshold,lambdae):
    f1=0.1
    f2=0.01
    ids=list(np.random.choice(N,size=int(np.ceil(f1*N)),replace=False))
    position={}
    for j in ids:
        position[j]=list(np.random.choice(M,size=int(np.ceil(f2*M)),replace=False))
    truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
    discovered_truths,weights=CRH(sensory_data,M,N,threshold)

    # ids=list(np.random.choice(N,size=1))
    # position={}
    # position[ids[0]]=list(np.arange(M))
    # truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
    # discovered_truths,weights=CRH(sensory_data,M,N,threshold)
    
    # ids=list(np.random.choice(N,size=1,replace=False))
    # position={}
    # position[ids[0]]=np.random.choice(M,size=1,replace=False)
    # truths,sensory_data=outliers(LOW,HIGH,lambdae,M,N,scala,ids,position)
    # discovered_truths,weights=CRH(sensory_data,M,N,threshold)

    outlier_elements=np.empty(shape=0)
    for j in ids:
        outlier_elements=np.union1d(outlier_elements,position[j])
    for j in outlier_elements:
        plt.axvline(x=j,ymin=0,ymax=np.abs(discovered_truths-truths)[int(j)],color="red")
    plt.plot(np.arange(M),np.abs(discovered_truths-truths))
    plt.show()
def effect_of_weight_filtering(LOW,HIGH,M,N,scala,threshold,weight_threshold):
    pass


if __name__=='__main__':
    #---------------------------------------
    # default parameters
    M=100
    N=100
    threshold=1e-6
    LOW=-1e3
    HIGH=1e3-1
    lambdae=10
    #----------------------------------------------
    # passive attack
    # scala for adding outlier
    scala=(np.abs(LOW)+np.abs(HIGH))*10
    # weight threshold
    weight_threshold=0.25
    effect_of_outliers(LOW,HIGH,M,N,scala,threshold,lambdae,0,weight_threshold)
    # discover_outlier_from_truths(LOW,HIGH,M,N,scala,threshold,lambdae)

    