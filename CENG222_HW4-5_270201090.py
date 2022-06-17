import numpy as np
from matplotlib import pyplot as plt

#Part a

def f(theta,x):
    return ((2*theta)**2)/(x**3)

def MoM(array):
    
    mean=sum(array)/len(array)
    theta=(mean/2)
    return theta

def MLE(array):

    return min(array)

part_a_array=[0.3,0.6,0.8,0.9]

print("MoM estimate for the sample is ",MoM(part_a_array))
print("MLE estimate for the sample is ",MLE(part_a_array))



#Part b

def Inverse_Transform_Method(theta,x):
    return ((theta**2)/(1-x))**0.5

part_b_theta=2.4
sizeOfP=10000000
P_array=[]

for i in range(sizeOfP):
    newnum=np.random.uniform()
    P_array.append(Inverse_Transform_Method(part_b_theta,newnum))
plt.figure()

plt.plot(np.linspace(2.5,20.0,100),f(part_b_theta,np.linspace(2.5,20.0,100)),color='blue',label='PDF')
plt.hist(P_array,color='orange',bins=np.linspace(2.5,20.0,100),alpha=0.5,label='Histogram',density=True)
plt.legend()
plt.show()

#Part c


N=[1,2,3,4,5,10,50,100,500,1000]

def part_c_estimation(population,N):
    size=100000
    MoM_array=[]
    MLE_array=[]
    for i in range(size):
        counter=N
        array=[]
        while counter>0:
            randomIndex=np.random.randint(0,len(population))
            array.append(population[randomIndex])
            counter-=1
        MoM_array.append(MoM(array))
        MLE_array.append(MLE(array))
    plt.figure()
    plt.hist(MoM_array,bins=np.linspace(0,4.8,100),density=True,alpha=0.5,label=f"MoM estimate histogram for N= {N}")
    plt.hist(MLE_array,bins=np.linspace(0,4.8,100),density=True,alpha=0.5,label=f"MoM estimate histogram for N= {N}")
    plt.legend()
    
    return MoM_array,MLE_array

for i in N:
    MoM_est,MLE_est=part_c_estimation(P_array,i)
    print("For N = ",i,":")
    print("MoM estimate mean: ",np.mean(MoM_est),"    MoM estimate std: ",np.std(MoM_est))
    print("MLE estimate mean: ",np.mean(MLE_est),"    MLE estimate std: ",np.std(MLE_est))

plt.show()
