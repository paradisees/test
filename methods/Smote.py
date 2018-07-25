import numpy as np
from sklearn.neighbors import NearestNeighbors

from sentiment import random


class Smote:
    def __init__(self,samples,N=10,k=2):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn= random.randint(0, self.k - 1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap= sentiment.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
a=np.array([[1,0,1,0,1,1,1,1,1,],[1,0,0,0,1,0,0,0,1]])
s=Smote(a,N=100)
print (s.over_sampling())