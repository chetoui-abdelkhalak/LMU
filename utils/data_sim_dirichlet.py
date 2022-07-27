import numpy as np
import scipy as sp
import scipy.stats
import pickle
from torch.utils.data import Dataset
from utils.create_mixtures import My_dataset


def create_data(nb,W,noise=True):
    V=np.zeros((nb,65,2000))
    H=np.zeros((nb,4,2000))
    
    for i in range(nb):
        alpha=np.random.rand()*np.ones(2000)
        H[i]=sp.stats.dirichlet(alpha).rvs(size=4)

    for i in range(nb):
        V[i] = np.matmul(W[i],H[i])
        if noise:
            N = np.random.rand(V[i].shape[0],V[i].shape[1])
            N = 10.**(-4)*np.linalg.norm(V[i])/np.linalg.norm(N)*N
            V[i] = V[i] + N
    return (V,W,H)

class dataset_dirichlet(Dataset):
    def __init__(self,W):
        '''
        self.H = create_data(500)
        self.V,self.W=W@self.H,W
        '''
        self.V,self.W,self.H=create_data(900, W)
    def __getitem__(self, item):
        return self.V[item],self.W[item],self.H[item]
    def __len__(self):
        return len(self.V)
    
if __name__ == '__main__': 		
    with open('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/dataset_4_sources_realistic.pickle', 'rb') as f:
        ds4 = pickle.load(f)
    W=ds4[:900][1]
    dataset=dataset_dirichlet(W)
    with open('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/dataset_sim_dirichlet_random_2000.pkl', 'wb') as f:
        pickle.dump(dataset, f)
