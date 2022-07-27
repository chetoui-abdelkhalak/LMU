import numpy as np
import scipy as sp
import scipy.stats
import pickle
from torch.utils.data import Dataset
from utils.create_mixtures import My_dataset
import matplotlib.image as img

def scale(X, x_min=0, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def create_data(nb,W,noise=True):
    V=np.zeros((nb,65,2000))
    H=np.zeros((nb,4,2000))
    
    for i in range(4001,4001+nb):
        path1='C:/Users/abdel/Desktop/stage/test_set/test_set/cats/cat.'+str(i)+'.jpg'
        path2='C:/Users/abdel/Desktop/stage/test_set/test_set/cats/cat.'+str(i+1)+'.jpg'
        path3='C:/Users/abdel/Desktop/stage/test_set/test_set/cats/cat.'+str(i+2)+'.jpg'
        path4='C:/Users/abdel/Desktop/stage/test_set/test_set/cats/cat.'+str(i+3)+'.jpg'
        test1=img.imread(path1)
        test2=img.imread(path2)
        test3=img.imread(path3)
        test4=img.imread(path4)
        test1=test1.reshape((3,1,2000))
        test2=test2.reshape((3,1,2000))
        test3=test3.reshape((3,1,2000))
        test4=test4.reshape((3,1,2000))

        H[i-4001]=np.vstack((np.array(test1[0])/np.sum(test1[0]),np.array(test2[0])/np.sum(test2[0]),np.array(test3[0])/np.sum(test3[0]),np.array(test4[0])/np.sum(test4[0])))

        V[i-4001] = np.matmul(W[i-4001],H[i-4001])
        if noise:
            N = np.random.rand(V[i-4001].shape[0],V[i-4001].shape[1])
            N = 10.**(-8)*np.linalg.norm(V[i-4001])/np.linalg.norm(N)*N
            V[i-4001] = V[i-4001] + N


    return (V,W,H)

class dataset_cat(Dataset):
    def __init__(self,W):

        self.V,self.W,self.H=create_data(900, W,noise=True)
    def __getitem__(self, item):
        return self.V[item],self.W[item],self.H[item]
    def __len__(self):
        return len(self.V)
    
if __name__ == '__main__': 		
    with open('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/dataset_4_sources_realistic.pickle', 'rb') as f:
        ds4 = pickle.load(f)
    W=ds4[:900][1]
    dataset=dataset_cat(W)
    with open('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/dataset_cat_normalized.pkl', 'wb') as f:
        pickle.dump(dataset, f)
