import torch
import pickle
import numpy as np
from create_mixtures import My_dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from munkres import Munkres
from sklearn.preprocessing import normalize

def correctPerm(W0_en,W_en):
    # [WPerm,Jperm,err] = correctPerm(W0,W)
    # Correct the permutation so that W becomes the closest to W0.
    
    #W0_en = W0
    #W_en = W
    W0 = W0_en.copy()
    W = W_en.copy()
    
    #W0 = norm_col(W0)
    W0=normalize(W0,axis=0,norm='l2')
    #W = norm_col(W)
    W=normalize(W,axis=0,norm='l2')
    #print(W0)
    costmat = -W0.T@W; # Avec Munkres, il faut bien un -
    #print(costmat)
    
    m = Munkres()
    Jperm = m.compute(costmat.tolist())
    #print(Jperm)
    
    WPerm = np.zeros(np.shape(W0))
    indPerm = np.zeros(np.shape(W0_en)[1])
    
    for ii in range(W0_en.shape[1]):
        WPerm[:,ii] = W_en[:,Jperm[ii][1]]
        indPerm[ii] = Jperm[ii][1]
        
    return WPerm,indPerm.astype(int) 

def test_LMU2(V,path):
    H = 10**(-2)*np.ones((4, V.shape[1]))
    W = 10**(-2)*np.ones((V.shape[0], 4))
    act = nn.LeakyReLU(-0.1)
    model=torch.load(path)
    for i in range(25):
        A = model['model_state_dict']['deep_nmfs.'+str(i)+'.Aw']
        plt.plot(A[0])
        plt.show()
        Gp = W.T@W@H
        Gn = W.T@V
        B = np.array(act(A))[0]
        H = H*((Gn/Gp))
        Gp_w = W@H@(H.T)
        Gn_w = V@(H.T)
        W = W*(B*(Gn_w/Gp_w))
    return H, W

def test_LMU1(V,W,path):
    H = 10**(-2)*np.ones((4, V.shape[1]))
    #W = 10**(-2)*np.ones((V.shape[0], 4))

    act = nn.LeakyReLU(-0.1)
    model=torch.load(path)
    for i in range(25):
        A = model['model_state_dict']['deep_nmfs.'+str(i)+'.A']

        Gp = W.T@W@H
        Gn = W.T@V
        B = np.array(act(A))[0]
        H = H*(B@(Gn/Gp))
        Gp_w = W@H@(H.T)
        Gn_w = V@(H.T)
        W = W*((Gn_w/Gp_w))
    return H, W


def MU(V,W,H):
    Gp = W.T@W@H
    Gn = W.T@V
    H = H*((Gn/Gp))
    Gp_w = W@H@(H.T)
    Gn_w = V@(H.T)
    W = W*(Gn_w/Gp_w)
    return H, W


def scale(col, min, max):
    range = col.max() - col.min()
    a = (col - col.min()) / range
    return a * (max - min) + min

if __name__ == '__main__': 

    nb = 1
    noise = True
    loss = []

    with open('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/dataset_4_sources_realistic.pickle', 'rb') as f:
        ds4 = pickle.load(f)
        spectres = np.load(
            'C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/data/sources.pkl', allow_pickle=True)

    H=np.vstack((spectres['sync'].reshape(-1),spectres['therm'].reshape(-1),spectres['fe1'].reshape(-1),spectres['fe2'].reshape(-1)))
    alpha = 0.5*np.ones(500)
    criterion = nn.MSELoss()

    for i in range(nb):
        W = ds4[i][1]
        V = np.matmul(W, H)
        if noise:
            N = np.random.rand(V.shape[0], V.shape[1])
            N = 10.**(-5)*np.linalg.norm(V)/np.linalg.norm(N)*N
            V = V + N

    path='C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/AforHWdirich,lossW/1163_.pth' #cat not normalized
    #path='C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/732_.pth' #dirichlet
    path='C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/1734_.pth' #cat normalized
    H_est, W_est = test_LMU2(V,path)
    
    H_est = 10**(-2)*np.ones((4, V.shape[1]))
    loss0=[]
    loss00=[]
    '''
    for i in range(10000):
        print(i)
        H_est, W_est = MU(V,W_est,H_est)
        print("0 :",torch.numel(torch.tensor(V))*criterion(torch.tensor(V).float(),torch.tensor(W_est).float()@torch.tensor(H_est).float())/ torch.norm(torch.tensor(V).float())**2)
        
        print("1 :",torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(H_est).float())/ torch.norm(torch.tensor(H).float())**2+torch.numel(torch.tensor(W))*criterion(torch.tensor(W).float(), torch.tensor(W_est).float()) / torch.norm(torch.tensor(W).float())**2)
        W_est,a=correctPerm(W,W_est)
        loss0.append(torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(H_est).float())/ torch.norm(torch.tensor(H).float())**2+torch.numel(torch.tensor(W))*criterion(torch.tensor(W).float(), torch.tensor(W_est).float()) / torch.norm(torch.tensor(W).float())**2)
        Q=np.zeros(np.shape(H))
        Q[0]=H[2]
        Q[1]=H[1]
        Q[2]=H[0]
        Q[3]=H[3]
        c=torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(Q).float())/ torch.norm(torch.tensor(H).float())**2+torch.numel(torch.tensor(W))*criterion(torch.tensor(W).float(), torch.tensor(W_est).float()) / torch.norm(torch.tensor(W).float())**2
        print("2 :",c)
        loss00.append(c)
        #print("2 :",torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(H_est).float())/ torch.norm(torch.tensor(H).float())**2+torch.numel(torch.tensor(W))*criterion(torch.tensor(W).float(), torch.tensor(W1).float()) / torch.norm(torch.tensor(W).float())**2)
    '''

        
    #print(criterion(torch.tensor(V).float(),torch.tensor(W_est).float()@torch.tensor(H_est).float())/torch.norm(torch.tensor(V).float())**2)

   
    #H_est,a=correctPerm(H,H_est)

    '''
    model=torch.load('C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/T1.pth')
    H_est=model['H_est']
    W_est=model['W_est']
    W_est,a=correctPerm(W,W_est)
    Q=np.zeros(np.shape(H))
    Q[0]=H_est[2]
    Q[1]=H_est[1]
    Q[2]=H_est[0]
    Q[3]=H_est[3]
    '''
    
    #path='C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/2000_.pth' #A LMU1
    #path='C:/Users/abdel/Desktop/stage/Donnees_Abdelkhalak/694_.pth' #A bad init LMU1
    
    #H_est,W_est=test_LMU1(V,W_est,path)
    
    print(torch.numel(torch.tensor(V))*criterion(torch.tensor(V).float(),torch.tensor(W).float()@torch.tensor(H).float())/ torch.norm(torch.tensor(V).float())**2)
    print(torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(H_est).float())/ torch.norm(torch.tensor(H).float())**2)
    print(torch.norm(torch.tensor(W).float()- torch.tensor(W_est).float())**2 / torch.norm(torch.tensor(W).float())**2)
    loss.append(torch.numel(torch.tensor(H))*criterion(torch.tensor(H).float(), torch.tensor(H_est).float())/ torch.norm(torch.tensor(H).float())**2+torch.numel(torch.tensor(W))*criterion(torch.tensor(W).float(), torch.tensor(W_est).float()) / torch.norm(torch.tensor(W).float())**2)
    print(loss)
        
    '''
    loss=model['Wperm']
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(np.arange(200,len(loss)),loss[200:])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    '''

    plt.plot(W.T[0],label='ground truth')
    plt.plot(W_est.T[0],label='predicted')
    plt.xlabel('spectrum size')
    plt.ylabel('data values')
    plt.legend()
    plt.show()
    plt.plot(W.T[1],label='ground truth')
    plt.plot(W_est.T[1],label='predicted')
    
    plt.xlabel('spectrum size')
    plt.ylabel('data values')
    plt.legend()
    plt.show()
    plt.plot(W.T[2],label='ground truth')
    plt.plot(W_est.T[2],label='predicted')
    plt.xlabel('spectrum size')
    plt.ylabel('data values')
    plt.legend()
    plt.show()
    plt.plot(W.T[3],label='ground truth')
    plt.plot(W_est.T[3],label='predicted')
    plt.xlabel('spectrum size')
    plt.ylabel('data values')
    plt.legend()
    plt.show()
    
    
    plt.imshow(np.reshape(H[0],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H[0]));
    plt.show()
    
    plt.imshow(np.reshape(H_est[0],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H_est[0]));
    plt.show()
    
    plt.imshow(np.reshape(H[1],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H[1]));
    plt.show()
    
    plt.imshow(np.reshape(H_est[1],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H_est[1]));
    plt.show()
    
    plt.imshow(np.reshape(H[2],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H[2]));
    plt.show()
    
    plt.imshow(np.reshape(H_est[2],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H_est[2]));
    plt.show()
    
    plt.imshow(np.reshape(H[3],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H[3]));
    plt.show()
    
    plt.imshow(np.reshape(H_est[3],(346,346)),cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.clim(0, np.max(H_est[3]));
    plt.show()
