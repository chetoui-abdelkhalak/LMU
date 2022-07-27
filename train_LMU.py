import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR

from LMU import LMU
from LMU_2nd import LMU_2nd

from utils.create_mixtures import My_dataset
from utils.data_cat import dataset_cat
from utils.data_sim_dirichlet import dataset_dirichlet
from utils.dataset_therm import dataset_therm
from tqdm import tqdm
import pickle
import math 
from utils.create_mixtures import My_dataset
from utils.data_sim_dirichlet import dataset_dirichlet
import sys

def train(train_loader,val_loader,num_epochs=6,nb_layers=6,nb_c=500,nb_l=65):
    print(nb_c,nb_l,math.floor(np.sqrt(nb_c))*nb_l*4)
    sys.stdout.flush()
    criterion = nn.MSELoss()
    model=LMU_2nd(nb_l,nb_c,nb_layers)

    total_params = [p.numel() for p in model.parameters()]
    print(total_params)
    print('2nd model')
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001, betas=(0.9, 0.999))
    #optimizer=torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #scheduler = StepLR(optimizer, step_size=500, gamma=0.1) 
    train_total_loss = []
    val_total_loss = []

    
    for epoch in range(num_epochs):
        
        loop=tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        train_total = 0
        model.train()
        for i, (V,W,H) in loop:

            A_init=10**(-2)*torch.ones(V.shape[0],nb_l,4)
            S_init=10**(-2)*torch.ones(V.shape[0],4,nb_c)
            optimizer.zero_grad()
            H_est, W_est = model(V.float(),A_init,S_init)
            train_loss =torch.numel(W)*criterion(W.float(), W_est.float())/ torch.norm(W.float())**2+ torch.numel(H)*criterion(H.float(), H_est.float()) / torch.norm(H.float())**2

            
            train_loss.backward()

            
            optimizer.step()
            print(train_loss.item())
            train_total += train_loss.item()

        train_total /= i+1
        train_total_loss.append(train_total)
        print('W_est :', W_est[0])
        print('H_est :', H_est[0])
        WW_est=W_est
        HH_est=H_est

        
        model.eval()
        with torch.no_grad():
            loop=tqdm(enumerate(val_loader),total=len(val_loader),leave=False)
            val_total = 0
            for i, (V,W,H) in loop:

                A_init=10**(-2)*torch.ones(V.shape[0],nb_l,4)
                S_init=10**(-2)*torch.ones(V.shape[0],4,nb_c)
                H_est, W_est = model(V.float(),A_init,S_init)
                val_loss = torch.numel(W)*criterion(W.float(), W_est.float())/ torch.norm(W.float())**2+ torch.numel(H)*criterion(H.float(), H_est.float()) / torch.norm(H.float())**2

                print(val_loss.item())
                val_total += val_loss.item()

            val_total /= i+1
            val_total_loss.append(train_total)    
        
        #scheduler.step()
        #loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        if epoch % 5 == 0: 
            print()
        
        print("epoch:{} | training loss:{} | validation loss:{} ".format(epoch, train_total,val_total))
        #torch.save(model, 'LMU_vf_S_01+A44+step=100'+str(epoch)+'_.pth')    
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'training loss': train_total,
            'validation loss': val_total,
            'W' : WW_est,
            'H' : HH_est,
            }, '/tsi/clusterhome/achitou/Donnees_Abdelkhalak/ANN/'+str(epoch)+'_.pth')
    
    return train_total_loss, val_total_loss,model

    


    
    


    