import numpy as np
import pickle 
from MU_RN import MU,mu_update
import matplotlib.pyplot as plt
from utils.create_mixtures import My_dataset
from utils.data_sim_dirichlet import dataset_dirichlet
from utils.data_sim_dirichlet import dataset_dirichlet
from utils.data_cat import dataset_cat
from utils.cub_cat_red import cub_cat_red
from utils.dataset_therm import dataset_therm
from utils.dataset_cub_cat import cub_cat
from utils.split_A import split_A, train_val_dataset
from torch.utils.data import DataLoader
#from MU_rec import MU
from utils.starlet import Starlet_Forward, Starlet_Inverse
from utils.test_H import test_H

from train2 import train2
from train_LMU import train
import sys
from utils.apply_LMU import apply_LMU_realistic


if __name__ == '__main__': 

    
    with open('./Donnees_Abdelkhalak/data/dataset_cub_cat.pkl', 'rb') as f:
    			dataset = pickle.load(f)
    with open('./Donnees_Abdelkhalak/data/dataset_4_sources.pickle', 'rb') as f:
    			ds4 = pickle.load(f)
    with open('./Donnees_Abdelkhalak/data/sources.pkl', 'rb') as f:
    			spectres = pickle.load(f)	     
    
    #datasets = split_A(dataset) 
    datasets= train_val_dataset(dataset,val_split=0.2)
    train_set = datasets['train']
    val_set = datasets['val']
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set,batch_size=10 , shuffle=True, num_workers=4)
    
    train(train_loader,val_loader,num_epochs=2000,nb_layers=25,nb_c=dataset[0][0].shape[1],nb_l=dataset[0][0].shape[0])

    
