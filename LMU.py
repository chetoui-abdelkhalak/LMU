import torch
import torch.nn as nn
import numpy as np
import sys


class single_layer(nn.Module):
    def __init__(self,nb_l,nb_c,i):
        super(single_layer,self).__init__()
        self.nb_l=nb_l
        self.nb_c=nb_c
        
        self.A=torch.tensor([[1,1e-4,1e-4,1e-4],[1e-4,1,1e-4,1e-4],[1e-4,1e-4,1,1e-4],[1e-4,1e-4,1e-4,1]]).float().reshape((1, 4, 4))

        self.act=nn.LeakyReLU(-0.1)
        self.A=nn.Parameter(self.A, requires_grad = True)

    def forward(self,V,W,H):

        Gp=torch.bmm(torch.transpose(W,1,2),torch.bmm((W),H))
        Gn=torch.bmm(torch.transpose(W,1,2),V)

        B=self.act(self.A)

            
        H=H*(B@(Gn/Gp))

        
        Gp_w=torch.bmm(W,torch.bmm(H,torch.transpose(H,1,2)))
        Gn_w=torch.bmm(V,torch.transpose(H,1,2))

        
        W=W*((Gn_w/Gp_w))
  



        return H,W
        

class LMU(nn.Module) :
    def __init__(self,nb_l,nb_c,nb_layers):
        super(LMU,self).__init__()
                
        self.nb_layers=nb_layers
        self.deep_nmfs = nn.ModuleList(
            [single_layer( nb_l, nb_c,i) for i in range(self.nb_layers)]
        )
    
    def forward(self,V, W, H):
        
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            H,W = l(V,W,H)
            #print(H)
        return H,W

            
