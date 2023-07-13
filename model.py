import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import time


### SIREN is edited from paper Implicit Neural Representations with Periodic Activation Functions, https://github.com/vsitzmann/siren
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim,  output_dim,hiddenLayers = 1,batchnorm=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fcHidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(hiddenLayers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.batchnorm=batchnorm
        if self.batchnorm:
            self.bn=nn.BatchNorm1d(input_dim)
    def forward(self, x):
        if self.batchnorm:
            x = self.bn(x)
        x = torch.relu(self.fc1(x))
        for i in range(len(self.fcHidden)):
            x = torch.relu(self.fcHidden[i](x))
        x = self.fc2(x)
        return x





class Encoder(nn.Module):
    def __init__(self,capacity=8,latent_dims=2,vmin=0.5,vmax=2.5):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=c, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(c)# 
        self.conv2 = nn.Conv3d(in_channels=c, out_channels=c*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(2*c)
        self.conv3 = nn.Conv3d(in_channels=c*2, out_channels=c*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(4*c)
        self.fc = nn.Linear(in_features=4*c*4*4*4, out_features=latent_dims)
        self.pool=nn.AvgPool3d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm1d(latent_dims)
        self.vmin=vmin
        self.vmax=vmax
            
    def forward(self, x):
        x=(x-self.vmin)/(self.vmax-self.vmin)
        skip1=self.pool(self.pool(self.pool(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.bn4(self.fc(x))
        return x

class Decoder(nn.Module):
    def __init__(self,capacity=8,latent_dims=2,vmin=0.5,vmax=2.5,p=1):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims*p, out_features=4*c*4*4*4)
        self.bn3 = nn.BatchNorm3d(4*c)
        self.conv3 = nn.ConvTranspose3d(in_channels=4*c, out_channels=2*c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(2*c)
        self.conv2 = nn.ConvTranspose3d(in_channels=2*c, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(c)
        self.conv1 = nn.ConvTranspose3d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.vmin=vmin
        self.vmax=vmax
        self.lkrelu=nn.LeakyReLU(negative_slope=0.1)
        self.L=latent_dims
        self.p=p
    def forward(self, x):
        x=x.repeat(1,self.p)
        for i in range(self.p):
            x[:,self.L*i:self.L*(i+1)]=x[:,self.L*i:self.L*(i+1)]*(i+1)
        x=torch.sin(x)
        x = self.fc(x)
        x = F.relu(self.bn3(x.view(x.size(0), -1, 4, 4,4)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.conv1(x)
        x=torch.sigmoid(x)
        x =x*(self.vmin+self.vmax)
        x[x<self.vmin]=self.vmin
        x[x>self.vmax]=self.vmax
        return x

    
class Autoencoder(nn.Module):
    def __init__(self,capacity=64,latent_dims=1,vmin=0.5,vmax=2.5,p=1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(capacity=capacity,latent_dims=latent_dims,vmin=vmin,vmax=vmax)
        self.decoder = Decoder(capacity=capacity,latent_dims=latent_dims,vmin=vmin,vmax=vmax,p=p)
        # p is the number of repeat for the latent code.
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x,latent
        







    
class AssignmentModel(nn.Module):
    def __init__(self,model_traveltime, n_station, n_earthquake,tau_max,L=0,cheat=False,loc_earthquake_truth=None,time_earthquake_truth=None,z_truth=None):
        super().__init__()
        self.L=L
        if cheat:
            # In the cheat mode, the parameters will be initialized around the grount truth.  This is used to check whether the SIREN and auto-encoder are well trained.
            self.loc_earthquake=nn.Parameter(torch.tensor(loc_earthquake_truth+np.random.rand(n_earthquake,3)*0.1).float())
            self.time_earthquake=nn.Parameter(torch.tensor(time_earthquake_truth+np.random.rand(n_earthquake)*0.1*tau_max).float())
            if L>0:
                self.z=nn.Parameter(torch.tensor(z_truth+np.random.rand(L)*0.1).float())
        else:
            self.loc_earthquake=nn.Parameter(torch.tensor(np.random.rand(n_earthquake,3)).float())
            self.time_earthquake=nn.Parameter(torch.tensor(np.random.rand(n_earthquake)*tau_max).float())
            if L>0:
                self.z=nn.Parameter(torch.tensor(np.random.rand(L)).float())
        self.loc_earthquake.requires_grad_(True)
        self.time_earthquake.requires_grad_(True)
        model_traveltime.eval()
        self.model_travelling=model_traveltime
        for  p in self.model_travelling.parameters():
            p.requires_grad_(False)
        self.n_station=n_station
        self.n_earthquake=n_earthquake
        self.tau_max=tau_max
    def forward(self,sort_out=True):
        self.time_earthquake.data.clamp_(0, self.tau_max)
        self.loc_earthquake.data.clamp_(0,1)
        loc_earthquake=self.loc_earthquake
        time_earthquake=self.time_earthquake
        if self.L>0:
            self.z.data.clamp_(0, 1)
            z=self.z
            z=z.repeat(self.n_earthquake,1)
            Xin=torch.cat([z,loc_earthquake],dim=1)
            temp,_=self.model_travelling(Xin)
        else:
            temp,_=self.model_travelling(loc_earthquake)
        temp=temp+time_earthquake.repeat(self.n_station,1).T
        if sort_out:
            temp,_=torch.sort(temp,dim=0)
        return temp