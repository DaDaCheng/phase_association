
import torch

import torch.nn as nn

from collections import OrderedDict

import numpy as np
    
import torch.nn.functional as F


#constant wave speed assignment model
class AssignmentModel(nn.Module):
    def __init__(self, config, station_df, seed=0, device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        n_event=config['n_event_max']
        self.event_locs=nn.Parameter(torch.randn(n_event,3))
        self.event_locs.requires_grad_(True)
        #self.event_times=nn.Parameter(torch.rand(n_event)*2-1)
        self.event_times=nn.Parameter(torch.logit(torch.linspace(0.05, 0.95, steps=n_event)))
        self.event_times.requires_grad_(True)
        self.config=config
        self.station_df=station_df
        if config['P_phase']:
            self.velp=config['vel']['P']
        if config['S_phase']:
            self.vels=config['vel']['S']
        station_locs=[]
        for idx, station in station_df.iterrows():
            station_locs.append([station['x(km)'],station['y(km)'],station['z(km)']])
        self.station_locs=torch.tensor(station_locs).to(device)
        self.beta_time=config['beta_time']
        self.beta_space=config['beta_space']
       
    def loc_and_time(self):
        event_locs=torch.sigmoid(self.beta_space*self.event_locs)
        event_times=torch.sigmoid(self.beta_time*self.event_times)
        #print(event_times)
        event_locsx=(event_locs[:,0])*(self.config['x(km)'][1]-self.config['x(km)'][0])+self.config['x(km)'][0]
        event_locsy=(event_locs[:,1])*(self.config['y(km)'][1]-self.config['y(km)'][0])+self.config['y(km)'][0]
        event_locsz=(event_locs[:,2])*(self.config['z(km)'][1]-self.config['z(km)'][0])+self.config['z(km)'][0]
        event_locs=torch.vstack([event_locsx,event_locsy,event_locsz]).T
        event_times=event_times*(self.config['t(s)'][1]-self.config['t(s)'][0])+self.config['t(s)'][0]
        return event_locs, event_times

    def arrival_time(self,station_index,phase='P'):
        event_locs, event_times=self.loc_and_time()
        station_loc=self.station_locs[station_index]
        dist=(((event_locs-station_loc)**2).sum(dim=1))**0.5
        if phase=='P':
            return dist/self.velp+event_times
        if phase=='S':
            return dist/self.vels+event_times

    def forward(self,station_index,phase='P'):
        time =self.arrival_time(station_index,phase=phase)
        
        time,sort_index=torch.sort(time)
            
        return time,sort_index
        
        
    def reset_training(self):
        self.event_locs.requires_grad_(True)
        self.event_times.requires_grad_(True)
    def load_events(self,event_locs,event_times):
        self.event_locs.data=event_locs.float()
        self.event_times.data=event_times.float()


#neural field wave speed assignment model
class AssignmentModel_NF(nn.Module):
    def __init__(self, config, station_df, seed=0,device='cpu',model_traveltime=None):
        super().__init__()
        torch.manual_seed(seed)
        n_event=config['n_event_max']
        L=config['wave_speed_model_hidden_dim']
        self.L=L
        self.event_locs=nn.Parameter(torch.randn(n_event,3).float())
        self.event_locs.requires_grad_(True)
        self.event_times=nn.Parameter(torch.logit(torch.linspace(0.05, 0.95, steps=n_event)).float())
        self.event_times.requires_grad_(True)
        self.config=config
        self.station_df=station_df
        
        model_traveltime.eval()
        self.event_times=nn.Parameter(torch.logit(torch.linspace(0.05, 0.95, steps=n_event)).float())
        


        self.model_traveltime_p=nn.Parameter(torch.randn(L))
        self.model_traveltime_p.requires_grad_(True)
        self.model_traveltime=model_traveltime.to(device)
        for  p in self.model_traveltime.parameters():
            p.requires_grad_(False)
        
        
        if config['P_phase']:
            self.velp=config['vel']['P']
        if config['S_phase']:
            self.vels=config['vel']['S']
        station_locs=[]
        for idx, station in station_df.iterrows():
            station_locs.append([station['x(km)'],station['y(km)'],station['z(km)']])
        self.n_earthquake=n_event
        self.station_locs=torch.tensor(station_locs).to(device)
        self.beta_time=config['beta_time']
        self.beta_space=config['beta_space']
        self.beta_z=config['beta_z']
       
    def loc_and_time(self):
        
        event_locs=torch.sigmoid(self.beta_space*self.event_locs)
        
        event_times=torch.sigmoid(self.beta_time*self.event_times)
        #print(event_times)
        event_locsx=(event_locs[:,0])*(self.config['x(km)'][1]-self.config['x(km)'][0])+self.config['x(km)'][0]
        event_locsy=(event_locs[:,1])*(self.config['y(km)'][1]-self.config['y(km)'][0])+self.config['y(km)'][0]
        event_locsz=(event_locs[:,2])*(self.config['z(km)'][1]-self.config['z(km)'][0])+self.config['z(km)'][0]
        event_locs=torch.vstack([event_locsx,event_locsy,event_locsz]).T
        event_times=event_times*(self.config['t(s)'][1]-self.config['t(s)'][0])+self.config['t(s)'][0]
        return event_locs, event_times

    def arrival_time(self,station_index,phase='P'):
        event_locs=torch.sigmoid(self.beta_space*self.event_locs)
        event_times=torch.sigmoid(self.beta_time*self.event_times)
        event_times=event_times*(self.config['t(s)'][1]-self.config['t(s)'][0])+self.config['t(s)'][0]
        z=torch.sigmoid(self.beta_z*self.model_traveltime_p)
        z=z.reshape(-1,self.L).repeat(self.n_earthquake,1)
        Xin=torch.cat([z,event_locs],dim=1)
        temp,_=self.model_traveltime(Xin)
        if phase=='P':
            return temp[:,station_index]+event_times
        if phase=='S':
            return temp[:,station_index]/self.velp*self.vels+event_times

        
        
        

    def forward(self,station_index,phase='P'):
        time =self.arrival_time(station_index,phase=phase)
        time,sort_index=torch.sort(time)
        return time,sort_index
        
        
    def reset_training(self):
        self.event_locs.requires_grad_(True)
        self.event_times.requires_grad_(True)
        self.model_traveltime_p.requires_grad_(True)
        
    def load_events(self,event_locs,event_times):
        self.event_locs.data=event_locs.float()
        self.event_times.data=event_times.float()
        
    def load_model_traveltime_p(self, model_traveltime_p):
        self.model_traveltime_p.data = model_traveltime_p
    

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
    
    
     




class Encoder(nn.Module):
    def __init__(self,capacity=8,latent_dims=2):
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
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.bn4(self.fc(x))
        return x

class Decoder(nn.Module):
    def __init__(self, capacity=8, latent_dims=4):  # latent_dims=4 as per your clarification
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=4*c*4*4*4)  # latent vector mapped to 4x4x4 feature map
        self.bn3 = nn.BatchNorm3d(4*c)
        self.conv3 = nn.ConvTranspose3d(in_channels=4*c, out_channels=2*c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(2*c)
        self.conv2 = nn.ConvTranspose3d(in_channels=2*c, out_channels=c, kernel_size=4, stride=2, padding=2)
        self.bn1 = nn.BatchNorm3d(c)
        self.conv1 = nn.ConvTranspose3d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=2)  # Adjusted for 26x26x26 output
        self.lkrelu = nn.LeakyReLU(negative_slope=0.1)
        self.L = latent_dims

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.bn3(x.view(x.size(0), -1, 4, 4, 4)))  # Reshape and apply BatchNorm3d
        x = F.relu(self.bn2(self.conv3(x)))  # First upsampling
        x = F.relu(self.bn1(self.conv2(x)))  # Second upsampling
        x = self.conv1(x)  # Final layer to reach the target size
        return x
    
class Autoencoder(nn.Module):
    def __init__(self,capacity=64,latent_dims=1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(capacity=capacity,latent_dims=latent_dims)
        self.decoder = Decoder(capacity=capacity,latent_dims=latent_dims)
        # p is the number of repeat for the latent code.
    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x,latent
    
    


