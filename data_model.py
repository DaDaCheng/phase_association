
import torch
import numpy as np
from tqdm import tqdm
import skfmm

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.sparse import random
from numpy.random import default_rng


import multiprocessing


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, device=None):
        self.dim = dim
        self.device = device
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
        k_max = size//2
        if dim == 1:
            k = torch.arange(start=-k_max, end=k_max, step=1, device=device)

            self.sqrt_eig = size*np.sqrt(2.0)*sigma*((4*(np.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.arange(start=-k_max, end=k_max, step=1, device=device).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers
            self.sqrt_eig = (size**2)*np.sqrt(2.0)*sigma*((4*(np.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.
        elif dim == 3:
            wavenumers = torch.arange(start=-k_max, end=k_max, step=1, device=device).repeat(size,size,1)
            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)
            self.sqrt_eig = (size**3)*np.sqrt(2.0)*sigma*((4*(np.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self, N):
        np.random.seed(1)
        coeff = torch.randn(N, *self.size, device=self.device) + 1j*torch.randn(N, *self.size, device=self.device)
        if self.dim==2:
            coeff = torch.fft.fftshift(self.sqrt_eig*coeff,dim=(1,2))
            u = torch.fft.ifft2(coeff, dim=(1,2), norm="backward")
        elif self.dim==3:
            coeff = torch.fft.fftshift(self.sqrt_eig*coeff, dim=(1,2,3))
            u = torch.fft.ifft2(coeff,  norm="backward", dim=(1,2,3))
        return u
    
def get_background(grid_size,vmin,vmax):    
    V = np.empty((grid_size, grid_size, grid_size))
    b = (vmax-vmin)
    z = np.linspace(0,1,grid_size)
    for n in range(z.size):
        z = np.linspace(0,1,grid_size)
        V[:, :, n] = vmin + b*z[n]
    return  V


def add_gaussian_pertubation(x,y,z,sx,sy,sz,A,grid_size):
    Y,X,Z = np.meshgrid(np.linspace(0,1,grid_size), np.linspace(0,1,grid_size),np.linspace(0,1,grid_size))
    temp=(X-x)**2/sx**2+(Y-y)**2/sy**2+(Z-z)**2/sz**2
    temp=np.exp(-temp/2)
    return A*temp/temp.max()
    


def get_traveling_time_multi(V_list,src,grid_size):
    x = np.linspace(0,1,grid_size)
    dx = 1/(grid_size-1)
    Y, X, Z = np.meshgrid(x,x,x)
    loc_src_x=src[:,0]
    loc_src_y=src[:,1]
    loc_src_z=src[:,2]
    n_station=len(loc_src_x)
    n_wavespeed=V_list.shape[0]
    travel_time = np.zeros((n_wavespeed,n_station,grid_size,grid_size,grid_size))
    for j in tqdm(range(n_wavespeed)):
        V=V_list[j]
        for i in range(n_station):
            phi = ((X-loc_src_x[i])**2+(Y-loc_src_y[i])**2+(Z-loc_src_z[i])**2)
            phi=phi-np.min(phi)
            try:
                tt= skfmm.travel_time(phi,V,order=2,dx=dx)
            except:
                tt = skfmm.travel_time(phi,V,order=1,dx=dx)            
            travel_time[j,i,:,:,:]=tt
    return travel_time



def get_traveling_time(V,src):
    grid_size=V.shape[0]
    x = np.linspace(0,1,grid_size)
    dx = 1/(grid_size-1)
    Y, X, Z = np.meshgrid(x,x,x)
    loc_src_x=src[:,0]
    loc_src_y=src[:,1]
    loc_src_z=src[:,2]
    n_station=len(loc_src_x)
    travel_time = np.zeros((n_station,grid_size,grid_size,grid_size))
    for i in range(n_station):
        phi = ((X-loc_src_x[i])**2+(Y-loc_src_y[i])**2+(Z-loc_src_z[i])**2)
        phi=phi-np.min(phi)
        try:
            tt= skfmm.travel_time(phi,V,order=2,dx=dx)
        except:
            tt = skfmm.travel_time(phi,V,order=1,dx=dx)            
        travel_time[i,:,:,:]=tt
    return travel_time



class WaveSpeedDataset(Dataset):
    def __init__(self,n_wavespeed,n_wavespeed_test, grid_size,n_mode,density,vmin,vmax):
        
        self.data_wavespeed=np.zeros((n_wavespeed,grid_size,grid_size,grid_size),dtype=np.float32)
        self.data_wavespeed_test=np.zeros((n_wavespeed_test,grid_size,grid_size,grid_size),dtype=np.float32)
        self.n_wavespeed=n_wavespeed
        self.n_wavespeed_test=n_wavespeed_test
        
        V=get_background(grid_size,vmin,vmax)
        GRF = GaussianRF(dim=3, size=grid_size, alpha=2.5, tau=2.5, sigma=None, device=None)
        c = torch.abs(GRF.sample(n_mode))
        self.c=c
        rng = default_rng()
        for i in range(n_mode):
            c[i]=(c[i]-c[i].min())/(c[i].max()-c[i].min())*1.5
        z=np.zeros((n_wavespeed+n_wavespeed_test,n_mode))
        for i in range(n_wavespeed+n_wavespeed_test):
            S = random(n_mode,1, density=density, random_state=rng)
            z[i]=S.A.reshape(-1)
        z[z!=0]=(z[z!=0]-0.5)*2
        wavespeeds=np.einsum('ij,jabc->iabc',z,self.c).squeeze()+np.broadcast_to(V,[(n_wavespeed+n_wavespeed_test),grid_size,grid_size,grid_size])
        wavespeeds[wavespeeds<vmin]=vmin 
        wavespeeds[wavespeeds>vmax]=vmax
        wavespeeds=wavespeeds.astype(np.float32)
        self.data_wavespeed=wavespeeds[:n_wavespeed,:,:,:]
        self.data_wavespeed_test=wavespeeds[n_wavespeed:,:,:,:]
    def __len__(self):
        return self.n_wavespeed
    def __getitem__(self, idx):
        image = {'image':self.data_wavespeed[idx]}
        return image
    



class get_latent(object):
    def __init__(self, WaveSpeedData,AutoEn,device):
        AutoEn.eval()
        with torch.no_grad():
            data=torch.tensor(WaveSpeedData.data_wavespeed).unsqueeze(1).float().to(device)
            _,emb=AutoEn(data)
            self.z_matrix=emb.clone().detach().cpu().numpy()
            data=torch.tensor(WaveSpeedData.data_wavespeed_test).unsqueeze(1).float().to(device)
            _,emb=AutoEn(data)
            self.z_matrix_test=emb.clone().detach().cpu().numpy()
    def scale_z(self,z):
        rez=((z-np.min(self.z_matrix,axis=0))/(np.max(self.z_matrix,axis=0)-np.min(self.z_matrix,axis=0))).copy()
        return rez
    def scale_z_inv(self,z):
        rez= (z*(np.max(self.z_matrix,axis=0)-np.min(self.z_matrix,axis=0))+np.min(self.z_matrix,axis=0)).copy()
        return rez    
        
def compute_task(inputs):
    V_,loc_src_coord,n_station=inputs
    result = get_traveling_time(V_, loc_src_coord).reshape(n_station,-1).astype(dtype=np.float16)
    return result
    


class TravelTimeDataset(Dataset):
    def __init__(self,WaveSpeedData,z_matrix_scale,z_matrix_test_scale,n_station,grid_size,loc_station_index=None):
        self.z_matrix_scale=z_matrix_scale
        self.z_matrix_test_scale=z_matrix_test_scale
        self.n_wavespeed=WaveSpeedData.n_wavespeed
        self.n_wavespeed_test=WaveSpeedData.n_wavespeed_test
        self.data_wavespeed=np.array(WaveSpeedData.data_wavespeed)
        self.data_wavespeed_test=np.array(WaveSpeedData.data_wavespeed_test)
        self.n_gt=grid_size**3
        self.n_station=n_station
        ##station
        if loc_station_index==None:
            loc_station_index=np.random.randint(grid_size,size=[n_station,3])
            loc_station_index[:,2]=0 #only on the surface
        self.loc_station_index=loc_station_index
        self.loc_station_coord = loc_station_index/(grid_size-1)
        
        ##mesh
        x = np.linspace(0,1,grid_size)
        y = np.linspace(0,1,grid_size)
        z = np.linspace(0,1,grid_size)
        Y, X, Z = np.meshgrid(x,y,z)
        self.set_coord=np.vstack([X.reshape(-1),Y.reshape(-1),Z.reshape(-1)]).T
        self.data=np.zeros((self.n_wavespeed,n_station,grid_size**3),dtype=np.float16)
        self.data_test=np.zeros((self.n_wavespeed_test,n_station,grid_size**3),dtype=np.float16)
        data = [(self.data_wavespeed[i], self.loc_station_coord,n_station) for i in range(self.n_wavespeed)]
        pool = multiprocessing.Pool()
        result_iter = pool.imap(compute_task, data)
        result_list = []
        for result in tqdm(result_iter, total=len(data)):
            result_list.append(result)
        # Close the pool
        pool.close()
        # Join the pool
        pool.join()
        self.data=np.array(result_list).astype(np.float16)
        del result_list
        data = [(self.data_wavespeed_test[i], self.loc_station_coord,n_station) for i in range(self.n_wavespeed_test)]
        pool = multiprocessing.Pool()
        result_iter = pool.imap(compute_task, data)
        result_list = []
        for result in tqdm(result_iter, total=len(data)):
            result_list.append(result)
        # Close the pool
        pool.close()
        # Join the pool
        pool.join()
        self.data_test=np.array(result_list).astype(np.float16)
        del result_list
    def __len__(self):
        return self.n_wavespeed*self.n_gt
    def __getitem__(self, idx):
        idx_wavespeed,idx= idx//self.n_gt,idx%self.n_gt
        input=np.concatenate([self.z_matrix_scale[idx_wavespeed], self.set_coord[idx,:]])
        output=self.data[idx_wavespeed,:,idx]
        image = {'in':input,'out':output}
        return image
    def output_idx(self, i):
        V_=self.data_wavespeed[i]
        output=get_traveling_time(V_,self.loc_station_coord).reshape(self.n_station,-1)
        image = {'z':self.z_matrix_scale[i],'out':output}
        return image
    def output_test(self, i):
        V_=self.data_wavespeed_test[i]
        output=get_traveling_time(V_,self.loc_station_coord).reshape(self.n_station,-1)
        image = {'z':self.z_matrix_test_scale[i],'out':output}
        return image
    def output_V(self, V):
        output=get_traveling_time(V,self.loc_station_coord).reshape(self.n_station,-1)
        image = {'out':output}
        return image
    
    
class RepeatDataset(Dataset):
    def __init__(self, dataset, repeat):
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
    

class TravelTimeDatasetFixedWavespeed(Dataset):
    def __init__(self, V,n_station,grid_size,loc_station_index=None):
        if loc_station_index is None:
            loc_station_index=np.random.randint(grid_size,size=[n_station,3])
            loc_station_index[:,2]=0
        self.loc_station_index=loc_station_index
        self.loc_station_coord = self.loc_station_index/(grid_size-1)
        self.travel_time=get_traveling_time(V,self.loc_station_coord).reshape(n_station,-1)
        x = np.linspace(0,1,grid_size)
        y = np.linspace(0,1,grid_size)
        z = np.linspace(0,1,grid_size)
        Y, X, Z = np.meshgrid(x,y,z)
        self.set_coord=np.vstack([X.reshape(-1),Y.reshape(-1),Z.reshape(-1)]).T
        self.datalen=self.set_coord.shape[0]
    def __len__(self):
        return self.datalen
    def __getitem__(self, idx):
        sample = {'in':self.set_coord[idx,:], 'out': self.travel_time[:,idx]}
        return sample