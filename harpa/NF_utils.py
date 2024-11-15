from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import skfmm
from obspy import UTCDateTime as UT

import pandas as pd
import multiprocessing
    
class WaveSpeedDataset(Dataset):
    def __init__(self, V_trains, p_train):
        
        self.data_wavespeed=np.array(V_trains,dtype=np.float32)
        
        self.n_wavespeed=len(V_trains)
        self.p_train=np.array(p_train)
    
    def __len__(self):
        return self.n_wavespeed
    def __getitem__(self, idx):
        image = {'image':self.data_wavespeed[idx],'p': self.p_train}
        return image
    
def get_traveling_time_multi(V_list,station_df,config,dx=25,align_station=False):
    station_df=station_df.copy()
    grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
    grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
    grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
    xx = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
    yy = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
    zz = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
    X, Y, Z = np.meshgrid(xx,yy,zz)
    X=X.transpose(1,0,2)
    Y=Y.transpose(1,0,2)
    Z=Z.transpose(1,0,2)
    if align_station:
        station_df['x(km)'] = station_df['x(km)'].apply(lambda x: xx[np.argmin(np.abs(np.array(xx) - x))])
        station_df['y(km)'] = station_df['y(km)'].apply(lambda x: yy[np.argmin(np.abs(np.array(yy) - x))])
        station_df['z(km)'] = station_df['z(km)'].apply(lambda x: zz[np.argmin(np.abs(np.array(zz) - x))])
    
    station_locs=np.array([station_df['x(km)'],station_df['y(km)'],station_df['z(km)']]).T
    loc_src_x=station_locs[:,0]
    loc_src_y=station_locs[:,1]
    loc_src_z=station_locs[:,2]
    n_station=len(loc_src_x)
    n_wavespeed=len(V_list)
    travel_time = np.zeros((n_wavespeed,n_station,grid_size_x,grid_size_y,grid_size_z))
    for j in tqdm(range(n_wavespeed)):
        V=V_list[j]
        #V=np.ones_like(X)
        for i in range(n_station):
            phi = ((X-loc_src_x[i])**2+(Y-loc_src_y[i])**2+(Z-loc_src_z[i])**2)
            #print(np.min(phi))
            #phi=phi-np.min(phi)
            phi=phi-0.001
            try:
                tt= skfmm.travel_time(phi,V,order=2,dx=dx)
            except:
                tt = skfmm.travel_time(phi,V,order=1,dx=dx)            
            travel_time[j,i,:,:,:]=tt
    return travel_time


def compute_task(inputs):
    phi,V,dx=inputs
    result = skfmm.travel_time(phi,V,order=2,dx=dx).astype(dtype=np.float16)
    return result

def get_traveling_time_multi_mp(V_list,station_df,config,dx=25,align_station=False):
    grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
    grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
    grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
    xx = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
    yy = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
    zz = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
    X, Y, Z = np.meshgrid(xx,yy,zz)
    X=X.transpose(1,0,2)
    Y=Y.transpose(1,0,2)
    Z=Z.transpose(1,0,2)
    
    if align_station:
        station_df['x(km)'] = station_df['x(km)'].apply(lambda x: xx[np.argmin(np.abs(np.array(xx) - x))])
        station_df['y(km)'] = station_df['y(km)'].apply(lambda x: yy[np.argmin(np.abs(np.array(yy) - x))])
        station_df['z(km)'] = station_df['z(km)'].apply(lambda x: zz[np.argmin(np.abs(np.array(zz) - x))])
    station_locs=np.array([station_df['x(km)'],station_df['y(km)'],station_df['z(km)']]).T
    loc_src_x=station_locs[:,0]
    loc_src_y=station_locs[:,1]
    loc_src_z=station_locs[:,2]
    n_station=len(loc_src_x)
    n_wavespeed=len(V_list)
    data_list=[]
    for j in tqdm(range(n_wavespeed)):
        V=V_list[j]
        #V=np.ones_like(X)
        for i in range(n_station):
            phi = ((X-loc_src_x[i])**2+(Y-loc_src_y[i])**2+(Z-loc_src_z[i])**2)
            #print(np.min(phi))
            #phi=phi-np.min(phi)
            phi=phi-0.001
            # try:
            #     tt= skfmm.travel_time(phi,V,order=2,dx=dx)
            # except:
            #     tt = skfmm.travel_time(phi,V,order=1,dx=dx)            
            data_list.append((phi,V,dx))
    pool = multiprocessing.Pool()
    result_iter = pool.imap(compute_task, data_list)
    result_list = []
    for result in tqdm(result_iter, total=n_wavespeed*n_station):
            result_list.append(result)
    pool.close()
    pool.join()
    result_list=np.array(result_list).astype(np.float16).reshape(n_wavespeed,n_station,grid_size_x,grid_size_y,grid_size_z)
    return result_list

def add_gaussian_pertubation(x,y,z,sx,sy,sz,A,grid_size):
    Y,X,Z = np.meshgrid(np.linspace(0,1,grid_size), np.linspace(0,1,grid_size),np.linspace(0,1,grid_size))
    temp=(X-x)**2/sx**2+(Y-y)**2/sy**2+(Z-z)**2/sz**2
    temp=np.exp(-temp/2)
    return A*temp/temp.max()




def gen_station(n_station,config,seed=0):
    np.random.seed(seed)
    dx =config['dx(km)']
    grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
    grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
    grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
    x = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
    y = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
    z = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
    
    
    station_loc=[]
    for i in range(n_station):
        loc=[x[np.random.randint(len(x))],y[np.random.randint(len(y))],0.0]
        #loc=[x[2],y[18],0.0]
        station_loc.append(loc)        
    station_df = []
    for i in range(n_station):
        loc=station_loc[i]
        station_df.append({
            "id": 'S.'+str(i)+'.BH',
            "x(km)": loc[0],
            "y(km)": loc[1],
            "z(km)": loc[2]
        })
    station_df=pd.DataFrame(station_df)
    
    return station_df

def gen_source(n_source,frequence,config,seed=0,randomtime=False):
    np.random.seed(seed)
    source_loc=[]
    source_loc_lindex=[]
    source_time=[]
    
    dx =config['dx(km)']
    grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
    grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
    grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
    x = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
    y = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
    z = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
    
    Tmax=60/frequence*n_source
    if randomtime:
        Tlist=np.random.rand(n_source)*Tmax
    else:
        Tlist=np.linspace(0,Tmax,n_source)
    for n in range(n_source):
        #idxs=[np.random.randint(len(x)),np.random.randint(len(y)),np.random.randint(len(z))]
        idxs=[np.random.randint(1,len(x)-1),np.random.randint(1,len(y)-1),np.random.randint(11,len(z)-1)] #avoid boundary
        source_loc_lindex.append(idxs)
        source_loc.append([x[idxs[0]],y[idxs[1]],z[idxs[2]]])
        #source_time.append(np.random.rand()*60/frequence*8)
        source_time.append(Tlist[n])
        
    catalog_df=[]
    for source_index in range(n_source):
        loc=source_loc[source_index]
        catalog_df.append({
            "event_index": source_index,
            "time":  np.datetime64(int(1000*source_time[source_index]), 'ms'),
            "x(km)": loc[0],
            "y(km)": loc[1],
            "z(km)": loc[2],
        })
    catalog_df=pd.DataFrame(catalog_df)
    catalog_df['time'] = catalog_df['time'].apply(lambda x: UT(x))

            
        
    return source_loc,source_loc_lindex,source_time,catalog_df


import torch


class TravelTimeDataset(Dataset):
    def __init__(self,travel_time_list,V_paramter_list,config,station_df):
        self.n_wavespeed=len(V_paramter_list)
        self.n_station=len(station_df)
        self.dx=config['dx(km)']
        self.input_scale=100
        self.out_scale=100
        grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
        grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
        grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
        x = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
        y = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
        z = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
        X, Y, Z = np.meshgrid(x,y,z)
        X=X.transpose(1,0,2)
        Y=Y.transpose(1,0,2)
        Z=Z.transpose(1,0,2)
        
        self.X =(X-config['x(km)'][0])/(config['x(km)'][1]-config['x(km)'][0])
        self.Y =(Y-config['y(km)'][0])/(config['y(km)'][1]-config['y(km)'][0])
        self.Z =(Z-config['z(km)'][0])/(config['z(km)'][1]-config['z(km)'][0])
                
        self.grid_sizes=[grid_size_x,grid_size_y,grid_size_z]
        self.travel_time_list=travel_time_list
        self.n_gt=grid_size_x*grid_size_y*grid_size_z
        self.V_paramter_list=V_paramter_list
        ##station
        print(self.X.shape,self.Y.shape,self.Z.shape)
        print(travel_time_list.shape)
        #self.station_locs=np.array([station_df['x(km)'],station_df['y(km)'],station_df['z(km)']])
    def __len__(self):
        return self.n_wavespeed*self.n_gt
    def __getitem__(self, idx):
        idx_wavespeed,id_= idx//self.n_gt,idx%self.n_gt
        idx,id_= id_//(self.grid_sizes[1]*self.grid_sizes[2]),id_%(self.grid_sizes[1]*self.grid_sizes[2])
        idy,idz= id_//self.grid_sizes[2],id_%self.grid_sizes[2]
        input=np.concatenate([self.V_paramter_list[idx_wavespeed], [self.X[idx,idy,idz]], [self.Y[idx,idy,idz]],[self.Z[idx,idy,idz]]])
        output=self.travel_time_list[idx_wavespeed,:,idx,idy,idz]
        image = {'in':input,'out':output}
        return image 
    
    
    
    
    

class TravelTimeDataset_WS(Dataset):
    def __init__(self,travel_time_list,WaveSpeedData,p_list,model_autoencoder,device,config,beta=0):
        self.n_wavespeed=len(WaveSpeedData)
        self.dx=config['dx(km)']
        dx=self.dx
        self.beta=beta
        #ATE_train_loader = torch.utils.data.DataLoader(WaveSpeedData, batch_size=128, shuffle=False)
        model_autoencoder.eval()
        with torch.no_grad():
            data=torch.tensor(WaveSpeedData.data_wavespeed).unsqueeze(1).float().to(device)
            _,emb=model_autoencoder(data)
        p=emb.clone().detach().cpu().squeeze()
        self.z=torch.sigmoid(self.beta*p).numpy()
        self.p_list=p_list
        print('z,',self.z[:5,:])
        
        
        grid_size_x= int((config['x(km)'][1]-config['x(km)'][0])/dx+1)
        grid_size_y= int((config['y(km)'][1]-config['y(km)'][0])/dx+1)
        grid_size_z= int((config['z(km)'][1]-config['z(km)'][0])/dx+1)
        x = np.linspace(config['x(km)'][0],config['x(km)'][1],grid_size_x)
        y = np.linspace(config['y(km)'][0],config['y(km)'][1],grid_size_y)
        z = np.linspace(config['z(km)'][0],config['z(km)'][1],grid_size_z)
        X, Y, Z = np.meshgrid(x,y,z)
        X=X.transpose(1,0,2)
        Y=Y.transpose(1,0,2)
        Z=Z.transpose(1,0,2)
        
        self.X =(X-config['x(km)'][0])/(config['x(km)'][1]-config['x(km)'][0])
        self.Y =(Y-config['y(km)'][0])/(config['y(km)'][1]-config['y(km)'][0])
        self.Z =(Z-config['z(km)'][0])/(config['z(km)'][1]-config['z(km)'][0])
                
        self.grid_sizes=[grid_size_x,grid_size_y,grid_size_z]
        self.travel_time_list=travel_time_list
        self.n_gt=grid_size_x*grid_size_y*grid_size_z
        #self.V_paramter_list=V_paramter_list
        ##station
        print(self.X.shape,self.Y.shape,self.Z.shape)
        print(travel_time_list.shape)
        #self.station_locs=np.array([station_df['x(km)'],station_df['y(km)'],station_df['z(km)']])
    def __len__(self):
        return self.n_wavespeed*self.n_gt
    def __getitem__(self, idx):
        idx_wavespeed,id_= idx//self.n_gt,idx%self.n_gt
        idx,id_= id_//(self.grid_sizes[1]*self.grid_sizes[2]),id_%(self.grid_sizes[1]*self.grid_sizes[2])
        idy,idz= id_//self.grid_sizes[2],id_%self.grid_sizes[2]
        input=np.concatenate([self.z[idx_wavespeed], [self.X[idx,idy,idz]], [self.Y[idx,idy,idz]],[self.Z[idx,idy,idz]]])
        output=self.travel_time_list[idx_wavespeed,:,idx,idy,idz]
        p=self.p_list[idx_wavespeed]
        image = {'in':input,'out':output,'p':p}
        return image