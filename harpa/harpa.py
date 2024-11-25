from .utils import *
#from .plt_utils import *
from .utils import annotation
import json
from obspy import UTCDateTime as UT

import torch
from scipy.optimize import linear_sum_assignment

from torch import nn
import torch.nn.functional as F
from .sgld import SGLD
import copy
from datetime import datetime, timedelta
import random

from .model import AssignmentModel, AssignmentModel_NF

from collections import Counter

import multiprocessing as mp

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import platform
    

    

def preprocess_data(picks=None,station_df=None,config=None):

    picks=picks.copy()
    station_df=station_df.copy()
    picks,station_df=rename_df(picks,station_df)
    if len(picks)==0:
        pick_df=pd.DataFrame(columns=['station_id','timestamp','type','amp'])
    else:
        pick_df=picks.copy()

    pick_df['type'] = pick_df['type'].replace({'s': 'S', 'p': 'P'})
        
    if 'amp' in pick_df.columns:
        pick_df=pick_df[['station_id','timestamp','type','amp']]
    else:
        pick_df=pick_df[['station_id','timestamp','type']]
    station_df=station_df[['station_id','x(km)','y(km)','z(km)']]
    if len(pick_df)>0:
        start_time_ref=UT(min(pick_df.timestamp))
    else:
        return pick_df, station_df, config
    
    pick_df['time_ref']=pick_df['timestamp'].apply(lambda x: UT(x) - start_time_ref)
    
    start_time=min(pick_df.time_ref)
    end_time=max(pick_df.time_ref)
    
    config['start_time_ref']=start_time_ref
    
    if 'time_before' in config:
        time_before =  config['time_before']
    else:
        time_before=((config["x(km)"][1]-config["x(km)"][0])**2+(config["y(km)"][1]-config["y(km)"][0])**2+(config["z(km)"][1]-config["z(km)"][0])**2)**0.5/config["vel"]['P']
        config['time_before']=time_before
        
    config['t(s)']=[start_time-time_before,end_time]


    config["x(km)"]=   (config["x(km)"][0]-(config["x(km)"][1]-config["x(km)"][0])*config['boundary_rate'],config["x(km)"][1]+(config["x(km)"][1]-config["x(km)"][0])*config['boundary_rate'])
    config["y(km)"]=   (config["y(km)"][0]-(config["y(km)"][1]-config["y(km)"][0])*config['boundary_rate'],config["y(km)"][1]+(config["y(km)"][1]-config["y(km)"][0])*config['boundary_rate'])
    config["z(km)"]=   (config["z(km)"][0]-(config["z(km)"][1]-config["z(km)"][0])*config['boundary_rate'],config["z(km)"][1]+(config["z(km)"][1]-config["z(km)"][0])*config['boundary_rate'])


    station_df['arrival_time_list_P'] = None
    station_df['arrival_pickindex_list_P'] = None
    station_df['arrival_time_list_S'] = None
    station_df['arrival_pickindex_list_S'] = None
    if config['P_phase']:
        for idx, station in station_df.iterrows():
            station_id=station.station_id
            arrival_time_list=pick_df['time_ref'][(pick_df.station_id==station_id)&(pick_df.type=='P')]
            arrival_pickindex_list=pick_df.index[(pick_df.station_id==station_id)&(pick_df.type=='P')]
            arrival_time_list=np.array(arrival_time_list.values)
            arrival_pickindex_list=np.array(arrival_pickindex_list.values)
            sorted_indices=np.argsort(arrival_time_list)
            station_df.at[idx,'arrival_time_list_P']=arrival_time_list[sorted_indices].tolist()
            station_df.at[idx,'arrival_pickindex_list_P']=arrival_pickindex_list[sorted_indices].tolist()

            
    if config['S_phase']:
        for idx, station in station_df.iterrows():
            station_id=station.station_id
            arrival_time_list=pick_df['time_ref'][(pick_df.station_id==station_id)&(pick_df.type=='S')]
            arrival_pickindex_list=pick_df.index[(pick_df.station_id==station_id)&(pick_df.type=='S')]
            arrival_time_list=np.array(arrival_time_list.values)
            arrival_pickindex_list=np.array(arrival_pickindex_list.values)
            sorted_indices=np.argsort(arrival_time_list)
            station_df.at[idx,'arrival_time_list_S']=arrival_time_list[sorted_indices].tolist()
            station_df.at[idx,'arrival_pickindex_list_S']=arrival_pickindex_list[sorted_indices].tolist()
        
            

    return pick_df, station_df, config

def run_harpa_wrapper(args):
    return run_harpa(*args)


def association(picks,station_df,config,verbose=0,model_traveltime=None):
    picks=copy.deepcopy(picks)
    default_config = {
        'neural_field': False,
        'optimize_wave_speed': False,
        'optimize_wave_speed_after_decay': True,
        'lr': 0.01,
        'noise': 1e-3,
        'patience_max': 10,
        'lr_decay': 0.1,
        'epoch_before_decay': 1000,
        'epoch_after_decay': 1000,
        'dbscan_min_samples': 1,
        'DBSCAN': True,
        'noisy_pick': True,
        'LSA': True,
        'wasserstein_p': 2,
        'max_time_residual':  2,
        'min_peak_pre_event': 16,
        'min_peak_pre_event_s': 0,
        'min_peak_pre_event_p': 0,
        'n_event_max_rate': 1,
        'n_event_max': None,
        'second_adjust': True,
        'P_phase': True, 
        'S_phase': False,
        'remove_overlap_events': False,
        'denoise_rate': 0,
        'beta_time':1,
        'beta_space':0.5,
        'beta_z':0.5,
        'init': 'data',
        'boundary_rate':0.05,
    }

    config = {**default_config, **config}



    if  config["DBSCAN"]:
        picks,unique_labels=DBSCAN_cluster(picks,station_df,config)
    else:
        picks['labels']=0
        unique_labels=[0]
        
    if "ncpu" not in config:
        config["ncpu"] = max(1, min(len(unique_labels) // 4, min(100, mp.cpu_count() - 1)))
    else:
        config["ncpu"] = min(mp.cpu_count(), config["ncpu"])
    if verbose>0:
        print(f"Associating {len(picks)} picks separated into {len(unique_labels)} slides with {config['ncpu']} CPUs")

    if config['neural_field']:
        config["ncpu"] = 1
        pick_df_list=[]
        catalog_df_list=[]
        for slice_index in tqdm(range(len(unique_labels))):    
            pick_df,catalog_df,z=run_harpa(picks=picks[picks['labels']==slice_index],station_df=station_df,config=config,verbose=verbose,skip=slice_index<0,model_traveltime=model_traveltime)
            pick_df_list.append(pick_df)
            catalog_df_list.append(catalog_df)
        
    else:
        if config["ncpu"] == 1:
            pick_df_list=[]
            catalog_df_list=[]
            for slice_index in tqdm(range(len(unique_labels))):    
                pick_df,catalog_df=run_harpa(picks=picks[picks['labels']==slice_index],station_df=station_df,config=config,verbose=verbose,skip=slice_index<0)
                pick_df_list.append(pick_df)
                catalog_df_list.append(catalog_df)
        else:
            args = [
            [
                picks[picks['labels']==slice_index],    # picks
                station_df,              # station_df
                config,            # harpa_config
                'cpu',                   # device
                verbose,                 # verbose
                slice_index<0,            # skip the association
                slice_index + 1024      # random seed
            ]
            for slice_index in range(-1,len(unique_labels))
            ]
            chunk_size = max(len(unique_labels) // (config["ncpu"] * 20), 1)
            #chunk_size=1
            results = process_map(run_harpa_wrapper, args, max_workers=config["ncpu"], chunksize=chunk_size)
            pick_df_list, catalog_df_list = zip(*results)
            pick_df_list = list(pick_df_list)
            catalog_df_list = list(catalog_df_list)

            # if platform.system().lower() in ["darwin", "windows"]:
            #     context = "spawn"
            # else:
            #     context = "fork"
            # chunk_size = max(len(unique_labels) // (config["ncpu"] * 20), 1)
            # with mp.get_context(context).Pool(config["ncpu"]) as p:
            #     results = p.starmap(
            #         run_harpa,
            #         [
            #             [
            #                 picks[picks['labels']==slice_index],    # picks
            #                 station_df,              # station_df
            #                 config,            # harpa_config
            #                 'cpu',                   # device
            #                 verbose,                 # verbose
            #                 slice_index<0,            # skip the association
            #                 slice_index + 1024      # random seed
            #             ]
            #             for slice_index in range(-1,len(unique_labels))
            #         ],
            #         chunksize=chunk_size,
            #     )
                            
            # pick_df_list, catalog_df_list=[],[]
            # for  pick_df, catalog_df in results:
            #     pick_df_list.append(pick_df)
            #     catalog_df_list.append(catalog_df)

    pick_df, catalog_df=reindex_picks_and_events(pick_df_list,catalog_df_list,config,overlap=0,verbose=verbose)
    if verbose>0:
        print(f' Associated {len(catalog_df)} unique events')
    if config['neural_field']:
        return pick_df, catalog_df,z
    else:
        return pick_df, catalog_df
        

def run_harpa(picks=None,station_df=None,config=None,device='cpu',verbose=0, skip=False,seed=0,model_traveltime=None):
    #print(".", end="")
    config=copy.deepcopy(config)

    lr = config['lr']
    noise = config['noise']
    patience_max = config['patience_max']
    lr_decay = config['lr_decay']
    epoch_before_decay = config['epoch_before_decay']
    epoch_after_decay = config['epoch_after_decay']
    if (~config['P_phase']) & (~config['S_phase']):
        ValueError('At least of one of P or S phase required')
    phases=[]
    if config['P_phase']:
        phases.append('P')
    if config['S_phase']:
        phases.append('S')
    
    pick_df, station_df, config=preprocess_data(picks=picks,station_df=station_df,config=config)
    
    n_station=len(station_df)
        
    if verbose>10:
        print("Pick format:", pick_df.iloc[:2])
        print("Station format:", station_df.iloc[:2])
        
    if (len(pick_df)<config['min_peak_pre_event']) or skip:
        pick_df['event_index']=-1
        pick_df['diff_time']=None
        if 'time_ref' in pick_df.columns:
            pick_df = pick_df.drop(columns=['time_ref'])
        catalog_df=pd.DataFrame(columns=['time','event_index','x(km)','y(km)','z(km)','num_p_picks','num_s_picks','num_picks'])
        
        return pick_df,catalog_df
    
    
    if config['n_event_max'] is None:
        n_event_max=0
        pick_station_id = pick_df.apply(lambda x: x.station_id + "_" + x.type, axis=1).to_numpy()
        n_event_max = max(Counter(pick_station_id).values())
        n_event_max= int(n_event_max*config['n_event_max_rate'])
        
    else:
        n_event_max=config['n_event_max']
        
    config['n_event_max']=n_event_max
    
    if verbose>6:
        print('max number of event in this slice:',n_event_max, 'number of picks in this slice:',len(pick_df))
    
    if verbose>4:
        print(config)
                    
    
    
    if config['neural_field']:
        if isinstance(model_traveltime, str):
            from .model import Siren
            L=config['wave_speed_model_hidden_dim']
            first_omega_0=30
            hidden_omega_0=30
            model_traveltime_cpu= Siren(in_features=3+L, out_features=len(station_df), hidden_features=128, 
                                    hidden_layers=3, outermost_linear=True,first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0).to('cpu')
            #print('loading model')
            model_traveltime_cpu.load_state_dict(torch.load(model_traveltime))
            #print('loading done')
            model_traveltime=copy.deepcopy(model_traveltime_cpu)
        else:
            model_traveltime=copy.deepcopy(model_traveltime.to('cpu'))
        Harpa=AssignmentModel_NF(config, station_df, seed=seed,device=device,model_traveltime=model_traveltime).to(device)
    else:    
        Harpa=AssignmentModel(config, station_df, seed=seed,device=device).to(device)
    # initialization of the location and time, not necessary  
    if 'init' in config:
        if config['init'] == 'data':
            times=pick_df['time_ref'].values
            meta = station_df.merge(pick_df["station_id"], how="right", on="station_id", validate="one_to_many")
            xs=meta['x(km)'].values
            ys=meta['y(km)'].values
            index = np.argsort(times)[:: max(len(times) // n_event_max, 1)][:n_event_max]
            t_init = times[index]
            x_init = xs[index]
            y_init = ys[index]
            z_init = np.linspace(config["z(km)"][0], config["z(km)"][1], 3)[1:-1]
            z_init = np.broadcast_to(z_init, (n_event_max)).reshape(-1)
            
            x_init =(x_init-config['x(km)'][0])/(config['x(km)'][1]-config['x(km)'][0])
            y_init =(y_init-config['y(km)'][0])/(config['y(km)'][1]-config['y(km)'][0])
            z_init =(z_init-config['z(km)'][0])/(config['z(km)'][1]-config['z(km)'][0])
            t_init =(t_init-config['t(s)'][0])/(config['t(s)'][1]-config['t(s)'][0])
            #event_locs=torch.tensor([x_init,y_init,z_init]).T
            event_locs = torch.tensor(np.array([x_init, y_init, z_init]).T)
            event_locs = torch.clamp(event_locs, min=0.05, max=0.95)
            event_locs=torch.logit(event_locs)/Harpa.beta_space

            event_times=torch.tensor(t_init)
            event_times = torch.clamp(event_times, min=0.05, max=0.95)
            event_times=torch.logit(event_times)/Harpa.beta_time
            Harpa.load_events(event_locs,event_times)
        
            
        if config['init'] == 'test':
            print('initlizing with ground true data (only for code check)')
            source_loc=np.array(config['true_xyz'])
            x_init=source_loc[:,0]
            y_init=source_loc[:,1]
            z_init=source_loc[:,2]
            df=config['start_time_ref']-UT(1970, 1, 1, 0, 0, 0, 0)
            t_init=np.array(config['true_time'])-df
            
            
            x_init =(x_init-config['x(km)'][0])/(config['x(km)'][1]-config['x(km)'][0])
            y_init =(y_init-config['y(km)'][0])/(config['y(km)'][1]-config['y(km)'][0])
            z_init =(z_init-config['z(km)'][0])/(config['z(km)'][1]-config['z(km)'][0])
            t_init =(t_init-config['t(s)'][0])/(config['t(s)'][1]-config['t(s)'][0])
            
            #event_locs=torch.tensor([x_init,y_init,z_init]).T
            event_locs = torch.tensor(np.array([x_init, y_init, z_init]).T)
            event_locs = torch.clamp(event_locs, min=0.05, max=0.95)
            event_locs=torch.logit(event_locs)/Harpa.beta_space
            
            event_times=torch.tensor(t_init)
            event_times = torch.clamp(event_times, min=0.05, max=0.95)
            event_times=torch.logit(event_times)/Harpa.beta_time
            Harpa.load_events(event_locs,event_times)
            
            if config['neural_field'] and 'true_z' in config:
                z=torch.tensor(config['true_z'])
                z = torch.clamp(z, min=0.05, max=0.95)
                p=torch.logit(z)/Harpa.beta_z
                Harpa.load_model_traveltime_p(p)

    
    Harpa.reset_training()    
    if config['neural_field'] and (not config['optimize_wave_speed']):
        z = torch.tensor(config['wave_speed_z'])
        z = torch.clamp(z, min=0.05, max=0.95)
        p=torch.logit(z)/Harpa.beta_z
        Harpa.load_model_traveltime_p(p)
        if config['optimize_wave_speed_after_decay']:
            Harpa.model_traveltime_p.requires_grad = False

    optimizer = SGLD(Harpa.parameters(), lr=lr,noise=noise)   

    loss_count=0
    if verbose>2:
        print('Running SGLD...')
        
    loss_count=[]
    loss_best=1000
    patience_max=patience_max
    patience_count=0
    noisy_pick=config['noisy_pick']
    LSA=config['LSA']
    wasserstein_p=config['wasserstein_p']
    for step in range(epoch_before_decay+epoch_after_decay):
        station_index=np.random.randint(n_station)
        phase=random.choice(phases)
        #station_index=1
        true_time_list=station_df['arrival_time_list_'+phase][station_index]
        #if step==10000:
        if step==epoch_after_decay:
            if config['neural_field']:
                if config['optimize_wave_speed_after_decay']:
                    Harpa.model_traveltime_p.requires_grad = True
            optimizer = SGLD(Harpa.parameters(), lr=lr*lr_decay,noise=noise*lr_decay)
        if len(true_time_list)>0:
            time_list,sort_index=Harpa(station_index,phase=phase)
            true_time_list=torch.tensor(true_time_list).to(device)
            #loss,_=compute_assignment_loss(time_list,true_time_list)
            loss,_=compute_assignment_loss(time_list,true_time_list,noisy_pick=noisy_pick,LSA=LSA,p=wasserstein_p)
            #print(time_list,true_time_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_count.append(loss.item())
        if (step+1)%100==0:
            if len(loss_count)>0:
                f_loss_loss_count=np.array(loss_count).mean()
            else:
                f_loss_loss_count=loss_best-1
            if step>10000:
                if f_loss_loss_count<loss_best:
                    loss_best=f_loss_loss_count
                    event_locs=Harpa.event_locs.data
                    event_times=Harpa.event_times.data
                    patience_count=0
                    if config['neural_field']:
                        if config['optimize_wave_speed_after_decay']:
                            model_traveltime_p=Harpa.model_traveltime_p.data
                    
                else:
                    patience_count=patience_count+1
                if patience_count>patience_max:
                    break
            if verbose>4:
                if config['neural_field']:
                    z=torch.sigmoid(Harpa.model_traveltime_p.data*Harpa.beta_z)
                    z=z.numpy()
                    print(f'Step: {step}, Loss: {f_loss_loss_count :.2f}, z: {z}')
                else:
                    print(f'Step: {step}, Loss: {f_loss_loss_count :.2f}')
            loss_count=[]
    if step>10000:
        Harpa.load_events(event_locs,event_times)
    if verbose>2:
        print(f'Final SGLD loss: {loss_best :.2f}')
        
    
    max_time_residual=config['max_time_residual']
    min_peak_pre_event=config['min_peak_pre_event']
    min_peak_pre_event_s=config['min_peak_pre_event_s']
    min_peak_pre_event_p=config['min_peak_pre_event_p']
    start_time_ref=config['start_time_ref']
    # print(pick_df)
    # print(station_df)
    if config['second_adjust']:
        pick_df, catalog_df=annotation(pick_df,station_df,Harpa,max_time_residual,min_peak_pre_event,start_time_ref,sort_evnet_index=False,phases=phases,min_peak_pre_event_s=min_peak_pre_event_s,min_peak_pre_event_p=min_peak_pre_event_p)
        
        
        Harpa.reset_training()
        n_event_max=len(Harpa.event_times)
        for training_index in range(n_event_max):
            event_picks_index_P=pick_df.index[(pick_df['event_index']==training_index) & (pick_df['type']=='P')]
            event_picks_index_S=pick_df.index[(pick_df['event_index']==training_index) & (pick_df['type']=='S')]

            if len(event_picks_index_P)+len(event_picks_index_S)>0:
                optimizer = torch.optim.Adam(Harpa.parameters(), lr=0.01)
                #optimizer = SGLD(Harpa.parameters(), lr=0.01,noise=0.)
                pick_times_list_P=torch.tensor(pick_df['time_ref'][event_picks_index_P].values).to(device)
                pick_times_list_S=torch.tensor(pick_df['time_ref'][event_picks_index_S].values).to(device)
                
                id_list_P=pick_df['station_id'][event_picks_index_P]
                id_list_S=pick_df['station_id'][event_picks_index_S]
                event_picks_index_station_index_P = [station_df.index[station_df['station_id'] == id_value].tolist()[0] for id_value in id_list_P]
                event_picks_index_station_index_S = [station_df.index[station_df['station_id'] == id_value].tolist()[0] for id_value in id_list_S]
                lastloss=100000
                for step in range(100):
                    optimizer.zero_grad()
                    loss=0
                    if config['P_phase']:
                        for i,station_index in enumerate(event_picks_index_station_index_P):
                            arrival_time_P=Harpa.arrival_time(station_index,phase='P')[training_index]
                            loss+=(pick_times_list_P[i]-arrival_time_P)**2
                    if config['S_phase']:
                        for i,station_index in enumerate(event_picks_index_station_index_S):
                            arrival_time_S=Harpa.arrival_time(station_index,phase='S')[training_index]
                            loss+=(pick_times_list_S[i]-arrival_time_S)**2                            
                    if step%10==0:
                        if verbose>8:
                            print(f'Training index: {training_index}, Step: {step}, Loss: {loss.item() :.2f}')
                    #loss=loss/max_p_travel_time
                    loss.backward()
                    optimizer.step()
                    if loss.item()-lastloss>0.000001:
                        break
                    lastloss=loss.item()
                              
    pick_df, catalog_df=annotation(pick_df,station_df,Harpa,max_time_residual,min_peak_pre_event,start_time_ref,sort_evnet_index=True,phases=phases,min_peak_pre_event_s=min_peak_pre_event_s,min_peak_pre_event_p=min_peak_pre_event_p)

    if config['denoise_rate']>0 and len(catalog_df)>0:
        if verbose>2:
            print('Filtering based on Nearest Station Ratio:',config['denoise_rate'])
        pick_df, catalog_df=denoise_events(pick_df,catalog_df,station_df,km=True,MIN_NEAREST_STATION_RATIO=config['denoise_rate'],verbose=verbose)
    if 'time_ref' in pick_df.columns:
        pick_df = pick_df.drop(columns=['time_ref'])
    if 'time_ref' in catalog_df.columns:
        catalog_df = catalog_df.drop(columns=['time_ref'])
    if config['neural_field']:
        z=torch.sigmoid(Harpa.model_traveltime_p.data*Harpa.beta_z).to('cpu')
        z=z.numpy()
        return pick_df,catalog_df, z
    else:
        return pick_df,catalog_df

