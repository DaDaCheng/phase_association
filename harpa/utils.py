from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import copy
from obspy import UTCDateTime as UT
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd


def empty_pick_def():
    return pd.co


def rename_df(picks,stations):

    if not ('station_id' in stations.columns):
        stations.rename(columns={"id": "station_id" }, inplace=True)
    if not ('station_id' in picks.columns):
        picks.rename(columns={"id": "station_id" }, inplace=True)
        
    if not ('event_index' in picks.columns):
        picks.rename(columns={"event_idx": "event_index" }, inplace=True)
    return picks, stations



def count_ari_nmi(true_label,my_label):
    ari = adjusted_rand_score(true_label, my_label)
    nmi = normalized_mutual_info_score(true_label, my_label)
    return ari,nmi

def compute_assignment_loss(list_a,list_b):
    if len(list_a)>len(list_b):
        a=list_a
        b=list_b
    else:
        b=list_a
        a=list_b
    distance_matrix = torch.abs(a[:, None] - b[None, :])

    row_ind, col_ind = linear_sum_assignment(distance_matrix.clone().detach().cpu().numpy())
    
    matched_a = a[row_ind]
    matched_b = b[col_ind]
    
    loss = torch.sum(torch.abs(matched_a - matched_b))
    return loss, row_ind.tolist()






def annotation(pick_df,station_df,Harpa,max_time_residue,min_peak_pre_event,start_time_ref,sort_evnet_index=False,phases=['P'],min_peak_pre_event_p=-1,min_peak_pre_event_s=-1):
    pick_df=pick_df.copy()
    pick_df['event_index'] = None
    pick_df['diff_time'] = None
    Harpa=copy.deepcopy(Harpa)
    n_event=len(Harpa.event_times)
    n_station=station_df.index.max()+1
    for phase in phases:
        
        for station_index in range(n_station):
            true_time_list=station_df['arrival_time_list_'+phase][station_index]
            pickindex_list=station_df['arrival_pickindex_list_'+phase][station_index]
            #print(phase,station_index,true_time_list,pickindex_list)
            n_pick=len(true_time_list)
            if n_pick<1:
                continue
            else:
                with torch.no_grad():
                    time_list,sort_index=Harpa(station_index,phase=phase) 
                    time_list,sort_index=time_list.to('cpu'),sort_index.to('cpu')
                true_time_list=torch.tensor(true_time_list)
                
                _,index=compute_assignment_loss(time_list,true_time_list)
                
                if len(time_list)>=len(true_time_list):
                    assignment_results=sort_index[index]
                    for i,pickindex in enumerate(pickindex_list):
                        even_idx_new=assignment_results[i].item()
                        time_ref_new=time_list[index[i]].item()
                        dtime=time_ref_new-true_time_list[i].item()
                        if np.abs(dtime)>max_time_residue:
                            even_idx_new=-1
                        pick_df.at[pickindex,'event_index']=even_idx_new
                        pick_df.at[pickindex,'diff_time']=dtime
                else:
                    for i,pickindex in enumerate(pickindex_list):
                        index=np.array(index)
                        if i in index:
                            even_idx_new=sort_index[index==i].item()
                            time_ref_new=time_list[index==i].item()
                            dtime=time_ref_new-true_time_list[i].item()
                            if np.abs(dtime)>max_time_residue:
                                even_idx_new=-1
                        else:
                            even_idx_new=-1
                            dtime=None
                        pick_df.at[pickindex,'event_index']=even_idx_new
                        pick_df.at[pickindex,'diff_time']=dtime
                  
    
    event_locs,event_times=Harpa.loc_and_time()
    event_locs,event_times=event_locs.detach().clone().cpu().numpy(),event_times.detach().clone().cpu().numpy()
    catalog_df=[]
    old_event_idx_list=[]
    new_event_idx_list=[]
    for event_index in range(n_event):
        old_event_idx_list.append(event_index)
        
        # if '..' in pick_df['station_id'].iloc[0]:
        #     unique_station_id_prefix = pick_df['station_id'][(pick_df.event_index == event_index) & (pick_df.type == 'P')].str.extract(r'^(.*?)\.\.')
        #     num_p_picks = unique_station_id_prefix[0].nunique()
        #     unique_station_id_prefix = pick_df['station_id'][(pick_df.event_index == event_index) & (pick_df.type == 'S')].str.extract(r'^(.*?)\.\.')
        #     num_s_picks = unique_station_id_prefix[0].nunique()
        # else:
        #     num_p_picks=((pick_df.event_index==event_index)& (pick_df.type=='P')).sum()
        #     num_s_picks=((pick_df.event_index==event_index)& (pick_df.type=='S')).sum()
        num_p_picks=pick_df['station_id'][(pick_df.event_index==event_index)& (pick_df.type=='P')].nunique()
        num_s_picks=pick_df['station_id'][(pick_df.event_index==event_index)& (pick_df.type=='S')].nunique()
        
        num_picks=num_s_picks+num_p_picks

        num_picks_bool= num_picks>=min_peak_pre_event
        if 'P' in phases:
            num_picks_bool=num_picks_bool and num_p_picks>min_peak_pre_event_p
        if 'S' in phases:
            num_picks_bool=num_picks_bool and num_s_picks>min_peak_pre_event_s
        if num_picks_bool:
            catalog_df.append({
                "time": start_time_ref+event_times[event_index],
                "event_index": event_index,
                "time_ref":event_times[event_index],
                "x(km)": event_locs[event_index,0],
                "y(km)": event_locs[event_index,1],
                "z(km)": event_locs[event_index,2],
                "num_p_picks":  num_p_picks,
                "num_s_picks":  num_s_picks,
                "num_picks":  num_picks,
            })
            new_event_idx_list.append(event_index)
        else:
            new_event_idx_list.append(-1)
            
    if len(catalog_df)>0:
        catalog_df = pd.DataFrame(catalog_df)
    else:
        catalog_df=pd.DataFrame(columns=['time','event_index','time_ref','x(km)','y(km)','z(km)','num_p_picks','num_s_picks','num_picks'])

    replace_dict = dict(zip(old_event_idx_list, new_event_idx_list))
    pick_df['event_index'] = pick_df['event_index'].replace(replace_dict)
    
    if sort_evnet_index:
        catalog_df=catalog_df.sort_values('time_ref')
        catalog_df.reset_index(drop=True, inplace=True)
        new_event_idx_list=catalog_df.index.values
        old_event_idx_list=catalog_df.event_index.values
        catalog_df['event_index']=new_event_idx_list
        
        replace_dict = dict(zip(old_event_idx_list, new_event_idx_list))
        pick_df['event_index'] = pick_df['event_index'].replace(replace_dict)
    return pick_df,catalog_df





def denoise_events(pick_df,catalog_df,station_df,km=True, MIN_NEAREST_STATION_RATIO = 0.3,verbose=2):
    events = catalog_df.copy()
    picks = pick_df.copy()
    pick_df = pick_df.copy()
    stations = station_df.copy()
    if not ('station_id' in stations.columns):
        stations.rename(columns={"id": "station_id" }, inplace=True)
    if not ('station_id' in picks.columns):
        picks.rename(columns={"id": "station_id" }, inplace=True)
        
    if not ('event_index' in picks.columns):
        picks.rename(columns={"event_idx": "event_index" }, inplace=True)
        
    
    if km:
        longitude='x(km)'
        latitude='x(km)'
    else:
        longitude='longitude'
        latitude='latitude'

    stations = stations[stations["station_id"].isin(picks["station_id"].unique())]

    neigh = NearestNeighbors(n_neighbors=min(len(stations), 10))
    neigh.fit(stations[[longitude, latitude]].values)

    picks = picks.merge(events[["event_index", longitude, latitude]], on="event_index", suffixes=("", "_event"))
    picks = picks.merge(stations[["station_id", longitude, latitude]], on="station_id", suffixes=("", "_station"))

    filtered_events = []
    #for i, event in tqdm(events.iterrows(), total=len(events)):
    for i, event in events.iterrows():
        sid = neigh.kneighbors([[event[longitude], event[latitude]]])[1][0]
        picks_ = picks[picks["event_index"] == event["event_index"]]
        # longitude, latitude = picks_[["longitude", "latitude"]].mean().values
        # sid = neigh.kneighbors([[longitude, latitude]])[1][0]
        stations_neigh = stations.iloc[sid]["station_id"].values
        picks_neigh = picks_[picks_["station_id"].isin(stations_neigh)]
        stations_with_picks = picks_neigh["station_id"].unique()
        if len(stations_with_picks) / len(stations_neigh) > MIN_NEAREST_STATION_RATIO:
            #print(event)
            filtered_events.append(event)
    if verbose>1:             
        print(f"Events before filtering: {len(events)}")
        print(f"Events after filtering: {len(filtered_events)}")
    if len(filtered_events)>0:    
        catalog_df = pd.DataFrame(filtered_events)
        catalog_df = catalog_df.reset_index(drop=True)
        n_event_max=pick_df.event_index.max()+1
        old_event_idx_list=(catalog_df.event_index.values).tolist()
        new_event_idx_list=(catalog_df.index.values).tolist()
        for i in range(n_event_max):
            if i in old_event_idx_list:
                pass
            else:
                old_event_idx_list.append(i)
                new_event_idx_list.append(-1)
                            
        replace_dict = dict(zip(old_event_idx_list, new_event_idx_list))
        pick_df['event_index'] = pick_df['event_index'].replace(replace_dict)
        catalog_df['event_index'] = catalog_df['event_index'].replace(replace_dict)
    else:
        catalog_df=pd.DataFrame(columns=['time','event_index','time_ref','x(km)','y(km)','z(km)','num_p_picks','num_s_picks','num_picks'])
        pick_df['event_index']=-1
    return pick_df, catalog_df



def catalog_freq(catalog_df, interval=2*60*60):
    # Convert 'time' to UTCDateTime and then to seconds
    catalog_df['time'] = catalog_df['time'].apply(lambda x: UT(x))
    catalog_df = catalog_df.sort_values('time').reset_index(drop=True)
    catalog_df['time_seconds'] = catalog_df['time'].apply(lambda x: x.timestamp)

    # Create arrays for quick vectorized operations
    time_seconds = catalog_df['time_seconds'].values
    
    # Calculate frequency
    catalog_df['freq'] = np.zeros(len(catalog_df))

    for i in tqdm(range(len(catalog_df))):
        # Calculate time window bounds
        time_left = time_seconds[i] - interval/2
        time_right = time_seconds[i] + interval/2
        
        # Use binary search to find the number of events in the time window
        left_index = np.searchsorted(time_seconds, time_left, side='left')
        right_index = np.searchsorted(time_seconds, time_right, side='right')
        
        n_events = right_index - left_index
        catalog_df.at[i, 'freq'] = n_events/interval

    # Drop the helper column
    catalog_df.drop(columns=['time_seconds'], inplace=True)

    return catalog_df



def catalog_interval(catalog_df, interval_number=10):
    catalog_df['time'] = catalog_df['time'].apply(lambda x: UT(x))
    catalog_df = catalog_df.sort_values('time').reset_index(drop=True)
    catalog_df['time_seconds'] = catalog_df['time'].apply(lambda x: x.timestamp)

    # Create arrays for quick vectorized operations
    time_seconds = catalog_df['time_seconds'].values
    
    # Calculate frequency
    #catalog_df['freq'] = np.zeros(len(catalog_df))
    catalog_df['interval(s)']=None
    half_interval=int(interval_number/2)
    for i in range(len(catalog_df)):
        if i-half_interval<=0:
            inter=(time_seconds[interval_number]-time_seconds[0])/interval_number
        elif i+half_interval>=len(catalog_df):
            inter=(time_seconds[-1]-time_seconds[-1-interval_number])/interval_number
        else:
            inter=(time_seconds[i+half_interval]-time_seconds[i-half_interval])/interval_number
        # if inter<0:
        #     print(i)
        catalog_df.at[i, 'interval(s)'] = inter

    # Drop the helper column
    catalog_df.drop(columns=['time_seconds'], inplace=True)

    return catalog_df



def calc_mag(data, event_loc, station_loc, weight, min=-2, max=8):
    dist = np.linalg.norm(event_loc[:, :-1] - station_loc, axis=-1, keepdims=True)
    # mag_ = ( data - 2.48 + 2.76 * np.log10(dist) )
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (data - c0 - c3 * np.log10(np.maximum(dist, 0.1))) / c1 + 3.5
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # mag_ = (data - c0 - c3*np.log10(dist))/c1
    # mag = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    # (Watanabe, 1971) https://www.jstage.jst.go.jp/article/zisin1948/24/3/24_3_189/_pdf/-char/ja
    # mag_ = 1.0/0.85 * (data + 1.73 * np.log10(np.maximum(dist, 0.1)) + 2.50)
    mu = np.sum(mag_ * weight) / (np.sum(weight) + 1e-6)
    std = np.sqrt(np.sum((mag_ - mu) ** 2 * weight) / (np.sum(weight) + 1e-12))
    mask = np.abs(mag_ - mu) <= 2 * std
    mag = np.sum(mag_[mask] * weight[mask]) / (np.sum(weight[mask]) + 1e-6)
    mag = np.clip(mag, min, max)
    return mag


def merge_catalogs(catalog_out_df,catalog_df_next,pick_df_next,threshold_time=1000,threshold_xyz = 1000):
    import numpy as np
    from obspy import UTCDateTime
    from scipy.spatial.distance import cdist

    def process_catalog_inputs(catalog_out_df, catalog_df_next):
        if not catalog_df_next[catalog_df_next["event_index"].isin(catalog_out_df["event_index"])].empty:
            # Merge those events in both catalog_out_df and catalog_df_next
            agg_dict = {col: 'first' for col in catalog_out_df.columns if col not in ["num_p_picks", "num_s_picks", "num_picks"]}
            agg_dict["num_p_picks"] = "sum"
            agg_dict["num_s_picks"] = "sum"
            agg_dict["num_picks"] = "sum"
            catalog_out_df = pd.concat([catalog_out_df, catalog_df_next[catalog_df_next["event_index"].isin(catalog_out_df["event_index"])]]).groupby('event_index', as_index=False).agg(agg_dict)
            catalog_df_next = catalog_df_next[~catalog_df_next["event_index"].isin(catalog_out_df["event_index"])]
        return catalog_out_df, catalog_df_next
    catalog_out_df, catalog_df_next = process_catalog_inputs(catalog_out_df, catalog_df_next)
    # define overlap catalog
    catalog_overlap = catalog_out_df[
                    (catalog_out_df['time'] > catalog_df_next["time"].min() - pd.Timedelta(seconds=threshold_time)) &
                    (catalog_out_df['time'] < catalog_df_next["time"].max() + pd.Timedelta(seconds=threshold_time))
                ]
    catalog_non_overlap = catalog_out_df[
                    (catalog_out_df['time'] <= catalog_df_next["time"].min() - pd.Timedelta(seconds=threshold_time)) |
                    (catalog_out_df['time'] >= catalog_df_next["time"].max() + pd.Timedelta(seconds=threshold_time))
                ]
    if catalog_overlap.empty:
        catalog_out_df = pd.concat([catalog_out_df,catalog_df_next])
        return catalog_out_df,pick_df_next
    # Compute time differences in seconds
    time_diff = np.abs(catalog_overlap["time"].values[:, None] - catalog_df_next["time"].values)
    # Compute geographic distances using the x, y, z columns
    coords_out = catalog_overlap[["x(km)", "y(km)", "z(km)"]].values
    coords_next = catalog_df_next[["x(km)", "y(km)", "z(km)"]].values
    geo_dist = cdist(coords_out, coords_next)
    # Identify pairs that satisfy both the time and geographic distance conditions
    valid_pairs = (time_diff <= threshold_time) & (geo_dist <= threshold_xyz)
    # Prepare lists to store merged rows
    merged_rows = []
    # Loop through valid pairs and merge
    for i, j in zip(*np.where(valid_pairs)):
        merged_row = catalog_overlap.iloc[i].copy()
        next_row = catalog_df_next.iloc[j]
        # print(f"Merged: {merged_row['event_index']} and {next_row['event_index']}, Time difference: {time_diff[i, j]} seconds, Geographic distance: {geo_dist[i, j]} km")
        # print("=====================================")
         # Calculate the average of time in seconds since the epoch
        avg_time = (merged_row["time"].timestamp + next_row["time"].timestamp) / 2
        avg_time = UTCDateTime(avg_time)  # Convert back to UTCDateTime
        # Calculate the merged row
        merged_row["x(km)"] = (merged_row["x(km)"] + next_row["x(km)"]) / 2
        merged_row["y(km)"] = (merged_row["y(km)"] + next_row["y(km)"]) / 2
        merged_row["z(km)"] = (merged_row["z(km)"] + next_row["z(km)"]) / 2
        merged_row["time"] = avg_time
        merged_row["num_p_picks"] = merged_row["num_p_picks"] + next_row["num_p_picks"]
        merged_row["num_s_picks"] = merged_row["num_s_picks"] + next_row["num_s_picks"]
        merged_row["num_picks"] = merged_row["num_picks"] + next_row["num_picks"]
        # Update event_index in pick_df_next for this merged pair
        event_index_to_update = merged_row["event_index"]
        catalog_event_index = next_row["event_index"]
        pick_df_next.loc[pick_df_next["event_index"] == catalog_event_index, "event_index"] = event_index_to_update

    # Convert merged rows into a DataFrame
    merged_df = pd.DataFrame(merged_rows)
    # Remove merged rows from the original dataframes
    catalog_overlap = catalog_overlap.drop(catalog_overlap.index[np.where(valid_pairs)[0]])
    catalog_df_next = catalog_df_next.drop(catalog_df_next.index[np.where(valid_pairs)[1]])
    # Concatenate the merged rows and remaining rows
    catalog_out_df = pd.concat([catalog_overlap, merged_df,catalog_non_overlap,catalog_df_next])
    catalog_out_df = catalog_out_df.drop_duplicates(subset='event_index', keep='first')

    return catalog_out_df,pick_df_next


def get_index_update_map_from_duplicates(pick_df_next, pick_df_overlap_eventpick):
    """
    Function to update event indices in the next DataFrame based on overlapping event picks from previous data.
    
    Parameters:
    - pick_df_next: DataFrame containing the next picks 
    - pick_df_overlap_eventpick: DataFrame with overlapping event picks from previous data
    
    Returns:
    - map of event_index to updated event_index
    """
    THRESHOLD_OF_DUPLICATED_INDEX_MATCH = 0.8
    index_update_map = {}
    # Define indices for merging and comparison
    index_cols = ['station_id', 'timestamp', 'type']
    # Create a merged DataFrame to compare event indices directly
    merged_df = pd.merge(pick_df_next.set_index(index_cols), pick_df_overlap_eventpick.set_index(index_cols),
                         suffixes=('_next', '_overlap'), how='inner', left_index=True, right_index=True) # with inner join, only duplicate indices are kept
    # Use groupby to count the number of matching event index pairs
    comparison_counts = merged_df.groupby( ['event_index_next', 'event_index_overlap']).size()
    # Sum up the total counts for each event index in the _next DataFrame
    total_counts = comparison_counts.groupby(level=['event_index_next']).sum()
    # Calculate percentages of matching event indices
    comparison_percentages = comparison_counts / total_counts
    # Filter entries where more than THRESHOLD_OF_DUPLICATED_INDEX_MATCH of the matches is duplicate of the same event index
    dominant_indices = comparison_percentages[comparison_percentages > THRESHOLD_OF_DUPLICATED_INDEX_MATCH]
    # Update pick_df_next with the dominant event indices where conditions are met
    for idx, _ in dominant_indices.items():
        event_index_to_update, event_index_overlap = idx
        if event_index_to_update == -1:
            continue
        index_update_map[event_index_to_update] = event_index_overlap

    return index_update_map

def reindex_picks_and_events(pick_df_list,catalog_df_list,harpa_config,overlap=None,verbose=1000):

    def merge_pick_df(pick_df_overlap,pick_df_next,pick_out_df,pick_df_next_first_event_time,pick_df_next_last_event_time):
        import pandas as pd
        # Merge overlapping picks and remove duplicates
        pick_merged_df = pd.merge(pick_df_overlap, pick_df_next, left_index=True, right_index=True, how='outer', suffixes=('_overlap', '_next'))  
        mask = (pick_merged_df['event_index_overlap'] == -1) | (pick_merged_df['event_index_overlap'].isna())
        for column in pick_df_overlap.columns:
            condition_met = (pd.notna(pick_merged_df[f'{column}_next']) & mask)
            # Use np.where for a vectorized conditional operation
            pick_merged_df[column] = np.where(condition_met, pick_merged_df[f'{column}_next'], pick_merged_df[f'{column}_overlap'])
        columns_to_drop = [col for col in pick_merged_df.columns if col.endswith('_overlap') or col.endswith('_next')]
        pick_merged_df.drop(columns=columns_to_drop, inplace=True)
        pick_merged_df.index.name = pick_df_overlap.index.name
        start_time = pick_df_next_first_event_time - pd.Timedelta(seconds=overlap)
        end_time = pick_df_next_last_event_time + pd.Timedelta(seconds=overlap)
        # Concatenate non-overlapping picks with the merged picks
        pick_df_non_overlap = pick_out_df.query(
            "(timestamp <= @start_time) | "
            "(timestamp >= @end_time)"
        )
        pick_out_df = pd.concat([pick_df_non_overlap, pick_merged_df]).copy()

        return pick_out_df

    # Print the status message
    if verbose>2:
        print('Reindexing picks and events...')

    if overlap==None:
        overlap = harpa_config['window_lenth']-harpa_config['window_lenth_shift']
    # Initialize output DataFrames as None
    pick_out_df = None
    catalog_out_df = None
    n_repeat_event=0
    min_peak_pre_event=harpa_config['min_peak_pre_event']
    if harpa_config['remove_overlap_events']:
        remove_overlap_events=True
        threshold_time= harpa_config['threshold_time']
        threshold_xyz = harpa_config['threshold_xyz']
    else:
        remove_overlap_events=False
    # Iterate through each pair of pick_df and catalog_df in the provided lists
    for pick_df, catalog_df in tqdm(zip(pick_df_list, catalog_df_list), total=len(pick_df_list)):
        if not isinstance(pick_df, pd.DataFrame):
            continue  # Skip if pick_df is not a DataFrame
        if pick_df.empty:
            continue  # Skip if pick_df is empty

        if not isinstance(pick_out_df, pd.DataFrame):
            # If output DataFrames are not yet initialized, copy the first DataFrame pair
            pick_out_df = pick_df.copy()
            catalog_out_df = catalog_df.copy()
            current_index = pick_out_df.event_index.max() + 1
        else:
            # Create copies of the current DataFrame pair
            pick_df_next = pick_df.copy()
            catalog_df_next = catalog_df.copy()
            # Determine the time range of the current set of picks
            pick_df_next_first_event_time = pick_df_next.timestamp.min()
            pick_df_next_last_event_time = pick_df_next.timestamp.max()
            # now we remove overlaped picks
            if overlap > 0:
                # Identify overlapping picks between the current and previous DataFrames
                pick_df_overlap = pick_out_df[
                    (pick_out_df['timestamp'] > pick_df_next_first_event_time - pd.Timedelta(seconds=overlap)) &
                    (pick_out_df['timestamp'] < pick_df_next_last_event_time + pd.Timedelta(seconds=overlap))
                ]
                
                # Filter out picks that are duplicates based on station_id, timestamp, and type
                pick_df_overlap_eventpick = pick_df_overlap[pick_df_overlap['event_index'] > -1]
                # analyze duplicated picks to find match of event_index
                non_duplicate_df = pick_df_next[
                    ~pick_df_next.set_index(['station_id', 'timestamp', 'type']).index.isin(
                        pick_df_overlap_eventpick.set_index(['station_id', 'timestamp', 'type']).index)
                ].copy()
                index_update_map = get_index_update_map_from_duplicates(pick_df_next, pick_df_overlap_eventpick)
                # Identify the next valid event indices based on the minimum number of peaks required
                event_counts = pick_df_next['event_index'].value_counts()
                next_event_index_list = event_counts.index[(event_counts >= min_peak_pre_event)]

                next_event_index_list = next_event_index_list.tolist()
                if -1 in next_event_index_list:
                    next_event_index_list.remove(-1)


                # Create a mapping of old event indices to new ones
                old_event_idx_list = catalog_df_next.index.values
                new_event_idx_list = []
                temp_index = 0
                for index in catalog_df_next.index.values:
                    if index in index_update_map and pick_df_next[pick_df_next['event_index'] == index].shape[0] >= min_peak_pre_event:
                        # Update the event index based on the mapping obtained from duplicated picks
                        new_event_idx_list.append(index_update_map[index])
                    elif index in next_event_index_list and non_duplicate_df[non_duplicate_df['event_index'] == index].shape[0] >= min_peak_pre_event:
                        new_event_idx_list.append(temp_index + current_index)
                        temp_index = temp_index + 1
                    else:                   
                        new_event_idx_list.append(-1)
                
                # # Apply the new event index mapping
                replace_dict = dict(zip(old_event_idx_list, new_event_idx_list))
                pick_df_next['event_index'] = pick_df_next['event_index'].replace(replace_dict)
                
                # Count the number of P-type/S-type picks associated with each event
                num_p_picks = non_duplicate_df[non_duplicate_df['type'] == 'P'].groupby('event_index').size()
                num_s_picks = non_duplicate_df[non_duplicate_df['type'] == 'S'].groupby('event_index').size()
                catalog_df_next['num_p_picks'] = catalog_df_next['event_index'].map(num_p_picks).fillna(0).astype(int)
                catalog_df_next['num_s_picks'] = catalog_df_next['event_index'].map(num_s_picks).fillna(0).astype(int)
                catalog_df_next['num_picks'] = (catalog_df_next['num_p_picks'] + catalog_df_next['num_s_picks']).fillna(0).astype(int)
                catalog_df_next['event_index'] = catalog_df_next['event_index'].replace(replace_dict)
                # Remove events that are no longer valid
                n_event_next = len(catalog_df_next)
                catalog_df_next = catalog_df_next[catalog_df_next['event_index'] > -1]
                catalog_df_next = catalog_df_next.reset_index(drop=True)

                if catalog_df_next.empty:
                    pick_out_df = merge_pick_df(pick_df_overlap,pick_df_next,pick_out_df,pick_df_next_first_event_time,pick_df_next_last_event_time)  # merge pick_out_df with next and continue
                    current_index = catalog_out_df.event_index.max() + 1 if not catalog_out_df.empty else 0
                    continue
                # Update the number of repeated events
                n_event_next_remove_repeat = len(catalog_df_next)
                n_repeat_event += n_event_next - n_event_next_remove_repeat
                
                # Reindex the catalog and concatenate it with the existing catalog
                catalog_df_next.index += current_index
                if catalog_out_df.empty:
                    catalog_out_df = catalog_df_next.copy()
                elif remove_overlap_events:
                    catalog_out_df,pick_df_next = merge_catalogs(catalog_out_df, catalog_df_next,pick_df_next, threshold_time, threshold_xyz)
                else:
                    catalog_out_df=pd.concat([catalog_out_df,catalog_df_next])
                    agg_dict = {}
                    for col in catalog_out_df.columns:
                        if col in ["num_p_picks", "num_s_picks", "num_picks"]:
                            agg_dict[col] = "sum"
                        else:
                            agg_dict[col] = "first"
                    catalog_out_df = catalog_out_df.groupby('event_index', as_index=False).agg(agg_dict)
                # Merge overlapping picks and remove duplicates
                pick_out_df = merge_pick_df(pick_df_overlap,pick_df_next,pick_out_df,pick_df_next_first_event_time,pick_df_next_last_event_time)          
            else:
                # If no overlap, simply reindex the next picks and catalog and concatenate them with the existing DataFrames
                pick_df_next = pick_df.copy()
                pick_df_next.loc[pick_df_next['event_index'] > -1, 'event_index'] += current_index
                pick_out_df = pd.concat([pick_out_df, pick_df_next])
                catalog_df_next = catalog_df.copy()
                catalog_df_next.loc[:, 'event_index'] += current_index
                catalog_df_next.index += current_index
                catalog_out_df = pd.concat([catalog_out_df, catalog_df_next])
        # Update the current index to continue reindexing in the next iteration
        
        if len(catalog_out_df)==0:
            current_index=0
        else:
            current_index = catalog_out_df.event_index.max() + 1 

    # Print the final result indicating the number of removed repeated events and the number of unique events left
    if verbose>2:
        print(f'removed {n_repeat_event} repeated events, left {len(catalog_out_df)} unique events')
    #if overlap > 0:
    # update event_index to be sequential integers
    unique_events = sorted(catalog_out_df['event_index'].unique())
    min_value = min(unique_events)  # Start from the minimum value in the unique list
    mapping = {old_idx: i + min_value for i, old_idx in enumerate(unique_events)}
    catalog_out_df['event_index'] = catalog_out_df['event_index'].map(mapping)
    catalog_out_df=catalog_out_df.reset_index()
    pick_out_df['event_index'] = pick_out_df['event_index'].map(mapping).fillna(-1).astype(int)

    return  pick_out_df, catalog_out_df





def reindex_picks_and_events_fast(pick_df_list,catalog_df_list,harpa_config,overlap=None,verbose=1000):
    # Print the status message
    if verbose>2:
        print('Reindexing picks and events...')

    if overlap==None:
        overlap = harpa_config['window_lenth']-harpa_config['window_lenth_shift']
    # Initialize output DataFrames as None
    pick_out_df = None
    catalog_out_df = None
    n_repeat_event=0
    min_peak_pre_event=harpa_config['min_peak_pre_event']
    if harpa_config['remove_overlap_events']:
        remove_overlap_events=True
        threshold_time= harpa_config['threshold_time']
        threshold_xyz = harpa_config['threshold_xyz']
    else:
        remove_overlap_events=False
    # Iterate through each pair of pick_df and catalog_df in the provided lists
    pick_out_list=[]
    catalog_out_list = []
    for pick_df, catalog_df in tqdm(zip(pick_df_list, catalog_df_list), total=len(pick_df_list)):
        if not isinstance(pick_df, pd.DataFrame):
            continue  # Skip if pick_df is not a DataFrame
        if pick_df.empty:
            continue  # Skip if pick_df is empty

        if not isinstance(pick_out_df, pd.DataFrame):
            # If output DataFrames are not yet initialized, copy the first DataFrame pair
            pick_out_df = pick_df.copy()
            catalog_out_df = catalog_df.copy()
            current_index = pick_out_df.event_index.max() + 1
            pick_out_list.append(pick_df)
            catalog_out_list.append(catalog_df)
        else:
            # Create copies of the current DataFrame pair
            pick_df_next = pick_df.copy()
            catalog_df_next = catalog_df.copy()

            if overlap > 0:
                pass
            else:
                # If no overlap, simply reindex the next picks and catalog and concatenate them with the existing DataFrames
                pick_df_next.loc[pick_df_next['event_index'] > -1, 'event_index'] += current_index
                pick_out_list.append(pick_df_next)
                catalog_df_next.loc[:, 'event_index'] += current_index
                catalog_df_next.index += current_index
                catalog_out_list.append(catalog_df_next)
        # Update the current index to continue reindexing in the next iteration
        
        if len(catalog_out_df)==0:
            current_index=0
        else:
            current_index = catalog_out_df.event_index.max() + 1 
    pick_out_df = pd.concat(pick_out_list, ignore_index=True)
    catalog_out_df = pd.concat(catalog_out_list, ignore_index=True)
    # Print the final result indicating the number of removed repeated events and the number of unique events left
    if verbose>2:
        print(f'removed {n_repeat_event} repeated events, left {len(catalog_out_df)} unique events')
    #if overlap > 0:
    # update event_index to be sequential integers
    unique_events = sorted(catalog_out_df['event_index'].unique())
    min_value = min(unique_events)  # Start from the minimum value in the unique list
    mapping = {old_idx: i + min_value for i, old_idx in enumerate(unique_events)}
    catalog_out_df['event_index'] = catalog_out_df['event_index'].map(mapping)
    catalog_out_df=catalog_out_df.reset_index()
    pick_out_df['event_index'] = pick_out_df['event_index'].map(mapping).fillna(-1).astype(int)

    return  pick_out_df, catalog_out_df


def calc_mag(data, dist, min=-2, max=8):
    data=np.log10(data*1e2)
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (data - c0 - c3 * np.log10(np.maximum(dist, 0.1))) / c1 + 3.5

    #mag = np.log(np.mean(np.exp(mag_)))
    mag = np.mean(mag_)
    mag = np.clip(mag, min, max)
    #print(len(mag_))
    return mag

def compute_mag(pick_df,catalog_df,station_df):
    pick_df=pick_df.copy()
    catalog_df=catalog_df.copy()
    station_df=station_df.copy()
    station_df['station_id']=station_df['id']


    merged_df = pd.merge(pick_df, catalog_df, on='event_index', how='left')
    merged_df = pd.merge(merged_df, station_df, on='station_id', how='left', suffixes=('', '_station'))

    # Apply event_mask to calculate the 'distance'
    event_mask = merged_df['event_index'] > -1
    merged_df.loc[event_mask, 'distance'] = np.sqrt(
        (merged_df.loc[event_mask, 'x(km)'] - merged_df.loc[event_mask, 'x(km)_station']) ** 2 +
        (merged_df.loc[event_mask, 'y(km)'] - merged_df.loc[event_mask, 'y(km)_station']) ** 2 +
        (merged_df.loc[event_mask, 'z(km)'] - merged_df.loc[event_mask, 'z(km)_station']) ** 2
    )

    pick_df['distance'] = merged_df['distance'].values
    catalog_df['mag']=None
    for idx, catalog in tqdm(catalog_df.iterrows(),total=len(catalog_df)):
        event_index=catalog.event_index
        dist=pick_df['distance'][pick_df['event_index']==event_index].values
        #amp=np.log10(pick_df['amp'][pick_df['event_index']==event_index])
        #amp=np.log10(np.array(pick_df['amp'][pick_df['event_index']==event_index].values.tolist()))
        amp=pick_df['amp'][pick_df['event_index']==event_index].values
        mag=calc_mag(amp, dist, min=-2, max=8)
        catalog_df.at[idx, 'mag'] = mag
    return catalog_df



def check_and_clean_event_index(pick_df, catalog_df,correction=True,stations=None,column='event_index'):

    """
    Checks if all "column" values in pick_df are present in catalog_df and vice versa.
    If not, removes the corresponding rows in both pick_df and catalog_df where the event_index is missing in either DataFrame.

    Parameters:
    pick_df (pd.DataFrame): DataFrame containing the event_index column.
    catalog_df (pd.DataFrame): DataFrame containing the event_index column.

    Returns:
    pick_df (pd.DataFrame): The cleaned pick_df.
    catalog_df (pd.DataFrame): The cleaned catalog_df.
    boolean: Indicates whether all event_index values in pick_df and catalog_df match.
    """
    
    # Identify event_index values in pick_df that are not in catalog_df
    missing_in_catalog = pick_df.loc[~pick_df[column].isin(catalog_df[column]), column]
    
    # Identify event_index values in catalog_df that are not in pick_df
    missing_in_pick = catalog_df.loc[~catalog_df[column].isin(pick_df[column]), column]
    
    # If there are any missing event_index values in either DataFrame
    if not missing_in_catalog.empty or not missing_in_pick.empty:
        # Remove rows from pick_df where event_index is missing in catalog_df
        pick_df = pick_df[~pick_df[column].isin(missing_in_catalog)]
        
        # Remove rows from catalog_df where event_index is missing in pick_df
        catalog_df = catalog_df[~catalog_df[column].isin(missing_in_pick)] 
        
        all_in_catalog = False
    else:
        all_in_catalog = True
        
        
    if correction:
        station_df=stations.rename(columns={"id":"station_id"})
        merged_df = pd.merge(pick_df, catalog_df, on='event_index', how='left')
        merged_df = pd.merge(merged_df, station_df, on='station_id', how='left', suffixes=('', '_station'))

        # Apply event_mask to calculate the 'distance'
        event_mask = merged_df['event_index'] > -1
        merged_df.loc[event_mask, 'distance'] = np.sqrt(
            (merged_df.loc[event_mask, 'x(km)'] - merged_df.loc[event_mask, 'x(km)_station']) ** 2 +
            (merged_df.loc[event_mask, 'y(km)'] - merged_df.loc[event_mask, 'y(km)_station']) ** 2 +
            (merged_df.loc[event_mask, 'z(km)'] - merged_df.loc[event_mask, 'z(km)_station']) ** 2
        )
                # Initialize the 'new_time' column with None values
        merged_df['new_time'] = np.nan  # Using np.nan instead of None for numerical operations

        # Update 'new_time' where type is 'P'
        condition_P = merged_df['type'] == 'P'
        print()
        #merged_df.loc[condition_P, 'new_time'] = merged_df['time'][condition_P] + merged_df['distance'][condition_P] / 6
            
        merged_df.loc[condition_P, 'new_time'] = merged_df['time'][condition_P] + pd.to_timedelta(merged_df['distance'][condition_P] / 6, unit='s')

    

        # Update 'new_time' where type is 'S'
        condition_S = merged_df['type'] == 'S'
        #merged_df.loc[condition_S, 'new_time'] = merged_df['time'][condition_S] + merged_df['distance'][condition_S] / (6 / 1.75)
        merged_df.loc[condition_S, 'new_time'] = merged_df['time'][condition_S] + pd.to_timedelta(merged_df['distance'][condition_S] / (6/1.75), unit='s')

        pick_df['timestamp']=merged_df['new_time'].values
        #pick_df['timestamp'] = pick_df['timestamp'].apply(lambda x: np.datetime64(int(x.timestamp() * 1e3), 'ms'))

    
    return pick_df, catalog_df, all_in_catalog


def compress_data(pick_df,catalog_df,rate=1,shuffle=False):
    pick_df=pick_df.copy()
    catalog_df=catalog_df.copy()
    t_min_p=pick_df['timestamp'].min()
    t_min_c=catalog_df['time'].min()
    t_min=min(t_min_p,t_min_c)

    
    if shuffle:
        catalog_df['time_old']=catalog_df['time'].values
        catalog_df['time'] = np.random.permutation(catalog_df['time'].values)
        
        pick_df['timestamp_old']=pick_df['timestamp'].values
        for i, event in catalog_df.iterrows():
            event_index=event.event_index
            time_diff=(event['time']-event['time_old']).total_seconds()
            pick_df.loc[pick_df['event_index'] == event_index, 'timestamp'] += pd.Timedelta(seconds=time_diff)
    def scale(t):
        return pd.Timedelta(seconds=(t-t_min).total_seconds()*rate)+t_min
    catalog_df['time_old']=catalog_df['time'].values
    catalog_df['time']=catalog_df['time'].apply(scale)
    pick_df['timestamp_old']=pick_df['timestamp'].values
    for i, event in catalog_df.iterrows():
        event_index=event.event_index
        time_diff=(event['time']-event['time_old']).total_seconds()
        pick_df.loc[pick_df['event_index'] == event_index, 'timestamp'] += pd.Timedelta(seconds=time_diff)
    return pick_df, catalog_df






def remove_repeat_events(picks,catalog_df,time_threshold=5,space_threshold=10):
    t_pred=catalog_df['time'].apply(lambda x: pd.Timestamp(x)).to_numpy().astype(np.float64)/1e9
    evaluation_matrix_time = np.abs(t_pred[np.newaxis, :] - t_pred[:, np.newaxis]) < time_threshold

    dis_matrix=np.zeros((len(t_pred),len(t_pred)))

    dis_name=['x(km)','y(km)','z(km)']
    for name in dis_name:
        data=catalog_df[name].to_numpy()

        dis_matrix = dis_matrix+ (data[np.newaxis, :] - data[:, np.newaxis])**2
    dis_matrix=dis_matrix**0.5
    evaluation_matrix_space=dis_matrix<space_threshold
    evaluation_matrix=evaluation_matrix_space *evaluation_matrix_time


    upper_triangle_indices = np.triu_indices(evaluation_matrix.shape[0], k=1)

    true_coords = np.column_stack(upper_triangle_indices)[evaluation_matrix[upper_triangle_indices]]

    picks_=picks.copy()
    catalog_df_=catalog_df.copy()

    picks_['station_id_phase']=picks_.apply(lambda x: x.station_id + "_" + x.phase_type, axis=1).to_numpy()

    for x_i,y_i in true_coords:
        x_label=catalog_df_['event_index'].iloc[x_i]
        y_label=catalog_df_['event_index'].iloc[y_i]
        x_station_id_phase=picks_['station_id_phase'][picks_['event_index']==x_label]
        y_station_id_phase=picks_['station_id_phase'][picks_['event_index']==y_label]
        common_elements = set(x_station_id_phase) & set(y_station_id_phase)
        if common_elements:
            pass
        else:
            picks_.loc[picks_['event_index'] == y_label, 'event_index'] = x_label
            catalog_df_.drop(index=y_i, inplace=True)
            #catalog_df_.loc[catalog_df_['event_index'] == y_label, 'event_index'] = x_label
    catalog_df_=catalog_df_.sort_values('time')
    catalog_df_.reset_index(drop=True, inplace=True)
    new_event_idx_list=catalog_df_.index.values
    old_event_idx_list=catalog_df_.event_index.values
    catalog_df_['event_index']=new_event_idx_list

    replace_dict = dict(zip(old_event_idx_list, new_event_idx_list))
    picks_['event_index'] = picks_['event_index'].replace(replace_dict)
    print(f'removed {len(catalog_df)-len(catalog_df_)} repeated events, left {len(catalog_df_)} unique events')
    return picks_,catalog_df_




# From GaMMA
def DBSCAN_cluster(picks,stations,config):
    from datetime import datetime
    from sklearn.cluster import DBSCAN
    #print(picks)
    #print(picks["timestamp"].iloc[0])
    if type(picks["timestamp"].iloc[0]) is str:
        picks.loc[:, "timestamp"] = picks["timestamp"].apply(lambda x: datetime.fromisoformat(x))
    t = (
        picks["timestamp"]
        .apply(lambda x: x.tz_convert("UTC").timestamp() if x.tzinfo is not None else x.tz_localize("UTC").timestamp())
        .to_numpy()
    )

    timestamp0 = np.min(t)
    t = t - timestamp0
    data = t[:, np.newaxis]

    meta = stations.merge(picks["id"], how="right", on="id", validate="one_to_many")
    #locs = meta[config["dims"]].to_numpy()
    locs = meta[['x(km)', 'y(km)', 'z(km)']].to_numpy()
    
    vel = config["vel"] 
    if "prob" in picks.columns:
        pass
    else:
        picks["prob"]=1.0
    phase_weight = picks["prob"].to_numpy()[:, np.newaxis]
    
    if "dbscan_eps" in config:
        eps=config["dbscan_eps"]
    else:
        eps=estimate_eps(stations, config["vel"]["P"]) 
    
    
    if "dbscan_min_samples" in config:
        dbscan_min_samples=config["dbscan_min_samples"]
    else:
        dbscan_min_samples=3
    db = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit(
        np.hstack([data[:, 0:1], locs[:, :2] / np.average(vel["P"])]),
        sample_weight=np.squeeze(phase_weight),
    )

    labels = db.labels_
    unique_labels = set(labels)
    unique_labels = unique_labels.difference([-1])
    picks['labels']=labels
    return picks,unique_labels



# from GaMMA.utils
def estimate_eps(stations, vp, sigma=3.0):
    from scipy.sparse.csgraph import minimum_spanning_tree
    X = stations[["x(km)", "y(km)", "z(km)"]].values
    D = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=-1))
    Tcsr = minimum_spanning_tree(D).toarray()

    # Tcsr = Tcsr[Tcsr > 0]
    # # mean = np.median(Tcsr)
    # mean = np.mean(Tcsr)
    # std = np.std(Tcsr)
    # eps = (mean + sigma * std) / vp

    eps = np.max(Tcsr) / vp * 1.5

    return eps