## read picks
import pandas as pd
import os
from pyproj import Proj
from gamma.utils import estimate_eps
def load_ridgecrest(time_start=None,time_end=None):
    
    #time_start='2019-07-04 17:00:00.000000',time_end='2019-07-05 00:00:00.000000'
    data_path = lambda x: os.path.join(f"./Data/Ridgecrest/test_data", x)
    region='ridgecrest'
    station_csv = data_path("stations.csv")
    picks_csv = data_path("picks.csv")
    picks = pd.read_csv(picks_csv, parse_dates=["phase_time"])
    picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob", "phase_amplitude": "amp"}, inplace=True)
    #print("Pick format:", picks.iloc[:10])
    
    if time_start is not None:
        picks=picks[picks['timestamp']>=pd.Timestamp(time_start)]
        
    
    if time_end is not None:
        picks=picks[picks['timestamp']<=pd.Timestamp(time_end)]
    
    ## read stations
    stations = pd.read_csv(station_csv)
    stations.rename(columns={"station_id": "id"}, inplace=True)
    #print("Station format:", stations.iloc[:10])

    ## Automatic region; you can also specify a region
    x0 = stations["longitude"].median()
    y0 = stations["latitude"].median()
    xmin = stations["longitude"].min()
    xmax = stations["longitude"].max()
    ymin = stations["latitude"].min()
    ymax = stations["latitude"].max()
    config = {}
    config["center"] = (x0, y0)
    config["xlim_degree"] = (2 * xmin - x0, 2 * xmax - x0)
    config["ylim_degree"] = (2 * ymin - y0, 2 * ymax - y0)

    ## projection to km
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x/1e3)
    
        
        # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["x(km)"] = proj(longitude=config["xlim_degree"], latitude=[config["center"][1]] * 2)[0]
    config["y(km)"] = proj(longitude=[config["center"][0]] * 2, latitude=config["ylim_degree"])[1]
    if region == "ridgecrest":
        config["z(km)"] = (0, 20)
    elif region == "chile":
        config["z(km)"] = (0, 250)
    else:
        print("Please specify z(km) for your region")
        raise NotImplementedError
    #picks,stations,_=check_and_clean(picks, stations,column='id')
    return  picks, stations, config,proj




def load_ridgecrest_full(time_start=None, time_end=None,drop_index=True):
    data_path = lambda x: os.path.join(f"../Data/Ridgecrest", x)
    region='ridgecrest'
    #station_csv = data_path("test_data/stations.csv")
    station_csv = data_path("stations.csv")
    picks_csv = data_path("picks_gamma.csv")
    catalog_csv= data_path("catalog_gamma.csv")
    picks  =  pd.read_csv(picks_csv, delimiter='\t')
    catalog = pd.read_csv(catalog_csv, delimiter="\t")
    #print(picks)
    #id	timestamp	type	prob	amp	event_idx	prob_gamma
    picks.rename(columns={"station_id": "id", "timestamp": "timestamp", "type": "type", "prob": "prob", "phase_amplitude": "amp", "event_idx": "event_idx", "prob_gamma": "prob_gamma" }, inplace=True)
    #print("Pick format:", picks.iloc[:10])
    if drop_index:
        picks=picks.drop('event_idx',axis=1)
    else:
        picks=picks.rename(columns={"event_idx": "event_index"})
    picks["timestamp"]=picks["timestamp"].apply(lambda x: pd.Timestamp(x))

    if time_start is not None:
        picks=picks[picks['timestamp']>=pd.Timestamp(time_start)]

    if time_end is not None:
        picks=picks[picks['timestamp']<=pd.Timestamp(time_end)]

    ## read stations
    stations = pd.read_csv(station_csv, delimiter='\t')
    stations.rename(columns={"station": "id",'elevation(m)':'elevation_m'}, inplace=True)
    #print("Station format:", stations.iloc[:10])

    ## Automatic region; you can also specify a region
    x0 = stations["longitude"].median()
    y0 = stations["latitude"].median()
    xmin = stations["longitude"].min()
    xmax = stations["longitude"].max()
    ymin = stations["latitude"].min()
    ymax = stations["latitude"].max()
    config = {}
    config["center"] = (x0, y0)
    config["xlim_degree"] = (2 * xmin - x0, 2 * xmax - x0)
    config["ylim_degree"] = (2 * ymin - y0, 2 * ymax - y0)

    ## projection to km
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x/1e3)

        
        # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["x(km)"] = proj(longitude=config["xlim_degree"], latitude=[config["center"][1]] * 2)[0]
    config["y(km)"] = proj(longitude=[config["center"][0]] * 2, latitude=config["ylim_degree"])[1]
    if region == "ridgecrest":
        config["z(km)"] = (0, 20)
    elif region == "chile":
        config["z(km)"] = (0, 250)
    else:
        print("Please specify z(km) for your region")
        raise NotImplementedError
    #picks,stations,_=check_and_clean(picks, stations,column='id')
    return  picks, stations, config, proj,catalog




def load_chile(time_start=None,time_end=None):
    data_path = lambda x: os.path.join(f"../Data/Chile/test_data", x)
    region='chile'
    station_csv = data_path("stations.csv")
    picks_csv = data_path("picks.csv")
    picks = pd.read_csv(picks_csv, parse_dates=["phase_time"])
    picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob", "phase_amplitude": "amp"}, inplace=True)
    #print("Pick format:", picks.iloc[:10])
    
    if time_start is not None:
        picks=picks[picks['timestamp']>=pd.Timestamp(time_start)]
        
    
    if time_end is not None:
        picks=picks[picks['timestamp']<=pd.Timestamp(time_end)]
    
    ## read stations
    stations = pd.read_csv(station_csv)
    stations.rename(columns={"station_id": "id"}, inplace=True)
    #print("Station format:", stations.iloc[:10])

    ## Automatic region; you can also specify a region
    x0 = stations["longitude"].median()
    y0 = stations["latitude"].median()
    xmin = stations["longitude"].min()
    xmax = stations["longitude"].max()
    ymin = stations["latitude"].min()
    ymax = stations["latitude"].max()
    config = {}
    config["center"] = (x0, y0)
    config["xlim_degree"] = (2 * xmin - x0, 2 * xmax - x0)
    config["ylim_degree"] = (2 * ymin - y0, 2 * ymax - y0)
    
    ### setting GMMA configs
    config["use_dbscan"] = True
    if region == "chile":
        config["use_amplitude"] = False
    else:
        config["use_amplitude"] = True
    config["method"] = "BGMM"  
    if config["method"] == "BGMM": ## BayesianGaussianMixture
        config["oversample_factor"] = 5
    if config["method"] == "GMM": ## GaussianMixture
        config["oversample_factor"] = 1


    # DBSCAN: 
    ##!!Truncate the picks into segments: change the dbscan_eps to balance speed and event splitting. A larger eps prevent spliting events but can take longer time in the preprocessing step.



    ## projection to km
    proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x/1e3)
    
        
        # earthquake location
    config["vel"] = {"p": 7.0, "s": 7.0 / 1.75}
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["x(km)"] = proj(longitude=config["xlim_degree"], latitude=[config["center"][1]] * 2)[0]
    config["y(km)"] = proj(longitude=[config["center"][0]] * 2, latitude=config["ylim_degree"])[1]
    
    # config["x(km)"] = [-200,300]
    # config["y(km)"] = [-400,450]
    
    
    if region == "ridgecrest":
        config["z(km)"] = (0, 20)
    elif region == "chile":
        config["z(km)"] = (0, 250)
    else:
        print("Please specify z(km) for your region")
        raise NotImplementedError
    
    
    velocity_model = pd.read_csv(data_path("iasp91.csv"), names=["zz", "rho", "vp", "vs"])
    velocity_model = velocity_model[velocity_model["zz"] <= config["z(km)"][1]]
    vel = {"z": velocity_model["zz"].values, "p": velocity_model["vp"].values, "s": velocity_model["vs"].values}
    h = 1.0
    config["eikonal"] = {"vel": vel, "h": h, "xlim": config["x(km)"], "ylim": config["y(km)"], "zlim": config["z(km)"]}
    
    config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"]) 
    config["dbscan_min_samples"] = 3
    
    config["bfgs_bounds"] = (
    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
    (0, config["z(km)"][1] + 1),  # z
    (None, None),  # t
    )

    
    
    config["min_picks_per_eq"] = 5
    config["min_p_picks_per_eq"] = 0
    config["min_s_picks_per_eq"] = 0
    config["max_sigma11"] = 3.0 # second
    config["max_sigma22"] = 1.0 # log10(m/s)
    config["max_sigma12"] = 1.0 # covariance
    
    
    
    return  picks, stations, config,proj,velocity_model