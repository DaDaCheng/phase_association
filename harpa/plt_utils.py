
import numpy as np
from obspy import UTCDateTime

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import pandas as pd



colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown','pink']
colors=np.array(colors)


# colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown',
#         'pink', 'lime', 'olive', 'navy', 'teal', 'maroon', 'gold', 
#         'silver', 'chocolate', 'indigo', 'violet', 'turquoise', 
#         'crimson', 'darkgreen', 'darkblue', 'darkred', 'darkorange', 
#         'darkviolet', 'darkturquoise', 'darkgoldenrod', 'lightcoral', 
#         'lightseagreen','r', 'g', 'b', 'c', 'm', 'y', 'k'])
def plot_assosication(pick_df,event_idx_key,station_df,catalog,plot_y_as='x',xlim=None,plot_max=100):
    station_dict = {station: (x, y) for station, x, y in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"])}
    event_idx_list=pick_df[event_idx_key].values.tolist()
    #print(event_idx_list)
    #print(colors[event_idx_list])
    #type_list=pick_df['type'].values

    time_idx_list=pick_df['timestamp'].values
    time_idx_list=list(map(lambda x: UTCDateTime(str(x)), time_idx_list))

    start_time=min(time_idx_list)

    d_time_idx_list=list(map(lambda x: x-start_time, time_idx_list))

    if plot_y_as=='x':
        pick_x_list = [station_dict[pick_id][0] for pick_id in pick_df['station_id'].values]
        pick_x_list =np.array(pick_x_list)
        y_axis=pick_x_list
    if plot_y_as=='y':    
        pick_y_list = [station_dict[pick_id][1] for pick_id in pick_df['station_id'].values]
        pick_y_list =np.array(pick_y_list)
        y_axis=pick_y_list
    

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    event_idx_list=np.array(event_idx_list)
    plot_mask=event_idx_list<plot_max
    d_time_idx_list=np.array(d_time_idx_list)
    y_axis=np.array(y_axis)
    #print(event_idx_list[plot_mask],len(event_idx_list[plot_mask]),max(event_idx_list[plot_mask]))
    # print(d_time_idx_list)
    # print(plot_mask)
    # print(y_axis[plot_mask])
    # print(event_idx_list)
    # print(colors[event_idx_list[plot_mask]])
    ax.scatter(d_time_idx_list[plot_mask], y_axis[plot_mask], s=20, color=colors[event_idx_list[plot_mask]])


    #current_xlim = plt.gca().get_xlim()
    current_ylim = plt.gca().get_ylim()
    
    for event in catalog.iloc:
        time=UTCDateTime(event.time)
        #x=time_min+int((time-time_min)/interval*interval_new)
        x=time-start_time
        if event.event_index==plot_max:
            break
        ax.plot([x,x],current_ylim, color=colors[int(event.event_index)])

    if xlim is not None:
        ax.set_xlim(xlim)   
    return ax 
        
        
def plot_line(pick_df,catalog,station_df,event_idx_key='event_idx_new',plot_max=100,xlim=None,phases=['P','S']):
    fig = plt.figure(figsize=(12, 3))
    station_dict = {station: (x, y, z) for station, x, y, z in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"],station_df["z(km)"])}
    ax = fig.add_subplot(111)
    start_time=UTCDateTime(str(pick_df.timestamp.values.min()))
    markers=[0,2]
    for idx, event in catalog.iterrows():
        if event.event_index==plot_max:
            break
        for p,phase in enumerate(phases):
            event_x,event_y,event_z=event['x(km)'],event['y(km)'],event['z(km)']
            pick_index_list=pick_df.index[(pick_df[event_idx_key]==event.event_index) &(pick_df['type']==phase) ]        
            pick_time=pick_df.timestamp[pick_index_list].values
            if len(pick_time)>0:
                
                pick_time=list(map(lambda x: UTCDateTime(str(x)), pick_time))
                pick_time=list(map(lambda x: x-start_time, pick_time))
                if xlim is not None:
                    if np.array(pick_time).max()>xlim[1]:
                        continue
                #print(pick_time)
                station_x= np.array([station_dict[sid][0] for sid in pick_df.station_id[pick_index_list].values])
                station_y= np.array([station_dict[sid][1] for sid in pick_df.station_id[pick_index_list].values])
                station_z= np.array([station_dict[sid][2] for sid in pick_df.station_id[pick_index_list].values])
                dist=((event_x-station_x)**2+(event_y-station_y)**2+(event_z-station_z)**2)**0.5
                ax.scatter(pick_time,dist,s=20,color=colors[event.event_index],marker=markers[p])
            ax.text(event.time-start_time,0,event.event_index, fontsize=12, ha='right', va='bottom')
                        
    if xlim is not None:
        ax.set_xlim(xlim)   

    return ax

def plot_assosication_time_window(pick_df,event_idx_key,station_df,catalog,plot_y_as='x',time_start=None,time_end=None,legend=False):
    pick_df=pick_df.copy()
    station_df=station_df.copy()
    catalog=catalog.copy()
    if isinstance(catalog['time'].iloc[0],str):
        catalog['time']= catalog['time'].apply(lambda x: pd.Timestamp(x))
    
    if 'id' in station_df.columns:
        station_df.rename(columns={"id": "station_id"}, inplace=True)
        
    if 'id' in pick_df.columns:
        pick_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'phase_time' in pick_df.columns:
        pick_df.rename(columns={"phase_time": "timestamp"}, inplace=True)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)
    station_dict = {station: (x, y) for station, x, y in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"])}
    if time_start is not None:
        pick_df=pick_df[pick_df['timestamp']>=pd.Timestamp(time_start)]
        catalog=catalog[catalog['time']>=pd.Timestamp(time_start)]
    
    if time_end is not None:
        pick_df=pick_df[pick_df['timestamp']<=pd.Timestamp(time_end)]
        catalog=catalog[catalog['time']<=pd.Timestamp(time_end)]

    event_idx_list=pick_df[event_idx_key].values.tolist()
    event_idx_list=np.array(event_idx_list)
    event_idx_mask=event_idx_list>-1
    time_idx_list=pick_df['timestamp'].values


    if plot_y_as=='x':
        pick_x_list = [station_dict[pick_id][0] for pick_id in pick_df['station_id'].values]
        pick_x_list =np.array(pick_x_list)
        y_axis=pick_x_list
    if plot_y_as=='y':    
        pick_y_list = [station_dict[pick_id][1] for pick_id in pick_df['station_id'].values]
        pick_y_list =np.array(pick_y_list)
        y_axis=pick_y_list
    

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    
    time_idx_list=np.array(time_idx_list)
    y_axis=np.array(y_axis)   
    print(len(time_idx_list)) 
    print(event_idx_mask.sum())
    if event_idx_mask.sum()>0:
        ax.scatter(time_idx_list[event_idx_mask], y_axis[event_idx_mask], s=20, color=colors[(event_idx_list[event_idx_mask]%10)])
    if (~event_idx_mask).sum()>0:
        ax.scatter(time_idx_list[~event_idx_mask], y_axis[~event_idx_mask], s=20, c='black')

    #current_xlim = plt.gca().get_xlim()
    current_ylim = plt.gca().get_ylim()
    
    
    
    for event in catalog.iloc:
        #time=UTCDateTime(event.time)
        #x=time-start_time
        x=event.time
        ax.plot([x,x],current_ylim, color=colors[int(event.event_index)%10])
        
    ax.set_xlim([catalog.time.min()-pd.Timedelta(seconds=1),pick_df['timestamp'].max()])


    return ax 





def plot_assosication_line_window(pick_df,event_idx_key,station_df,catalog,plot_y_as='x',time_start=None,time_end=None,phases=['P','S'],legend=True):
    pick_df=pick_df.copy()
    station_df=station_df.copy()
    catalog=catalog.copy()
    if isinstance(catalog['time'].iloc[0],UTCDateTime):
        catalog['time'] = catalog['time'].apply(lambda x: x.datetime)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)
    pick_df['type'] = pick_df['type'].replace('p', 'P')
    pick_df['type'] = pick_df['type'].replace('s', 'S')
    if 'id' in station_df.columns:
        station_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'id' in pick_df.columns:
        pick_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'phase_time' in pick_df.columns:
        pick_df.rename(columns={"phase_time": "timestamp"}, inplace=True)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)    
        
    station_dict = {station: (x, y, z) for station, x, y,z in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"],station_df["z(km)"])}
    
    if isinstance(catalog['time'].iloc[0],str):
        catalog['time']= catalog['time'].apply(lambda x: pd.Timestamp(x))
        
    if time_start is not None:
        pick_df=pick_df[pick_df['timestamp']>=pd.Timestamp(time_start)]
        catalog=catalog[catalog['time']>=pd.Timestamp(time_start)]
    
    if time_end is not None:
        pick_df=pick_df[pick_df['timestamp']<=pd.Timestamp(time_end)]
        catalog=catalog[catalog['time']<=pd.Timestamp(time_end)]
    

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    markers=[0,2]
    for idx, event in catalog.iterrows():
        for p,phase in enumerate(phases):
            event_x,event_y,event_z=event['x(km)'],event['y(km)'],event['z(km)']
            pick_index_list=pick_df.index[(pick_df[event_idx_key]==event.event_index) &(pick_df['type']==phase) ]        
            pick_time=pick_df.timestamp[pick_index_list].values
            if len(pick_time)>0:
                
                #pick_time=list(map(lambda x: UTCDateTime(str(x)), pick_time))
                #pick_time=list(map(lambda x: x-start_time, pick_time))

                #print(pick_time)
                station_x= np.array([station_dict[sid][0] for sid in pick_df.station_id[pick_index_list].values])
                station_y= np.array([station_dict[sid][1] for sid in pick_df.station_id[pick_index_list].values])
                station_z= np.array([station_dict[sid][2] for sid in pick_df.station_id[pick_index_list].values])
                dist=((event_x-station_x)**2+(event_y-station_y)**2+(event_z-station_z)**2)**0.5
                ax.scatter(pick_time,dist,s=20,color=colors[event.event_index%10],marker=markers[p])
            ax.text(event.time,0,event.event_index, fontsize=12, ha='left', va='top',color=colors[event.event_index%10])
            ax.scatter(event.time,0,marker='o',s=40, color=colors[event.event_index%10])
    for p,phase in enumerate(phases):
        ax.scatter([],[],s=20,c='black',marker=markers[p],label=phase)
        
    if legend:        
        ax.legend()
    ax.set_xlim([catalog.time.min()-pd.Timedelta(seconds=1),pick_df['timestamp'].max()])   
    return ax 



import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
def plot3D(V,X,vmin=None,vmax=None,colorbar=True,show=True):
    V_re=V
    ddata=V_re
    grid_size=len(X)
    Y, X, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), -np.arange(grid_size))
    Vd=ddata.copy()
    r=1.1
    layout=dict()
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=Vd.flatten(),
        isomin=vmin,
        isomax=vmax,
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=100,
        colorscale='rdylbu',# needs to be a large number for good volume rendering
        reversescale=True,
        showscale=colorbar
        ),layout=layout)
    # fig.add_trace(go.Scatter3d(x=loc_source[:,0]*(grid_size-1), y=loc_source[:,1]*(grid_size-1),z=-loc_source[:,2]*(grid_size-1), mode='markers',marker=dict(color='green',symbol='diamond',size=5)))

    # fig.add_trace(go.Scatter3d(x=loc_earthquake_coord[:,0]*(grid_size-1), y=loc_earthquake_coord[:,1]*(grid_size-1),z=-loc_earthquake_coord[:,2]*(grid_size-1), mode='markers',marker=dict(color='black',symbol='circle',size=5)))

    # fig.add_trace(go.Scatter3d(x=lowest_loss_loc_earthquake_re[:,0]*(grid_size-1), y=lowest_loss_loc_earthquake_re[:,1]*(grid_size-1),z=-lowest_loss_loc_earthquake_re[:,2]*(grid_size-1), mode='markers',marker=dict(color='black',symbol='cross',size=8)))

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()

    linewidth=6
    linecolor='black'
    fig.add_trace(go.Scatter3d(x=[xmin, xmax,xmax,xmin,xmin], y=[ymin, ymin,ymax,ymax,ymin], z=[zmin,zmin,zmin,zmin,zmin],mode='lines',line=dict(color=linecolor, width=linewidth)))
    fig.add_trace(go.Scatter3d(x=[xmin, xmax,xmax,xmin,xmin], y=[ymin, ymin,ymax,ymax,ymin], z=[zmax,zmax,zmax,zmax,zmax],mode='lines',line=dict(color=linecolor, width=linewidth)))
    fig.add_trace(go.Scatter3d(x=[xmin, xmin], y=[ymin, ymin], z=[zmin,zmax],mode='lines',line=dict(color=linecolor, width=linewidth)))
    fig.add_trace(go.Scatter3d(x=[xmin, xmin], y=[ymax, ymax], z=[zmin,zmax],mode='lines',line=dict(color=linecolor, width=linewidth)))
    fig.add_trace(go.Scatter3d(x=[xmax, xmax], y=[ymin, ymin], z=[zmin,zmax],mode='lines',line=dict(color=linecolor, width=linewidth)))
    fig.add_trace(go.Scatter3d(x=[xmax, xmax], y=[ymax, ymax], z=[zmin,zmax],mode='lines',line=dict(color=linecolor, width=linewidth)))

    fig.update_traces(
        showlegend=False
    )


    if colorbar:
        fig.data[0].colorbar.x=0.93
        fig.data[0].colorbar.y=0.5
        fig.data[0].colorbar.thickness=20
        fig.data[0].colorbar.len=0.7
        fig.data[0].colorbar.title='Wave speed (km/s)'
        fig.data[0].colorbar.title.side='right'
        
        
        
        fig.data[0].colorbar.titlefont=dict(
                family='Helvetica',
                color='black',
                size=20
            )
        fig.data[0].colorbar.tickfont=dict(
                family='Helvetica',
                color='black',
                size=20
            )


    fig.update_layout(width=500, height=450,margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)")
    )

    camera = dict(
        eye=dict(x=0.9764760583095522*r, y=1.7327485214077152*r, z=0.6319680639439241*r),
        center=dict(x=0, y=0, z=-0.1)
    )
    fig.update_layout(scene_camera=camera)
    nx,ny,nz=grid_size,grid_size,grid_size

    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=[0,int(nx/4),int(nx/4*2),int(nx/4*3),nx],
                ticktext=[0, 25,50,75,100]
            ),
            xaxis_title="X (km)"
        )
    )

    
    fig.update_layout(
        scene=dict(
            yaxis=dict(
                tickvals=[0,int(nx/4),int(nx/4*2),int(nx/4*3),nx],
                ticktext=[0, 25,50,75,100]
            ),
            yaxis_title="Y (km)"
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    # Change the z-axis tick labels
    fig.update_layout(
        scene=dict(
            zaxis=dict(
                tickvals=[0,-int(nz/4),-int(nz/4*2),-int(nz/4*3),-nz],
                ticktext=[0,-25, -50,-75,-100]
            ),
            zaxis_title="Z (km)"
        )
    )

    fig.update_layout(
        font=dict(
            family='Helvetica',
            color='black',
            size=16
        )
    )

    if show:
        fig.show()
    return fig




def plot_assosication_window(pick_df,event_idx_key,station_df,catalog,plot_y_as='x',time_start=None,time_end=None,phases=['P'],legend=True):
    plt.rcParams.update({'font.size': 15})
    
    
    colors = ['#8B0000', '#FFA07A', '#483D8B', '#7B68EE', '#2F4F4F', 
          '#20B2AA', '#4B0082', '#9370DB', '#8B4513', '#DAA520', 
          '#2E8B57', '#66CDAA', '#556B2F', '#9ACD32', '#4682B4', 
          '#87CEFA']

    colors=np.array(colors)
    
    pick_df=pick_df.copy()
    station_df=station_df.copy()
    catalog=catalog.copy()
    pick_df1=pick_df.copy()
    station_df1=station_df.copy()
    catalog1=catalog.copy()
    s=14
    if isinstance(catalog['time'].iloc[0],UTCDateTime):
        catalog['time'] = catalog['time'].apply(lambda x: x.datetime)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)
    pick_df['type'] = pick_df['type'].replace('p', 'P')
    pick_df['type'] = pick_df['type'].replace('s', 'S')
    if 'id' in station_df.columns:
        station_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'id' in pick_df.columns:
        pick_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'phase_time' in pick_df.columns:
        pick_df.rename(columns={"phase_time": "timestamp"}, inplace=True)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)    
        
    station_dict = {station: (x, y, z) for station, x, y,z in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"],station_df["z(km)"])}
    
    if isinstance(catalog['time'].iloc[0],str):
        catalog['time']= catalog['time'].apply(lambda x: pd.Timestamp(x))
        
    if time_start is not None:
        pick_df=pick_df[pick_df['timestamp']>=pd.Timestamp(time_start)]
        catalog=catalog[catalog['time']>=pd.Timestamp(time_start)]
    
    if time_end is not None:
        pick_df=pick_df[pick_df['timestamp']<=pd.Timestamp(time_end)]
        catalog=catalog[catalog['time']<=pd.Timestamp(time_end)]
    

    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 3))
    ax = axes[0]
    markers=['o','o']
    for idx, event in catalog.iterrows():
        for p,phase in enumerate(phases):
            event_x,event_y,event_z=event['x(km)'],event['y(km)'],event['z(km)']
            pick_index_list=pick_df.index[(pick_df[event_idx_key]==event.event_index) &(pick_df['type']==phase) ]        
            pick_time=pick_df.timestamp[pick_index_list].values
            ax.scatter(event.time,0,marker='D',s=40, color=colors[event.event_index%10])
            #ax.plot([event.time,event.time+pd.Timedelta(seconds=50)],[0,50*v_background],'--',color=colors[event.event_index%10] )
            if len(pick_time)>0:
                
                #pick_time=list(map(lambda x: UTCDateTime(str(x)), pick_time))
                #pick_time=list(map(lambda x: x-start_time, pick_time))

                #print(pick_time)
                station_x= np.array([station_dict[sid][0] for sid in pick_df.station_id[pick_index_list].values])
                station_y= np.array([station_dict[sid][1] for sid in pick_df.station_id[pick_index_list].values])
                station_z= np.array([station_dict[sid][2] for sid in pick_df.station_id[pick_index_list].values])
                dist=((event_x-station_x)**2+(event_y-station_y)**2+(event_z-station_z)**2)**0.5
                ax.scatter(pick_time,dist,s=s,color=colors[event.event_index%10],marker=markers[p])
            #ax.text(event.time,0,event.event_index, fontsize=12, ha='left', va='top',color=colors[event.event_index%10])
            
    for p,phase in enumerate(phases):
        ax.scatter([],[],s=s,c='black',marker=markers[p],label=phase)
    ax.set_xlim([catalog.time.min()-pd.Timedelta(seconds=4),pick_df['timestamp'].max()+pd.Timedelta(seconds=5)])
    current_ylim = ax.get_ylim()
    ax.set_ylim([0-5,current_ylim[1]])
    plt.subplots_adjust(hspace=0.02)
    axes[0].set_xticklabels([])
    ax.set_ylabel('Dist (km)')
    
    pick_df=pick_df1.copy()
    station_df=station_df1.copy()
    catalog=catalog1.copy()
    if isinstance(catalog['time'].iloc[0],str):
        catalog['time']= catalog['time'].apply(lambda x: pd.Timestamp(x))
    
    if 'id' in station_df.columns:
        station_df.rename(columns={"id": "station_id"}, inplace=True)
        
    if 'id' in pick_df.columns:
        pick_df.rename(columns={"id": "station_id"}, inplace=True)
    if 'phase_time' in pick_df.columns:
        pick_df.rename(columns={"phase_time": "timestamp"}, inplace=True)
    if 'phase_type' in pick_df.columns:
        pick_df.rename(columns={"phase_type": "type"}, inplace=True)
    station_dict = {station: (x, y) for station, x, y in zip(station_df["station_id"], station_df["x(km)"], station_df["y(km)"])}
    if time_start is not None:
        pick_df=pick_df[pick_df['timestamp']>=pd.Timestamp(time_start)]
        catalog=catalog[catalog['time']>=pd.Timestamp(time_start)]
    
    if time_end is not None:
        pick_df=pick_df[pick_df['timestamp']<=pd.Timestamp(time_end)]
        catalog=catalog[catalog['time']<=pd.Timestamp(time_end)]

    event_idx_list=pick_df[event_idx_key].values.tolist()
    event_idx_list=np.array(event_idx_list)
    event_idx_mask=event_idx_list>-1
    time_idx_list=pick_df['timestamp'].values


    if plot_y_as=='x':
        pick_x_list = [station_dict[pick_id][0] for pick_id in pick_df['station_id'].values]
        pick_x_list =np.array(pick_x_list)
        y_axis=pick_x_list
    if plot_y_as=='y':    
        pick_y_list = [station_dict[pick_id][1] for pick_id in pick_df['station_id'].values]
        pick_y_list =np.array(pick_y_list)
        y_axis=pick_y_list
    


    ax = axes[1]
    
    
    time_idx_list=np.array(time_idx_list)
    y_axis=np.array(y_axis)   
    print(len(time_idx_list)) 
    print(event_idx_mask.sum())
    
        

    
    if event_idx_mask.sum()>0:
        ax.scatter(time_idx_list[event_idx_mask], y_axis[event_idx_mask], s=s, color=colors[(event_idx_list[event_idx_mask]%10)])
    if (~event_idx_mask).sum()>0:
        ax.scatter(time_idx_list[~event_idx_mask], y_axis[~event_idx_mask], s=s, c='black')

    #current_xlim = plt.gca().get_xlim()
    current_ylim = plt.gca().get_ylim()
    
    
    for event in catalog.iloc:
        #time=UTCDateTime(event.time)
        #x=time-start_time
        x=event.time
        ax.plot([x,x],[current_ylim[0]-10,current_ylim[1]+10], color=colors[int(event.event_index)%10])
    ax.set_ylim([current_ylim[0],95])
    ax.set_xlim(catalog.time.min()-pd.Timedelta(seconds=4),pick_df['timestamp'].max()+pd.Timedelta(seconds=5))
    
    
    ax.set_ylabel('x (km)')
    
    
    # ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
    x_data = axes[1].get_xticks()
    x_data_seconds = np.array(x_data)*60*24*60
    ax.set_xticks(x_data)
    ax.set_xticklabels([f'{int(x):d}' for x in x_data_seconds])
    ax.set_xlabel('Seconds')
    return axes 