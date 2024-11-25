# <span style="font-family: sans-serif;">Harpa</span>: High-Rate Phase Association with Travel Time Neural Fields


## About

This is the source code for paper _Harpa: High-Rate Phase Association with Travel Time Neural Fields_.


## Installation 
Simply run:
```
pip install -q git+https://github.com/DaDaCheng/phase_association.git
```
It requires `torch`,`obspy`, `pyproj`, `pandas`, `POT` for constant wave speed model, and `skfmm` from unknown wave speed model.


## Quick start in SeisBench and colab demo

HARPA seamlessly integrates with workflow in [SeisBench](https://github.com/seisbench/seisbench).


| Examples                                         |  |
|--------------------------------------------------|---|
| 2019 Ridgcrest earthquake                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16lE4eu0SM3xQVb-686XL-0evPXOTPzwC#scrollTo=ZUFnMmLlTHec) |
| 2014 Chile earthquake                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o7S8n2LtJChraLoHqoNykQ_m9aqWifG-?usp=sharing) |
| Unknown wave speed model and neural fields            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lAciDACeV24vHQFVjWraQE8KOb81ATEd?usp=sharing) |

## Usage  
To perform the association, you need to run 
```
from harpa import association
pick_df_out, catalog_df=association(pick_df,station_df,config)
```
with `pick_df` as the DataFrame containing information about seismic picks with the structure as

| id        | timestamp               | prob     | type |
|-----------|-------------------------|----------|------|
| CX.MNMCX. | 2014-04-01 00:08:45.410 | 0.113388 | p    |
| CX.MNMCX. | 2014-04-01 00:10:56.840 | 0.908900 | p    |
| CX.MNMCX. | 2014-04-01 00:13:01.930 | 0.795285 | p    |
| CX.MNMCX. | 2014-04-01 00:13:36.950 | 0.130214 | s    |


where `prob` is optional. And `station_df` contains information about seismic stations with the structure as

| id       | x(km)     | y(km)     | z(km)   |
|----------|-----------|-----------|---------|
| CX.PB01. | 449.359240| 7672.990624| -0.900  |
| CX.PB02. | 407.073618| 7642.202105| -1.015  |
| CX.PB03. | 422.289208| 7561.616351| -1.460  |
| CX.PB04. | 381.654501| 7529.786448| -1.520  |


## configs and notes
Mandatory configurations: 
| config    | description               | example  |
|-----------|-------------------------|----------|
| `x(km)` | x range for the searching space  | `[0,100]` |
| `y(km)` | y range for the searching space  | `[0,100]` |
| `z(km)` | y range for the searching space  | `[0,100]` |
|`vel`| homogenous wave speed mode  |`{"P": 7, "S": 4.0}`|



Other configurations: 
| config    | description               | default  |
|-----------|-------------------------|----------|
| `lr`|learning rate |  `0.1` |
| `noise` | $\epsilon$ in SGLD | `1e-3` |
| `lr_decay` | learning rate and noise decay | `0.1` |
| `epoch_before_decay` | number of epoch before decay | `1000` |
| `epoch_after_decay` | number of epoch after decay | `1000` |
| `LSA`| use linear sum assignment in computing loss  |`True`|
| `wasserstein_p`| the order of the Wasserstein distance  |`2`|
| `P_phase`| data contains P-phase | `True`|
| `S_phase`| data contains S-phase | `False`|
|`noisy_pick`| data contains missing or spurious picks |`True`|
|`min_peak_pre_event`| filter events by the minimum number of picks for each event|`16`|
|`min_peak_pre_event_p`| filter events by the minimum number of p picks for each event|`0`|
|`min_peak_pre_event_s`| filter events by the s picks for each event|`0`|
|`max_time_residual`|above events were counted only when the time residual was below this threshold. |`2`|
|`denoise_rate`| filter events by the ratio of nearest station|`0`|
|`DBSCAN`| using DBSCAN to divide the data into different slides |`True`|
|`remove_overlap_events`| remove repeated events if the slides overlap|`False`|
| `neural_field`| `True` for unknown wave speed model,<br> `False` for fixed wave speed model |`False` |
|`wave_speed_model_hidden_dim`|hidden dimension of the wave speed model if `neural_field` is true  ||
|`optimize_wave_speed`| if search wave speed model |`False`|
|`optimize_wave_speed_after_decay`|search wave speed model after learning rate decay|`True`|
|`second_adjust`|adjust the location slightly after training| `True`|
|`time_before`| time difference between the start of the search and the time of the first pick | 0.5 * maximum searching distance / P-wave speed |



* The multiprocessing might conflict with multithreading in Seisbench, so if you run Seisbench first,  please use
    ```
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        future = executor.submit(association, pick_df, station_df, config,verbose)
    pick_df_out, catalog_df = future.result()
    ```
    instead of
    ```
    pick_df_out, catalog_df=association(pick_df,station_df,config)
    ```
    
* Fixed wave speed needs less training epochs while unknown wave required more, e.g. 10000. Increase number of epochs can also help find more events, slightly.
* For unknown speed model, use
    ```
    pick_df_out, catalog_df=association(pick_df,station_df,config,model_traveltime=model_traveltime)
    ```
    where `model_traveltime` is a model with input as the source location and output as the traveltime from source to each stations. Details can be seen in the [example](https://colab.research.google.com/drive/1lAciDACeV24vHQFVjWraQE8KOb81ATEd?usp=sharing).


