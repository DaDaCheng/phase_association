# <span style="font-family: sans-serif;">Harpa</span>: High-Rate Phase Association with Travel Time Neural Fields


## About

This is the source code for paper _Harpa: High-Rate Phase Association with Travel Time Neural Fields_.


## Installation 
Simply run:
```
pip install -q git+https://github.com/DaDaCheng/phase_association.git
```
It requires `torch`,`obspy`, `pyproj`, `pandas` for constant wave speed model, and `skfmm` from unknown wave speed model.


## Quick start in SeisBench and colab demo

HARPA seamlessly integrates with workflow in [SeisBench](https://github.com/seisbench/seisbench).


| Examples                                         |  |
|--------------------------------------------------|---|
| 2014 Chile earthquake                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o7S8n2LtJChraLoHqoNykQ_m9aqWifG-?usp=sharing) |
| 2019 Ridgcrest earthquake                                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16lE4eu0SM3xQVb-686XL-0evPXOTPzwC#scrollTo=ZUFnMmLlTHec) |

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

| id       | longitude | latitude | elevation(m) | x(km)     | y(km)     | z(km)   |
|----------|-----------|----------|--------------|-----------|-----------|---------|
| CX.PB01. | -69.48740 | -21.04323| 900.0        | 449.359240| 7672.990624| -0.900  |
| CX.PB02. | -69.89603 | -21.31973| 1015.0       | 407.073618| 7642.202105| -1.015  |
| CX.PB03. | -69.75310 | -22.04847| 1460.0       | 422.289208| 7561.616351| -1.460  |
| CX.PB04. | -70.14918 | -22.33369| 1520.0       | 381.654501| 7529.786448| -1.520  |

Details about the `config` parameter and settings for the unknown wave speed model can be found in the [documentation](xxxx).



