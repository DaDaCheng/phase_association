
# HARPA Usage Guide

## Usage

To perform the association, use the following code:

```python
from harpa import association
pick_df_out, catalog_df = association(pick_df, station_df, config)
```

### Input Requirements

#### `pick_df`
A DataFrame containing seismic pick information with the following structure:

| id        | timestamp               | prob     | type |
|-----------|-------------------------|----------|------|
| CX.MNMCX. | 2014-04-01 00:08:45.410 | 0.113388 | p    |
| CX.MNMCX. | 2014-04-01 00:10:56.840 | 0.908900 | p    |
| CX.MNMCX. | 2014-04-01 00:13:01.930 | 0.795285 | p    |
| CX.MNMCX. | 2014-04-01 00:13:36.950 | 0.130214 | s    |

- `id`: Station id
- `timestamp`: arrival time of the picks (`np.datetime64`)
- `prob`: Optional column indicating the probability of the pick.
- `type`: Indicates the phase type (e.g., `p` or `s`).

#### `station_df`
A DataFrame containing seismic station information with the following structure:

| id       | x (km)     | y (km)     | z (km)   |
|----------|------------|------------|----------|
| CX.PB01. | 449.359240 | 7672.990624| -0.900   |
| CX.PB02. | 407.073618 | 7642.202105| -1.015   |
| CX.PB03. | 422.289208 | 7561.616351| -1.460   |
| CX.PB04. | 381.654501 | 7529.786448| -1.520   |

---

## Configurations and Notes

### Mandatory Configurations

| Config    | Description                          | Example      |
|-----------|--------------------------------------|--------------|
| `x(km)`   | X range for the search space         | `[0, 100]`   |
| `y(km)`   | Y range for the search space         | `[0, 100]`   |
| `z(km)`   | Z range for the search space         | `[0, 100]`   |
| `vel`     | Homogeneous wave speed model         | `{"P": 7, "S": 4.0}` |

### Optional Configurations

| Config                      | Description                                            | Default      |
|-----------------------------|--------------------------------------------------------|--------------|
| `lr`                        | Learning rate                                         | `0.1`        |
| `noise`                     | $\epsilon$ in SGLD                                    | `1e-3`       |
| `lr_decay`                  | Learning rate and noise decay                         | `0.1`        |
| `epoch_before_decay`        | Number of epochs before decay                         | `1000`       |
| `epoch_after_decay`         | Number of epochs after decay                          | `1000`       |
| `LSA`                       | Use linear sum assignment in computing loss           | `True`       |
| `wasserstein_p`             | Order of the Wasserstein distance                     | `2`          |
| `P_phase`                   | Indicates if data contains P-phase                   | `True`       |
| `S_phase`                   | Indicates if data contains S-phase                   | `False`      |
| `noisy_pick`                | Indicates if data contains missing/spurious picks    | `True`       |
| `min_peak_pre_event`        | Minimum number of picks per event                    | `16`         |
| `min_peak_pre_event_p`      | Minimum number of P-phase picks per event            | `0`          |
| `min_peak_pre_event_s`      | Minimum number of S-phase picks per event            | `0`          |
| `max_time_residual`         | Maximum allowable time residual for events           | `2`          |
| `denoise_rate`              | Filtering based on nearest station ratio             | `0`          |
| `DBSCAN`                    | Use DBSCAN to segment data into different slices     | `True`       |
| `remove_overlap_events`     | Remove repeated events if slices overlap             | `False`      |
| `neural_field`              | `True` for unknown wave speed model, `False` otherwise | `False`    |
| `wave_speed_model_hidden_dim` | Hidden dimension for wave speed model if `neural_field=True` | - |
| `optimize_wave_speed`       | Optimize wave speed model                             | `False`      |
| `optimize_wave_speed_after_decay` | Optimize wave speed after learning rate decay       | `True`       |
| `second_adjust`             | Adjust locations after training                      | `True`       |
| `time_before`               | Time difference between the search start and first pick | `0.5 * maximum_search_distance / P_wave_speed` |

---

### Notes

1. **Multiprocessing and Seisbench Compatibility**:
   - The multiprocessing module might conflict with Seisbench's multithreading. If you use Seisbench first, wrap your association call as follows:

    ```python
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        future = executor.submit(association, pick_df, station_df, config, verbose)
    pick_df_out, catalog_df = future.result()
    ```

   Otherwise, you can directly use:

    ```python
    pick_df_out, catalog_df = association(pick_df, station_df, config)
    ```

2. **Training Epochs**:
   - Fixed wave speed models require fewer training epochs. Increasing the number of epochs can help identify more events.
   - Unknown wave speed models may need more epochs (e.g., 10,000). 
   - 
3. **Using Unknown Speed Models**:
   - For unknown speed models, include a `model_traveltime` parameter:

    ```python
    pick_df_out, catalog_df = association(pick_df, station_df, config, model_traveltime=model_traveltime)
    ```

   - Here, `model_traveltime` is a model with the source location as input and travel times to each station as output. See the [example](https://colab.research.google.com/drive/1lAciDACeV24vHQFVjWraQE8KOb81ATEd?usp=sharing) for details.

3. **Verbose Settings**:
   - `verbose > 10`: Prints picks and stations.
   - `verbose > 8`: Prints training details for each epoch.
   - `verbose > 6`: Prints training settings, including CPUs.
   - `verbose > 4`: Prints configuration details.
   - `verbose > 2`: Prints other details.
   - `verbose = 0`: Silent mode.





--- 
