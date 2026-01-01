# Fast QRS detector

**Fast QRS detector** is a Python module for detecting QRS complexes on an electrocardiogram (ECG) and distributed under the Eclipse Public License (EPL).

The development of this library started in February 2024 as part of [Aura Healthcare](https://www.aura.healthcare) project, in [OCTO Technology](https://www.octo.com/fr) R&D team.


![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection.png)
![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection2.png)


**Website** : https://www.aura.healthcare

**Github** : https://github.com/Aura-healthcare

**Version** : 0.3.0


## Installation / Prerequisites

#### User installation

The easiest way to install fast QRS detector is using ``pip`` :

    $ pip install fast_qrs_detector

#### Dependencies

**Fast QRS detector** requires the following:
- Python (>= 3.6)
- numpy >= 1.18.0
- scipy >= 1.4.0
- pandas >= 1.0.0
- matplotlib >= 3.3.4

Note: The package is tested with Python versions from 3.6 up to 3.11.


## Getting started

### `qrs_detector` Function

This package provides the `qrs_detector` function to detect QRS complexes on an electrocardiogram (ECG).

```python
from fast_qrs_detector import qrs_detector
import numpy as np
import pandas as pd

# Basic usage with NumPy array (returns indices)
signal_array = np.random.rand(10000) # Replace with your actual ECG data
sampling_freq = 360 # Hz
qrs_indices = qrs_detector(signal_array, sampling_freq)

# Usage with 1-column Pandas DataFrame (returns indices)
signal_df_1col = pd.DataFrame({'ecg': signal_array})
qrs_indices_df1 = qrs_detector(signal_df_1col, sampling_freq)

# Usage with 2-column Pandas DataFrame (signal + timestamps)
# Create example timestamps (e.g., starting from a specific time)
timestamps = pd.date_range('2024-01-01', periods=len(signal_array), freq=pd.Timedelta(seconds=1/sampling_freq))
signal_df_2col = pd.DataFrame({'time': timestamps, 'voltage': signal_array})

# Returns timestamps corresponding to QRS peaks
qrs_timestamps = qrs_detector(signal_df_2col, sampling_freq)

# Usage with NaN interpolation (for 2-column DataFrame only)
signal_with_nans = signal_df_2col.copy()
# Introduce some NaNs
signal_with_nans.loc[100:105, 'voltage'] = np.nan # Short gap
signal_with_nans.loc[500:600, 'voltage'] = np.nan # Long gap

# Interpolate gaps up to 10 samples, split signal if gap > 10
# Ignore resulting segments shorter than 60 samples
qrs_ts_interpolated = qrs_detector(signal_with_nans,
                                   sampling_freq,
                                   max_nan_interpolation=10,
                                   min_segment_len=60)

```

**Function Signature:**

```python
fast_qrs_detector.qrs_detector(signal_data, freq_sampling: int, min_rr_sec=0.33, max_nan_interpolation=0, min_segment_len=50)
```

**Parameters:**

- `signal_data` (`numpy.ndarray` | `pandas.DataFrame`): The ECG signal.
    - If `numpy.ndarray`: A 1D array of signal values. Must not contain NaNs.
    - If `pandas.DataFrame`: Can have one or two columns.
        - **One column:** Must be numeric (signal values). Must not contain NaNs.
        - **Two columns:** One column must be numeric (signal values), and the other must be datetime-like (`datetime64` or `timedelta64`). NaNs in the signal column can be handled via `max_nan_interpolation`.
- `freq_sampling` (`int`): The sampling frequency of the signal in Hz.
- `min_rr_sec` (`float`, optional): Minimum RR interval in seconds, which determines the maximum detectable heart rate. Defaults to `0.33` (~180 BPM). See [Heart Rate Limit Parameter](#heart-rate-limit-parameter) for details.
- `max_nan_interpolation` (`int`, optional): **Only for 2-column DataFrame input.** Max consecutive NaNs in the signal column to interpolate linearly. Gaps larger than this will cause the signal to be split into segments. Defaults to 0 (no interpolation).
- `min_segment_len` (`int`, optional): **Only for 2-column DataFrame input.** The minimum number of data points required in a segment (after splitting by large NaN gaps) for it to be processed. Defaults to 50.

**Returns:**

- `numpy.ndarray`: If input was a NumPy array or 1-column DataFrame. Contains the integer indices of the detected QRS peaks.
- `pandas.Index`: If input was a 2-column DataFrame. Contains the timestamp values corresponding to the detected QRS peaks.


### Plot functions

```python
import matplotlib.pyplot as plt
from fast_qrs_detector import print_signal_with_qrs

# Assuming you have:
# signal_data: Your ECG data (NumPy array or DataFrame)
# qrs_results: The output from qrs_detector (indices or timestamps)

# Basic usage (plots a segment and marks QRS)
print_signal_with_qrs(signal_data, qrs_predicted=qrs_results, mini=10000, maxi=15000)

# Show the plot (necessary in scripts, often automatic in Jupyter)
plt.show()

# Example with true labels and timestamps (if signal_data is a 2-col DataFrame)
# true_qrs_timestamps = ... # Load your ground truth timestamps
# print_signal_with_qrs(signal_data,
#                       qrs_predicted=qrs_results, # Should be timestamps
#                       true_qrs=true_qrs_timestamps, # Should be timestamps
#                       mini='2024-07-17 10:05:00',
#                       maxi='2024-07-17 10:05:10',
#                       description="ECG segment with timestamps")
# plt.show()
```
**Function Signature & Parameters:**

```python
fast_qrs_detector.print_signal_with_qrs(signal_data, qrs_predicted, true_qrs=None, mini=0, maxi=None, description="")
```

- `signal_data` (`numpy.ndarray` | `pandas.DataFrame`): The ECG signal data (see `qrs_detector` for format details).
- `qrs_predicted` (list-like | `pd.Index`): Predicted QRS locations (indices or timestamps, matching `signal_data` type).
- `true_qrs` (list-like | `pd.Index`, optional): True QRS locations. Defaults to `None`.
- `mini` (int | str | `pd.Timestamp` | `datetime`, optional): Start index or timestamp for plotting. Defaults to 0.
- `maxi` (int | str | `pd.Timestamp` | `datetime`, optional): End index or timestamp for plotting. Defaults to end of signal.
- `description` (str, optional): Title for the plot. Defaults to "".

**Important Note on Displaying Plots:**

This function *creates* the plot but does not automatically display it by calling `plt.show()`. This gives you flexibility:
- In **Jupyter Notebooks** with `%matplotlib inline`, the plot usually appears automatically after the cell runs.
- In **Python scripts** or other environments (or if the plot doesn't appear automatically in Jupyter), you need to explicitly call `matplotlib.pyplot.show()` *after* calling `print_signal_with_qrs` to display the figure.
- This design allows you to add more elements to the plot (e.g., `plt.title(...)`, `plt.xlabel(...)`) or save it (`plt.savefig(...)`) *before* displaying it. 

## Heart Rate Limit Parameter

The `min_rr_sec` parameter controls the minimum interval between consecutive R-peaks, which sets the maximum detectable heart rate:

| `min_rr_sec` | Max HR (BPM) | Use Case |
|--------------|--------------|----------|
| `0.33` (default) | ~180 | Standard ECG, resting to moderate exercise |
| `0.30` | ~200 | Exercise ECG |
| `0.25` | ~240 | High-intensity exercise |
| `0.20` | ~300 | Tachycardia, stress tests |

### Benchmark Results

A comprehensive evaluation was conducted on the main 4 datasets with tolerance = 100ms:

![F1-Score by Dataset and BPM](figures/heatmap_bpm.png)

**Key findings:**
- The default value (180 BPM / 0.33s) provides the best overall average F1-score
- Lower thresholds (higher BPM limits) slightly improve performance in very noisy signals (SNR <= 0dB)
- In clean to moderately noisy signals, the 180 BPM limit performs best
- Gains from increasing the BPM limit are marginal (+0.1% on standard datasets)

**Recommendation:** Keep the default `min_rr_sec=0.33` unless you specifically need to detect very high heart rates (>180 BPM), which typically only occurs during high-intensity exercise or pathological tachycardia.

## Quality control and performances

[This algorithm uses a benchmark previously made by Aura.](https://github.com/ecg-tools/benchmark-qrs-detectors)
 It compares different public libraries in terms of accuracy on 5 public datasets. This algorithm achieves better results in terms of accuracy, as well as efficiency. Compared with the best of the algorithms tested, this algorithm achieves better results with an average time 5 times shorter. [More details here.](README2.md)


## References

Here are the main references used to made this algorithm:

> Zidelmal, Z., Amirou, A., Adnane, M., & Belouchrani, A. (2012). QRS detection based on wavelet coefficients. Computer Methods And Programs In Biomedicine, 107(3), 490‑496. https://doi.org/10.1016/j.cmpb.2011.12.004

> Lu, X., Pan, M., & Yu, Y. (2018). QRS detection based on improved adaptive threshold. Journal Of Healthcare Engineering, 2018, 1‑8. https://doi.org/10.1155/2018/5694595 

> Modak, S., Abdel-Raheem, E., & Taha, L. Y. (2021). A novel adaptive multilevel thresholding based algorithm for QRS detection. Biomedical Engineering Advances, 2, 100016. https://doi.org/10.1016/j.bea.2021.100016 

> M. Šarlija, F. Jurišić and S. Popović (2017). "A convolutional neural network based approach to QRS detection," Proceedings of the 10th International Symposium on Image and Signal Processing and Analysis,121-125 https://doi.org/10.1109/ISPA.2017.8073581 

## Author

**Jean-Charles Fournier** - (https://github.com/Genialmanos)


## License

This project is licensed under the *Eclipse Public License - v 2.0* - see the [LICENSE.md](https://github.com/Genialmanos/QRS_Detector/blob/main/LICENSE) file for details

## Acknowledgments

I hereby thank Clément Le Couedic and Fabien Peigné, my coworkers who gave me time to Open Source this library.
