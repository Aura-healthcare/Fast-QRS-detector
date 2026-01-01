import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_timedelta64_dtype

def qrs_detector(sig, freq_sampling, min_rr_sec=0.33):
    """
    Detects QRS complexes in an ECG signal.

    Args:
        sig (pd.Series, np.ndarray, or pd.DataFrame): The input ECG signal.
            - If Series or ndarray: Contains the power/amplitude values.
            - If DataFrame: Should contain one numeric power/amplitude column,
              or one numeric power/amplitude column and one time-like (datetime/timedelta) column.
              The function attempts to automatically identify columns based on dtype.
        freq_sampling (int): The sampling frequency of the signal in Hz.
        min_rr_sec (float, optional): Minimum RR interval in seconds. This sets the 
            maximum detectable heart rate. Defaults to 0.33 (corresponding to ~180 BPM).
            Lower values allow detecting higher heart rates (e.g., 0.2 for ~300 BPM) 
            but may reduce robustness in noisy signals.

    Returns:
        np.ndarray: An array containing either:
            - The indices of the detected QRS complexes if the input was a Series, ndarray,
              or a single-column DataFrame.
            - The timestamps/timedeltas corresponding to the detected QRS complexes if the input
              was a two-column DataFrame with identifiable time and power columns.

    Raises:
        ValueError: If the input DataFrame structure is ambiguous or doesn't meet requirements.
        TypeError: If the input signal type is not supported.
    """
    is_timestamp_mode = False
    power_signal = None
    time_values = None

    if isinstance(sig, pd.DataFrame):
        n_cols = sig.shape[1]
        if n_cols == 1:
            if is_numeric_dtype(sig.iloc[:, 0]):
                power_signal = sig.iloc[:, 0].values
            else:
                raise ValueError("Single-column DataFrame must contain numeric data.")
        elif n_cols == 2:
            time_col_name, power_col_name = None, None
            col_names = sig.columns
            dtypes = sig.dtypes

            time_like_cols = [name for name in col_names if is_datetime64_any_dtype(dtypes[name]) or is_timedelta64_dtype(dtypes[name])]
            numeric_cols = [name for name in col_names if is_numeric_dtype(dtypes[name])]

            numeric_cols_only = [name for name in numeric_cols if name not in time_like_cols]


            if len(time_like_cols) == 1 and len(numeric_cols_only) == 1:
                time_col_name = time_like_cols[0]
                power_col_name = numeric_cols_only[0]
                is_timestamp_mode = True
            else:
                raise ValueError(
                    "Input DataFrame must have exactly two columns: "
                    "one time-like (datetime/timedelta) and one numeric (power), "
                    "or provide a single-column numeric DataFrame, Series, or array."
                )

            if is_timestamp_mode:
                time_values = sig[time_col_name].values
                power_signal = sig[power_col_name].values

        else:
            raise ValueError("Input DataFrame must have exactly one or two columns.")

    elif isinstance(sig, pd.Series):
        if is_numeric_dtype(sig):
            power_signal = sig.values
        else:
            raise ValueError("Input Series must contain numeric data.")
    elif isinstance(sig, (np.ndarray, list)):
         temp_sig = np.asarray(sig)
         if is_numeric_dtype(temp_sig):
             power_signal = temp_sig
         else:
             raise ValueError("Input array or list must contain numeric data.")
    else:
        raise TypeError(f"Unsupported input type: {type(sig)}. Provide DataFrame, Series, ndarray, or list.")

    cleaned_ecg = preprocess_ecg(power_signal, freq_sampling, 5, 22, size_window = int( 0.1 * freq_sampling))
    peaks = detect_peaks(cleaned_ecg, no_peak_distance= int(freq_sampling*0.65), distance = int(freq_sampling * min_rr_sec))
    qrs_indices = threshold_detection(cleaned_ecg, peaks, freq_sampling, initial_search_samples= int(freq_sampling * 0.83), long_peak_distance=int(freq_sampling*1.111))

    if is_timestamp_mode:
        valid_indices = qrs_indices[qrs_indices < len(time_values)]
        if len(valid_indices) != len(qrs_indices):
             print(f"Warning: {len(qrs_indices) - len(valid_indices)} QRS indices were out of bounds for the provided time values.")
        return time_values[valid_indices]
    else:
        return qrs_indices

def detect_peaks(cleaned_ecg, no_peak_distance, distance=0):
    last_max = -np.inf  # The most recent encountered maximum value
    last_max_pos = -1  # Position of the last_max in the array
    peaks = [np.argmax(cleaned_ecg[:no_peak_distance])]  # Detected peaks positions
    peak_values = [cleaned_ecg[peaks[0]]]  # Detected peaks values

    
    for i, current_value in enumerate(cleaned_ecg):
       
        # Update the most recent maximum if the current value is greater
        if current_value > last_max:
            last_max = current_value
            last_max_pos = i
        
        # Check if the current value is less than half the last max
        # or if we are beyond the no_peak_distance from the last max
        if current_value <= last_max / 2 or (i - last_max_pos >= no_peak_distance):
            # Check if the last peak is within the `distance` of the current peak
            if last_max_pos - peaks[-1] < distance:
                # If within the distance, choose the higher peak
                if last_max > peak_values[-1]:
                    peaks[-1] = last_max_pos
                    peak_values[-1] = last_max
            else:
                # Otherwise, start a new peak group
                peaks.append(last_max_pos)
                peak_values.append(last_max)
            
            # Reset the last max after adding a peak
            last_max = current_value
            last_max_pos = i
    
    return np.array(peaks)

def threshold_detection(cleaned_ecg, peaks, fs, initial_search_samples=300, long_peak_distance=400):

    spk = 0.13 * np.max(cleaned_ecg[:initial_search_samples])
    npk = 0.1 * spk
    threshold = 0.25 * spk + 0.75 * npk
    
    qrs_peaks = []
    noise_peaks = []
    qrs_buffer = []
    last_qrs_time = 0
    min_distance = int(fs * 0.12)
    
    for i, peak in enumerate(peaks):
        peak_value = cleaned_ecg[peak]
        
        if peak_value > threshold:
            if qrs_peaks and (peak - qrs_peaks[-1] < min_distance):
                if peak_value > cleaned_ecg[qrs_peaks[-1]]:
                    qrs_peaks[-1] = peak
            else:
                qrs_peaks.append(peak)
                last_qrs_time = peak
            
            spk = 0.25 * peak_value + 0.75 * spk
            
            qrs_buffer.append(peak)
            if len(qrs_buffer) > 10:
                qrs_buffer.pop(0)
        else:
            noise_peaks.append(peak)
            npk = 0.25 * peak_value + 0.75 * npk
        
        threshold = 0.25 * spk + 0.75 * npk
        
        if peak - last_qrs_time > long_peak_distance:
            spk *= 0.5
            threshold = 0.25 * spk + 0.75 * npk
            for lookback_peak in peaks[i-5:i+1]:
                if lookback_peak != last_qrs_time:
                    if last_qrs_time < lookback_peak < peak and cleaned_ecg[lookback_peak] > threshold:
                        qrs_peaks.append(lookback_peak)
                        spk = 0.875 * spk + 0.125 * cleaned_ecg[lookback_peak]
                        threshold = 0.25 * spk + 0.75 * npk
                        last_qrs_time = lookback_peak
                        break
        
        if len(qrs_buffer) > 1:
            rr_intervals = np.diff(qrs_buffer)
            mean_rr = np.mean(rr_intervals)
            if peak - last_qrs_time > 1.5 * mean_rr:
                spk *= 0.5
                threshold = 0.25 * spk + 0.75 * npk
    
    return np.array(qrs_peaks)

def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    low = cutoff / nyquist
    b, a = butter(order, low, btype='low')
    y = filtfilt(b, a, data)
    return y

def differentiate(data):
    return np.diff(data, prepend=data[0])

def squaring(data):
    return np.square(data)

def moving_window_integration(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_ecg(data, fs, high, low, size_window):
    signal = highpass_filter(data, high, fs)
    signal = lowpass_filter(signal, low, fs)
    signal = differentiate(signal)
    signal = squaring(signal)
    signal = moving_window_integration(signal, size_window)
    return signal