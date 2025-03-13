# Utils functions for data processing and visualization

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass_filter_all(part, lowcut = 1, highcut = 40, fs = 200, order = 5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_part = part.copy()
    channels = part.columns[1:-2]
    for channel in channels:
        filtered_part[channel] = filtfilt(b, a, part[channel])
    return filtered_part

def get_part_after_feedback(data, feedback_times, feedback_id = 0):
    t = feedback_times.iloc[feedback_id]
    return data[(data['Time'] >= t - 0.2) & (data['Time'] <= t + 1.0)], t

def plot_channel(part, t, channel, title = None):
    time = (part['Time'] - t) * 1000
    plt.plot(time, part[channel])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()