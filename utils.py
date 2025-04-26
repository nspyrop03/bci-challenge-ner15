# Utils functions for data processing and visualization

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# List of important channels for the P300 speller paradigm as found in bibliography
important_channels = ['Cz', 'Pz', 'FCz', 'Fz', 'C1', 'CP1', 'CPz', 'CP2', 'C2', 'C3', 'FC3', 'P1', 'FC1', 'FC2', 'F1', 'C4']
train_subjects = ['02', '06', '07', '11', '12', '13', '14', '16', '17', '18', '20', '21', '22', '23', '24', '26']
test_subjects = ['01', '03', '04', '05', '08', '09', '10', '15', '19', '25']

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

def bandpass_filter_arr(signal, lowcut = 1, highcut = 40, fs = 200, order = 5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# The paper that describes the experiment setup says that the letter chosen is displayed for 1.3s => after = 1.3 by default
def get_part_after_feedback(data, feedback_times, feedback_id = 0, before = 0.2, after = 1.3):
    t = feedback_times.iloc[feedback_id]
    return data[(data['Time'] >= t - before) & (data['Time'] <= t + after)], t

def plot_channel(part, t, channel = important_channels[0], title = None):
    time = (part['Time'] - t) * 1000
    plt.plot(time, part[channel])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

def plot_signal(signal, ylabel = 'Amplitude', title = None):
    plt.plot(signal)
    plt.xlabel('Samples')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_feedback_time_diffs(feedback_times, feedback_indices, save_png=False, png_name=None, verbose=False):
    print(f'There are {len(feedback_times)} feedback times')
    time_diffs = []
    time_bounds = []
    for i in range(len(feedback_indices)):
        if i == 0:
            if verbose: print(f'[word {int(i/5)}, letter {i%5}] {feedback_times.iloc[i]}')
        else:
            diff = feedback_times.iloc[i] - feedback_times.iloc[i-1]
            time_diffs.append(diff)
            if i%5 == 0: time_bounds.append(i-1)
            if verbose: print(f'[word {int(i/5)}, letter {i%5}] {diff}')

    plt.figure(figsize=(8, 6))
    for x_line in time_bounds:
        plt.axvline(x=x_line, color='r', linestyle='--', linewidth=1)
    plt.plot(time_diffs, 'o')
    plt.xlabel("Feedback diff index")
    plt.ylabel("Time (s)")
    plt.title("Time between feedbacks")
    if save_png: plt.savefig(f"docs/images/{png_name}.png")
    plt.show()
