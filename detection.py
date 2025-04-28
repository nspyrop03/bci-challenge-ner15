import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import important_channels
import utils
import os
from template import Template, TemplateComparer

class FileProcessor:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.feedback_times = self.raw_data[self.raw_data['FeedBackEvent'] == 1]['Time']
        self.feedback_indices = self.feedback_times.index
        self.fs = 200 # sampling frequency
        self.default_channel = important_channels[0] # channel Cz

        # Check if it's the 5th session.
        # In the 5th session there may be a correction after the feedback that
        # lasts one additional second. So we have to keep this in mind
        self.is_last_session = "Sess05" in file_path

        # filter the data
        temp = utils.bandpass_filter_all(self.raw_data, highcut=30)
        self.filtered_data = utils.bandpass_filter_all(temp, lowcut=0.2, highcut=10)

        self.extracted_features = self.extract_all_features()
        self.errps = self.find_errp_all()

    # Helper function for the 5th session to cluster the feeedback time distances
    # and find whether there was a correction or not.
    # There are 4 classes:
    # 1) Normal-letter no correction
    # 2) Normal-letter with correction
    # 3) Last-letter no correction
    # 4) Last-letter with correction
    def __cluster_time_diff(self, diff_values, n_clusters=4):
        diff_values = diff_values.reshape(-1, 1)  # shape (n_samples, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(diff_values)
        return labels

    def get_feedback_diffs(self):
        time_diffs = []
        for i in range(1, len(self.feedback_indices)):
            diff = self.feedback_times.iloc[i] - self.feedback_times.iloc[i-1]
            time_diffs.append(diff)
        return time_diffs

    def get_feedback_labels(self):
        if not self.is_last_session: return None
        return self.__cluster_time_diff(self.get_feedback_diffs())
    
    def get_feedbacks_with_correction(self):
        if not self.is_last_session: return None

        time_diffs = self.get_feedback_diffs()
        labels = self.get_feedback_labels()
        levels = {0:[], 1:[], 2:[], 3:[]}
        for i in range(len(labels)):
            levels[labels[i]].append(time_diffs[i])
        for key in levels.keys():
            levels[key] = np.mean(levels[key])

        sorted_keys = sorted(levels, key=levels.get)
        
        # labels that indicate that correction follows
        correction_labels = [sorted_keys[1], sorted_keys[3]]
        feedback_correction_ids = []
        for i in range(len(labels)):
            if labels[i] in correction_labels:
                feedback_correction_ids.append(i)
        return feedback_correction_ids

    # Return True if after the feedback with id "feedback_id" follows 
    # a correction segment that lasts 1 second
    def has_feedback_correction(self, feedback_id):
        if not self.is_last_session: return False
        return feedback_id in self.get_feedbacks_with_correction()

    def is_last_letter(self, feedback_id):
        return feedback_id % 5 == 4

    def get_break_after_feedback(self, feedback_id, channel):
        break_time = 4.5 if self.is_last_letter(feedback_id) else 0.5
        
        # If correction follows the feedback segment, add 1 second
        if self.has_feedback_correction(feedback_id): break_time += 1

        t1 = self.feedback_times.iloc[feedback_id] + 1.3 # end of feedback
        t2 = t1 + break_time
        part = self.filtered_data[channel].to_numpy()
        part = part[int(t1*self.fs):int(t2*self.fs)+1]
        part_mean = np.mean(part)

        return break_time, part_mean
    
    # All the methods get the part after the given feedback_id.
    # Hence, when feedback_id=0 they get the part for the second letter
    # That's why I introduce feedback_id=-1 to work for the first letter.
    # The problem is that before the first letter there is a variant "test" phase.
    # To deal with this, I calculate the mean of the min and the max time of the erp segment
    # and use it as the duration of the erp segment for the first letter.
    def get_erp_segment(self, feedback_id, channel):
        part = self.filtered_data[channel].to_numpy()

        if feedback_id >= 0:
            break_time, _ = self.get_break_after_feedback(feedback_id, channel)
            
            t1 = self.feedback_times.iloc[feedback_id] + 1.3 + break_time
            if feedback_id < self.feedback_indices.size - 1:
                t2 = self.feedback_times.iloc[feedback_id + 1]
            else:
                t2 = self.filtered_data['Time'].iloc[-1]
            
            part = part[int(t1*self.fs):int(t2*self.fs)+1]

        else:
            duration = 9.3 
            t1 = self.feedback_times.iloc[0] - duration
            t2 = self.feedback_times.iloc[0]
            
            part = part[int(t1*self.fs):int(t2*self.fs)+1]

        return part
        
    def get_blinking_time(self, feedback_id, channel):
        part = self.get_erp_segment(feedback_id, channel)
        dt = len(part) / self.fs - 2
        if dt <= 6.64:
            return 2.64
        elif dt > 6.64 and dt < 7.78:
            d1 = abs(6.64 - dt)
            d2 = abs(7.78 - dt)
            return 2.64 if d1 < d2 else 5.28
        else:
            return 5.28

    def get_important_parts(self, feedback_id, channel, extra_offset = 0.1):
        erp_segment = self.get_erp_segment(feedback_id, channel)
        data = {}
        data['feedback_id'] = feedback_id
        data['channel'] = channel

        # find the green circle part
        green_end = 1
        data['green_raw'] = erp_segment[:int((green_end+extra_offset)*self.fs)]
        _, break_mean = self.get_break_after_feedback(feedback_id, channel)
        data['green_processed'] = data['green_raw'] - break_mean

        # find the between break part mean to process the blinking part
        between_break_end = 2
        between_break = erp_segment[int(green_end*self.fs):int(between_break_end*self.fs)]
        between_break_mean = np.mean(between_break)

        # find the blinking part
        blinking_time = self.get_blinking_time(feedback_id, channel)
        blinking_start = between_break_end - extra_offset
        blinking_end = blinking_start + blinking_time + extra_offset
        data['blinking_raw'] = erp_segment[int(blinking_start*self.fs):int(blinking_end*self.fs)]
        data['blinking_processed'] = data['blinking_raw'] - between_break_mean

        return data
    
    def get_feedback_erp_raw(self, index):
        part, _ = utils.get_part_after_feedback(self.filtered_data, self.feedback_times, feedback_id=index, before=0.0, after=1.3)
        part = part[important_channels]
        return part.to_numpy().T

    def get_erp_templates(self):
        # template_data = {
        #   "index": 0 to len-1,
        #   "type": "green/blinking",
        #   "template": Template(...)
        # } 
        template_data_list = []
        errp_data = []
        # i from -1 to 58 or 98 -> after feedback with id 0 to 59 or 99
        for i in range(-1, len(self.feedback_indices)-1):
            green_raw = []
            blinking_raw = []
            for channel in important_channels:
                raw_data = self.get_important_parts(i, channel)
                green_raw.append(raw_data['green_raw'])
                blinking_raw.append(raw_data['blinking_raw'])

            min_length = min(len(channel_data) for channel_data in green_raw)
            green_raw = [channel_data[:min_length] for channel_data in green_raw] # truncate to have the same length in each channel
            green_raw = np.array(green_raw)
            green_raw = green_raw[np.newaxis, :, :] # reshape to (1, n_channels, n_samples)
            template_data_list.append({
                "index": i+1,
                "type": "green",
                "template": Template(green_raw)
            })

            min_length = min(len(channel_data) for channel_data in blinking_raw)
            blinking_raw = [channel_data[:min_length] for channel_data in blinking_raw]
            window_time = 0.6
            window_size = int(window_time * self.fs)
            window_step = int(0.25 * self.fs)
            n_windows = (min_length - window_size) // window_step + 1
            for window in range(n_windows):
                part = blinking_raw[:, window*window_step:window*window_step+window_size]
                part = part[np.newaxis, :, :]
                template_data_list.append({
                    "index": i+1,
                    "type": "blinking",
                    "template": Template(part)
                })

            errp_data.append(self.get_feedback_erp_raw(i+1))
        min_samples = min(arr.shape[1] for arr in errp_data)
        errp_data = [arr[:, :min_samples] for arr in errp_data]
        errp_data = np.array(errp_data)
        feedback_template = Template(errp_data)

        return template_data_list, feedback_template

    def get_positive_negative_templates(self):
        if not self.is_last_session: return None, None

        errp_pos = []
        errp_neg = []
        fbc = self.get_feedbacks_with_correction()
        for i in range(len(self.feedback_indices)):
            if i in fbc:
                errp_neg.append(self.get_feedback_erp_raw(i))
            else:
                errp_pos.append(self.get_feedback_erp_raw(i))
        min_samples = min(arr.shape[1] for arr in errp_pos)
        errp_pos = [arr[:, :min_samples] for arr in errp_pos]
        errp_pos = np.array(errp_pos)
        min_samples = min(arr.shape[1] for arr in errp_neg)
        errp_neg = [arr[:, :min_samples] for arr in errp_neg]
        errp_neg = np.array(errp_neg)
        return Template(errp_pos), Template(errp_neg)

    def sliding_window(self, feedback_id, channel):
        data = self.get_important_parts(feedback_id, channel)
        blinking_data = data['blinking_processed']
        
        window_time = 0.5
        window_size = int(window_time * self.fs)
        window_step = int(0.25 * self.fs)
        windows = []
        num_windows = (len(blinking_data) - window_size) // window_step + 1
        
        for i in range(num_windows):
            window = {}
            window['signal'] = blinking_data[i*window_step:i*window_step+window_size]
            window['mean'] = np.mean(window['signal'])
            window['peak'] = np.max(window['signal'])
            window['latency'] = np.argmax(window['signal']) / self.fs
            window['amplitude'] = np.max(window['signal']) - np.min(window['signal'])
            windows.append(window)
            
        # not a sliding window but add the green circle part as a window
        green_data = data['green_processed']
        green_window = {
            'signal': green_data,
            'mean': np.mean(green_data),
            'peak': np.max(green_data),
            'latency': np.argmax(green_data) / self.fs,
            'amplitude': np.max(green_data) - np.min(green_data)
        }
        windows.append(green_window)

        return windows
        
    def extract_features(self, feedback_id, verbose=False):
        feature_matrix = []
        for channel in important_channels:
            windows = self.sliding_window(feedback_id, channel)
            channel_mean = []            
            channel_peak = []
            channel_latency = []
            channel_amplitude = []

            for w in windows:
                channel_mean.append(w['mean'])
                channel_peak.append(w['peak'])
                channel_latency.append(w['latency'])
                channel_amplitude.append(w['amplitude'])
            feature_matrix.append(channel_mean)
            feature_matrix.append(channel_peak)
            feature_matrix.append(channel_latency)
            feature_matrix.append(channel_amplitude)

            if verbose: print(f'Channel {channel} with {len(windows)} windows')

        return np.array(feature_matrix).T
    
    def extract_all_features(self, verbose=False):
        features = []
        # Starting from -1 to get the first letter as introduced in get_erp_segment()
        for id in range(-1, self.feedback_indices.size-1):
            temp = self.extract_features(id)
            if verbose: print(f'shape for id {id} is {temp.shape}')
            features.append(temp)

        features = np.vstack(features)
        if verbose: print(f'Final shape: {features.shape}')
        return features
    
    # The criterion is stricter when n is lower and threshold higher
    # 'n' is the maximum number of peaks that can be higher than the threshold
    # 'threshold' is the ratio of the peak value to the max peak value
    # 'lat_fix' should be equal to 'before' parameter of utils.get_part_after_feedback() function
    # n=2 and threshold=0.7 after testing, feel free to change it
    def find_errp(self, feedback_id, channel, n=2, threshold=0.7, before=0.3, after=0.3, lat_fix=0.2, verbose=False):
        part, t = utils.get_part_after_feedback(self.filtered_data, self.feedback_times, feedback_id=feedback_id, before=lat_fix, after=1.3)
        np_signal = part[channel].to_numpy()
        peaks, _ = find_peaks(np_signal)
        peak_values = np_signal[peaks]
        peak_values = peak_values[peak_values > 0]

        max_peak = np.max(peak_values)
        peak_values = peak_values[peak_values != max_peak] # remove the max peak
        peak_values.sort() # sort peaks in ascending order
        peak_values = peak_values[-n:]
        if verbose: print(peak_values, max_peak)
        
        times = 0
        for v in peak_values:
            if (v / max_peak) >= threshold:
                times += 1

        peak_index = np.argmax(np_signal)
        T = 1/self.fs
        before_samples = int(before/T)
        after_samples = int(after/T) 

        latency = peak_index * T - lat_fix
        if verbose: print(f'Latency: {latency}s')

        start_slice = max(peak_index - before_samples, 0)
        end_slice = min(peak_index + after_samples + 1, len(np_signal))

        if verbose: print(f'Signal slice: [{start_slice}, {end_slice}]')
        # return (check, errp, latency)
        return times < n, np_signal[start_slice:end_slice], latency
    
    def find_errp_all(self, n=2, threshold=0.7, before=0.3, after=0.3, lat_fix=0.2, verbose=False):
        errps = []
        for id in range(self.feedback_indices.size):
            errp = []
            for channel in important_channels:
                check, signal, latency = self.find_errp(id, channel, n, threshold, before, after, lat_fix, verbose)
                
                mean = np.mean(signal)
                peak = np.max(signal)
                amplitude = peak - np.min(signal)
                
                # added whether it's considered an errp or not as a feature
                errp.extend([check, mean, peak, latency, amplitude])
                
            if verbose: print(f'Feedback {id} with {len(errp)} features')
            errps.append(errp)
        return np.array(errps)
    
class SubjectData:
    def __init__(self, subject_name = "06", train=True):
        self.train = train
        self.subject_name = subject_name
        self.cache_p300 = f'cache/subject_{subject_name}_p300.npy'
        self.cache_errp = f'cache/subject_{subject_name}_errp.npy'
        self.raw_features, self.errp_features = self.__load_data()
        
        self.cache_template = f'cache/template/subject_{subject_name}'
        self.ctemp_positive = f'{self.cache_template}_positive.npy'
        self.ctemp_negative = f'{self.cache_template}_negative.npy'
        self.ctemp_feedback = f'{self.cache_template}_feedback.npy'
        self.ctemp_green = f'{self.cache_template}_green.npy'
        self.positive_template, self.negative_template, self.feedback_templates, self.green_templates = self.__load_templates()
    
    def __load_data(self):
        if os.path.exists(self.cache_p300) and os.path.exists(self.cache_errp):
            return np.load(self.cache_p300), np.load(self.cache_errp)
        else:
            features_p300 = []
            features_errp = []
            prefix_path = 'data/train' if self.train else 'data/test'
            for i in range(1, 6):
                file_path = f'{prefix_path}/Data_S{self.subject_name}_Sess0{i}.csv'
                fp = FileProcessor(file_path)
                features_p300.append(fp.extracted_features)
                features_errp.append(fp.errps)
            features_p300 = np.vstack(features_p300)
            features_errp = np.vstack(features_errp)

            os.makedirs(os.path.dirname(self.cache_p300), exist_ok=True)
            np.save(self.cache_p300, features_p300)
            np.save(self.cache_errp, features_errp)
            
            return features_p300, features_errp

    def __load_templates(self) -> tuple[Template, Template, list[Template], list[Template]]:
        if os.path.exists(self.ctemp_positive) and os.path.exists(self.ctemp_negative) and os.path.exists(self.ctemp_feedback) and os.path.exists(self.ctemp_green):
            pos = Template(np.load(self.ctemp_positive))
            neg = Template(np.load(self.ctemp_negative))
            fbs = np.load(self.ctemp_feedback)
            fb_temps = []
            for i in range(fbs.shape[0]):
                fb_temps.append(Template[fbs[i]])
            greens = np.load(self.ctemp_green)
            green_temps = []
            for i_fb in range(greens.shape[0]):
                green_temps.append(Template(greens[i_fb]))
            return pos, neg, fb_temps, green_temps
        else:
            prefix_path = 'data/train' if self.train else 'data/test'
            green_temps = []
            green_data = []
            fb_temps = []
            fb_data = []
            pos_temp, neg_temp
            for i in range(1, 6):
                file_path = f'{prefix_path}/Data_S{self.subject_name}_Sess0{i}.csv'
                fp = FileProcessor(file_path)
                temp_data_list, fb_temp = fp.get_erp_templates()
                fb_temps.append(fb_temp)
                fb_data.append(fb_temp.errp_raw)
                for temp_data in temp_data_list:
                    if temp_data['type'] == "green": 
                        green_temps.append(temp_data['template'])
                        green_data.append(temp_data['template'].errp_raw)
                
                if i == 5:
                    pos_temp, neg_temp = fp.get_positive_negative_templates()
                    
            np.save(self.ctemp_positive, pos_temp.errp_raw)
            np.save(self.ctemp_negative, neg_temp.errp_raw)
            fb_data = np.vstack(fb_data)
            green_data = np.vstack(green_data)
            np.save(self.ctemp_feedback, fb_data)
            np.save(self.ctemp_green, green_data)

            return pos_temp, neg_temp, fb_temps, green_temps
    def dbscan_features(self, pca_components=5, eps=0.5, top_percentage=0.3, verbose=False):
        """ Apply PCA and then DBSCAN clustering to classify P300 vs non-P300 windows """
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.raw_features)

        pca = PCA(n_components=pca_components)
        features_pca = pca.fit_transform(features_scaled)

        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(features_pca)

        # Find the cluster with the highest mean amplitude
        unique_labels = np.unique(labels)
        cluster_means = {label: self.raw_features[labels == label].mean(axis=0) for label in unique_labels if label != -1}
        #print(cluster_means)

        # Sort clusters by mean amplitude (assuming amplitudes are at indexes 3, 7, 11, ...)
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: np.mean(x[1][3::4]), reverse=True)

        l = len(unique_labels)
        stop = int(l*top_percentage)
        #print(f'l={l}, top={top_percentage}, stop={stop}')

        # Select top_percentage*l clusters as P300
        p300_labels = [cluster[0] for cluster in sorted_clusters[:stop]]
        if verbose: print("P300 Clusters:", p300_labels)
        
        # Convert labels into binary classes (1 = P300, 0 = Non-P300)
        binary_labels = np.zeros_like(labels)
        c = 0
        for i in range(len(labels)):
            if labels[i] in p300_labels:
                binary_labels[i] = 1
                c += 1
        if verbose: print(f'Found {c}/{len(labels)} ({(c/len(labels)*100):.2f}%) P300 segments')
        #print(binary_labels.shape)

        return binary_labels, labels, features_pca