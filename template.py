import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class Template:
    # errp_raw has to be a numpy array of shape (n_sequences, n_channels, n_samples)
    # That's because for the template of ErrP signals we get the coherent average
    # of every feedback segment.
    # There is no problem if n_sequences == 1
    def __init__(self, errp_raw):        
        min_samples = min(arr.shape[1] for arr in errp_raw)
        truncated = [arr[:, :min_samples] for arr in errp_raw]
        self.errp_raw = np.array(truncated)

    def get_coherent_average(self):
        return np.mean(self.errp_raw, axis=0)
    
    def get_resampled_and_normalized_curve(self, S=32):
        """
        signal: shape (n_channels, T)
        S: number of segments (e.g., 16)
        Returns: list of resampled & normalized curves, one per channel
        """
        signal = self.get_coherent_average()
        n_channels, T = signal.shape
        delta = T // (S + 1)
        
        resampled_curves = []
        for ch in range(n_channels):
            x = []
            y = []
            for s in range(S + 1):
                idx = s * delta
                if idx >= T:
                    break
                x.append(idx)
                y.append(signal[ch, idx])
            
            x = np.array(x)
            y = np.array(y)
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)  # add epsilon to avoid divide by zero
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
            
            resampled_curves.append((x_norm, y_norm))

        return resampled_curves  # list of (x, y) pairs per channel
    
    def plot_resampled_vs_original(self, channel_idx=0):
        original = self.get_coherent_average()[channel_idx]
        x_resampled, y_resampled = self.get_resampled_and_normalized_curve()[channel_idx]
        
        # Normalize original curve for fair comparison
        orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
        t = np.linspace(0, 1, len(original))  # normalized time axis
        
        plt.figure(figsize=(10, 4))
        plt.plot(t, orig_norm, label='Original (Normalized)', linewidth=2)
        plt.plot(x_resampled, y_resampled, marker='o', label='Resampled (SHCC input)', linestyle='--')
        plt.title(f"Channel {channel_idx} – Normalized vs Resampled")
        plt.xlabel("Normalized Time")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def __compute_shcc(x, y):
        """
        x, y: normalized coordinates of a resampled ERP curve
        Returns:
            SHCC chain code: np.array of slopes
        """
        slopes = []
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            slope = dy / (dx + 1e-8)  # avoid division by zero
            slopes.append(round(slope, 4))  # keep precision similar to paper

        return np.array(slopes)

    def get_shcc_chains(self):
        return np.array([self.__compute_shcc(x, y) for (x, y) in self.get_resampled_and_normalized_curve()])

    def get_tortuosity(self):
        chains = self.get_shcc_chains()
        tor = []
        for i in range(chains.shape[0]):
            tor.append(np.sum(np.abs(np.floor(chains[i, :-1]) / 100)))
        return np.array(tor)
    
class TemplateComparer:
    def __init__(self, basic_template: Template, trial_template: Template = None):
        self.basic_template = basic_template
        self.trial_template = trial_template

    def get_chains_distance(self):
        template_chains = self.basic_template.get_shcc_chains()
        trial_chains = self.trial_template.get_shcc_chains()
        distance = []
        for i in range(template_chains.shape[0]):
            distance.append(np.sum(np.abs(template_chains[i, :] - trial_chains[i, :])))
        return np.array(distance)
    
    def get_area_difference(self):
        template_curve = self.basic_template.get_resampled_and_normalized_curve()
        trial_curve = self.trial_template.get_resampled_and_normalized_curve()
        A_template = 0.5 * (template_curve[:, :-1] + template_curve[:, 1:]) 
        A_candidate = 0.5 * (trial_curve[:, :-1] + trial_curve[:, 1:])

        segmentwise_diff = A_template - A_candidate
        total_diff = np.sum(np.abs(segmentwise_diff), axis=1) 

        return segmentwise_diff, total_diff
    
    def get_tortuosity_difference(self):
        template_tortuosity = self.basic_template.get_tortuosity()
        trial_tortuosity = self.trial_template.get_tortuosity()
        return np.abs(template_tortuosity - trial_tortuosity) # l1 norm
    
    # helper function to invert normalize: 1 = best match, 0 = worst match
    def norm_similarity(self, x):
        return 1 - (x / (np.max(x) + 1e-6))

    def get_area_similarity(self):
        return self.norm_similarity(self.get_area_difference())
    
    def get_chains_similarity(self):
        return self.norm_similarity(self.get_chains_distance())

    def get_tortuosity_similarity(self):
        return self.norm_similarity(self.get_tortuosity_difference())
    
    def get_weighted_temp_similarity(self, w_a=0.4, w_c=0.4, w_t=0.2):
        return w_a * self.get_area_similarity() + w_c * self.get_chains_similarity() + w_t * self.get_tortuosity_similarity()
    
    # helper function to truncate signals that have to be compared element by element
    def __truncate_if_need(self, sig1, sig2):
        n1 = sig1.shape[1]
        n2 = sig2.shape[1]
        L = min(n1, n2)
        return sig1[:, :L], sig2[:, :L]

    def get_pearson_correlation(self):
        sig1, sig2 = self.__truncate_if_need(self.basic_template.errp_raw, self.trial_template.errp_raw)
        n_channels = sig1.shape[0]
        pearson_scores = np.zeros(n_channels)
        for ch in range(n_channels):
            corr = np.corrcoef(sig1[ch], sig2[ch])[0, 1]
            pearson_scores[ch] = (corr + 1) / 2  # map [-1,1] to [0,1]
        return pearson_scores
    
    def get_cosine_similarity(self):
        sig1, sig2 = self.__truncate_if_need(self.basic_template.errp_raw, self.trial_template.errp_raw)
        n_channels = sig1.shape[0]
        cosine_scores = np.zeros(n_channels)
        for ch in range(n_channels):
            cosine_scores[ch] = np.dot(sig1[ch], sig2[ch]) / (norm(sig1[ch]) * norm(sig2[ch]) + 1e-8)
        return cosine_scores
    
    def get_cross_correlation_peak(self):
        sig1, sig2 = self.__truncate_if_need(self.basic_template.errp_raw, self.trial_template.errp_raw)
        n_channels = sig1.shape[0]
        crosscorr_scores = np.zeros(n_channels)
        for ch in range(n_channels):
            x1 = sig1[ch] - np.mean(sig1[ch])
            x2 = sig2[ch] - np.mean(sig2[ch])
            corr = np.correlate(x1, x2, mode='full')
            peak = np.max(corr) / (np.std(x1) * np.std(x2) * len(x1) + 1e-8)
            crosscorr_scores[ch] = min(peak, 1.0)  # clip to [0,1]
        return crosscorr_scores
    
    def get_weighted_signal_similarity(self, w_pc=0.4, w_cs=0.3, w_cc=0.3):
        return w_pc*self.get_pearson_correlation() + w_cs*self.get_cosine_similarity() + w_cc*self.get_cross_correlation_peak()
    
