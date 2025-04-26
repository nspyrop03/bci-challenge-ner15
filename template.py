import pandas as pd
import numpy as np
import utils

class Template:
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        self.feedback_times = self.raw_data[self.raw_data['FeedBackEvent'] == 1]['Time']
        self.feedback_indices = self.feedback_times.index
        