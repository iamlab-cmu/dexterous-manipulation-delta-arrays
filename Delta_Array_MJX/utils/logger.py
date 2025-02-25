import numpy as np
from collections import defaultdict
import wandb

class MetricLogger:
    def __init__(self, dontlog):
        self.metrics = defaultdict(list)
        self.dontlog = dontlog
    
    def add_data(self, key, value):
        """Add a data point to the specified metric."""
        self.metrics[key].append(value)
    
    def log_metrics(self, max_length):
        """
        Resample all metrics to `max_length`, log them to wandb, and then flush the metrics.
        """
        if not self.dontlog:
            resampled = {}
            for key in self.metrics:
                data = self.metrics[key]
                if not data:
                    continue
                
                data_len = len(data)
                x_orig = np.arange(data_len)
                x_target = np.linspace(0, data_len - 1, max_length)
                resampled_data = np.interp(x_target, x_orig, data)
                resampled[key] = resampled_data
            
            for step in range(max_length):
                log_dict = {key: resampled[key][step] for key in resampled}
                wandb.log(log_dict)
        
        self.metrics.clear()