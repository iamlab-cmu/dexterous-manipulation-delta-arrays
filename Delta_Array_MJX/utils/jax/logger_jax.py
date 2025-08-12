import time
import numpy as np
from collections import defaultdict
import wandb

def logger_worker(log_queue, config):
    global dontlog
    dontlog = config['dont_log'] or config['test']
    if not dontlog:
        if config['wb_resume']:
            wandb.init(project="MARL_Dexterous_Manipulation", config=config, name=config['name'], 
                    id=config['wb_resume'], resume=True, entity=config['wb_entity'])
        else:
            wandb.init(project="MARL_Dexterous_Manipulation", config=config, name=config['name'], 
                    entity=config['wb_entity'])
    
    metrics = defaultdict(list)
    last_flush = time.time()
    
    while True:
        if not (log_queue.empty() or dontlog):
            msg = log_queue.get()
            if msg is None:
                print("Flushing")
                if metrics:
                    _do_flush(metrics, config, last_flush, final=True)
                break

            if msg["type"] == 0:
                metrics[msg["k"]].append(msg["v"])

            elif msg["type"] == 1:
                max_length = msg["len"]
                _do_flush(metrics, max_length)
                metrics.clear()
                last_flush = time.time()
            
def _do_flush(metrics, max_length):
    if not dontlog:
        resampled = {}
        for key, data in metrics.items():
            if not data: 
                continue
            data_len = len(data)
            x_orig = np.arange(data_len)
            x_target = np.linspace(0, data_len - 1, max_length)
            resampled[key] = np.interp(x_target, x_orig, data)
        # now step through and log
        for step in range(max_length):
            log_dict = {k: resampled[k][step] for k in resampled}
            wandb.log(log_dict)
            