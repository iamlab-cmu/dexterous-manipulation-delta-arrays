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
    
    while True:
        if not (log_queue.empty() or dontlog):
            metrics = log_queue.get()
            if metrics is None:
                print("Logger received termination signal. Shutting down.")
                break
            
            wandb.log(metrics)
            