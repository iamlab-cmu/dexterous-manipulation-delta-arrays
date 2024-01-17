import time

def log_data(logger, ep_rewards, episode, start_time):
    """ Store data about training progress in systematic data structures """
    logger.save_state({'ep_rewards': ep_rewards}, None)
    logger.log_tabular('Episode', episode)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)            
    # logger.log_tabular('Q1Vals', with_min_and_max=True)
    # logger.log_tabular('Q2Vals', with_min_and_max=True)
    # logger.log_tabular('LogPi', with_min_and_max=True)
    # logger.log_tabular('LossPi', average_only=True)
    # logger.log_tabular('LossQ', average_only=True)
    logger.log_tabular('Time', time.time()-start_time)
    logger.dump_tabular()