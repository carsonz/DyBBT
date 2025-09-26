import logging
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

sys.path.append('../../../')

import torch
from torch import multiprocessing as mp

from convlab.util.custom_util import (env_config, eval_policy, get_config,
                                      init_logging, load_config_file,
                                      log_start_args, move_finished_training,
                                      save_best, save_config, set_seed)
from system1policy import System1Policy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

if __name__ == '__main__':
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    begin_time = datetime.now()
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../../configs/dybbt.json",
                        help="Load path for DyBBT config file")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")

    args = parser.parse_args()
    path = args.path
    mode = args.mode
    save_eval = args.save_eval_dials

    # Initialize logging
    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)

    # Load and save config
    environment_config = load_config_file(path)
    save_config(vars(args), environment_config, config_save_path)

    conf = get_config(path, [('model', 'seed', 42)])
    seed = conf['model']['seed']
    logging.info('Train seed is ' + str(seed))
    set_seed(seed)

    # Initialize System1Policy
    policy_sys = System1Policy(conf, is_train=True)

    log_start_args(conf)
    logging.info(f"New episodes per epoch: {conf['model']['batchsz']}")

    # Configure environment
    env, sess = env_config(conf, policy_sys)

    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path
    policy_sys.sess = sess
    
    best_complete_rate = 0
    best_success_rate = 0
    best_return = -100
    # Early stopping mechanism variables
    no_improvement_count = 0
    max_no_improvement_epochs = 3

    # Main training loop
    logging.info("Start of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    global_step = 0
    for i in tqdm(range(conf['model']['epoch']), desc="Training Epochs", unit="epoch"):
        idx = i + 1
        
        # Update policy
        global_step = policy_sys.train(env, conf['model']['batchsz'], tb_writer, global_step)

        # Periodic evaluation
        if idx % conf['model']['eval_frequency'] == 0 and idx != 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(
                f"Evaluating after Dialogues: {idx * conf['model']['batchsz']} - {time_now}" + '-' * 60)

            eval_dict = eval_policy(
                conf, policy_sys, env, sess, save_eval, log_save_path)

            # Check for improvement
            current_complete_rate = eval_dict["complete_rate"]
            current_success_rate = eval_dict["success_rate_strict"]
            current_return = eval_dict["avg_return"]
            
            # Save previous best values for comparison
            prev_best_complete_rate = best_complete_rate
            prev_best_success_rate = best_success_rate
            prev_best_return = best_return
            
            # Use save_best function to save best model
            best_complete_rate, best_success_rate, best_return = save_best(
                policy_sys, best_complete_rate, best_success_rate, best_return,
                current_complete_rate, current_success_rate, current_return, save_path)
            
            # Determine if there is improvement
            if (current_complete_rate > prev_best_complete_rate or 
                current_success_rate > prev_best_success_rate or 
                current_return > prev_best_return):
                # Improvement detected, reset counter
                no_improvement_count = 0
            else:
                # No improvement, increment counter
                no_improvement_count += 1
                
            # Check if early stopping is needed
            if no_improvement_count >= max_no_improvement_epochs:
                logging.info(f"Early stopping triggered after {idx} epochs due to no improvement in {max_no_improvement_epochs} consecutive evaluations.")
                break
            
            # policy_sys.save(save_path, "last")
            for key in eval_dict:
                tb_writer.add_scalar(
                    key, eval_dict[key], idx * conf['model']['batchsz'])

    logging.info("End of Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    # Save training time
    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))