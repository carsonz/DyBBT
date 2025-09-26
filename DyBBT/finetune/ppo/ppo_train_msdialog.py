#!/usr/bin/env python3
"""
MSDialog PPO Fine-tuning Script - Adapted for MSDialog three domains
"""

import logging
import os
import sys
import time
import json
import random
import numpy as np
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


class MSDialogPPOTrainer:
    """MSDialog PPO Training Class"""
    
    def __init__(self, config, domain):
        self.config = config
        self.domain = domain
        self.setup_domain_specific_config()
    
    def setup_domain_specific_config(self):
        """Setup domain-specific configuration"""
        # Adjust training parameters based on domain
        if self.domain == 'movie':
            # Movie domain typically has more complex dialogues
            self.config['model']['batchsz'] = 32
            self.config['model']['epoch'] = 100
            self.config['model']['eval_frequency'] = 5
        elif self.domain == 'restaurant':
            # Restaurant domain has moderate complexity
            self.config['model']['batchsz'] = 40
            self.config['model']['epoch'] = 80
            self.config['model']['eval_frequency'] = 4
        elif self.domain == 'taxi':
            # Taxi domain is relatively simple
            self.config['model']['batchsz'] = 48
            self.config['model']['epoch'] = 60
            self.config['model']['eval_frequency'] = 3
    
    def load_msdialog_data(self, data_path):
        """Load MSDialog dataset for the specific domain"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Loaded MSDialog {self.domain} domain data with {len(data)} samples")
            return data
        except Exception as e:
            logging.error(f"Error loading MSDialog data from {data_path}: {e}")
            return []
    
    def prepare_msdialog_environment(self, policy_sys, train_data_path, val_data_path):
        """Prepare MSDialog-specific environment configuration"""
        # Load training and validation data
        train_data = self.load_msdialog_data(train_data_path)
        val_data = self.load_msdialog_data(val_data_path)
        
        # Update config with MSDialog-specific settings
        self.config['task'] = {
            'name': 'msdialog',
            'domain': self.domain,
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }
        
        # Configure environment with MSDialog data
        env, sess = env_config(self.config, policy_sys)
        
        # Inject MSDialog data into environment if needed
        if hasattr(env, 'set_msdialog_data'):
            env.set_msdialog_data(train_data, val_data)
        
        return env, sess


def main():
    """Main function for MSDialog PPO training"""
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    begin_time = datetime.now()
    
    # Parse arguments
    parser = ArgumentParser(description='MSDialog PPO Fine-tuning Script')
    parser.add_argument("--path", type=str, default="../../configs/dybbt.json",
                        help="Load path for DyBBT config file")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")
    parser.add_argument("--domain", type=str, required=True,
                        choices=['movie', 'restaurant', 'taxi'],
                        help="MSDialog domain name")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Training data path (e.g., msdialog_movie_train.json)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Validation data path (e.g., msdialog_movie_val.json)")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="SFT checkpoint path for initialization")

    args = parser.parse_args()
    path = args.path
    mode = args.mode
    save_eval = args.save_eval_dials
    domain = args.domain
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    sft_checkpoint = args.sft_checkpoint

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

    # Create MSDialog PPO trainer
    msdialog_trainer = MSDialogPPOTrainer(conf, domain)

    # Initialize System1Policy with MSDialog configuration
    policy_sys = System1Policy(conf, is_train=True)
    
    # Load SFT checkpoint if provided
    if sft_checkpoint and os.path.exists(sft_checkpoint):
        logging.info(f"Loading SFT checkpoint from {sft_checkpoint}")
        try:
            # Load SFT checkpoint for model initialization
            policy_sys.load_sft_checkpoint(sft_checkpoint)
            logging.info(f"Successfully loaded SFT checkpoint from {sft_checkpoint}")
        except Exception as e:
            logging.warning(f"Failed to load SFT checkpoint: {e}")

    log_start_args(conf)
    logging.info(f"New episodes per epoch: {conf['model']['batchsz']}")
    logging.info(f"MSDialog domain: {domain}")

    # Configure MSDialog-specific environment
    env, sess = msdialog_trainer.prepare_msdialog_environment(policy_sys, train_data_path, val_data_path)

    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path
    policy_sys.sess = sess
    
    # Early stopping variables
    best_complete_rate = 0
    best_success_rate = 0
    best_return = -100
    no_improvement_count = 0
    max_no_improvement_epochs = 3

    # Main training loop
    logging.info("Start of MSDialog PPO Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    global_step = 0
    for i in tqdm(range(conf['model']['epoch']), desc=f"MSDialog {domain} Training Epochs", unit="epoch"):
        idx = i + 1
        
        # Update policy with MSDialog data
        global_step = policy_sys.train(env, conf['model']['batchsz'], tb_writer, global_step)

        # Periodic evaluation
        if idx % conf['model']['eval_frequency'] == 0 and idx != 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(
                f"Evaluating MSDialog {domain} after Dialogues: {idx * conf['model']['batchsz']} - {time_now}" + '-' * 60)

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
            
            # Log metrics to TensorBoard
            for key in eval_dict:
                tb_writer.add_scalar(
                    f"msdialog_{domain}/{key}", eval_dict[key], idx * conf['model']['batchsz'])

    logging.info("End of MSDialog PPO Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    # Save training time
    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))
    
    logging.info(f"MSDialog {domain} domain PPO training completed successfully!")


if __name__ == "__main__":
    main()