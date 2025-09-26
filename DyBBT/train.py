# -*- coding: utf-8 -*-
"""
DyBBT Training Script - Adapted from ConvLab PPO training for DyBBT algorithm
"""

import logging
import os
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

sys.path.append('../')

import numpy as np
import torch
from torch import multiprocessing as mp

from DyBBT import DyBBT
from convlab.policy.rlmodule import Memory
from convlab.util.custom_util import (env_config, eval_policy, get_config,
                                      init_logging, load_config_file,
                                      log_start_args, move_finished_training,
                                      save_best, save_config, set_seed)
from convlab.dialog_agent.env import Environment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

try:
    mp.set_start_method('spawn', force=True)
    mp = mp.get_context('spawn')
except RuntimeError:
    pass


def sampler(pid, queue, evt, env: Environment, policy, batchsz, train_seed=0, user_reward=False):
    """
    Sampler function for DyBBT - samples data from environment
    """
    buff = Memory()
    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    set_seed(train_seed)

    while sampled_num < batchsz:
        # Reset environment for each trajectory
        s = env.reset()
        turn = 0
        
        for t in range(traj_len):
            # Use DyBBT policy to predict action
            a = policy.predict(s, turn)
            
            # Interact with environment
            next_s, r, done = env.step(a, user_reward=user_reward)
            
            # Convert state to vector for training
            s_vec, action_mask = policy.vector.state_vectorize(s)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)
            
            # Convert next state to vector
            next_s_vec, next_action_mask = policy.vector.state_vectorize(next_s)
            next_s_vec = torch.Tensor(next_s_vec)
            
            # Save to buffer
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, 
                     next_s_vec.numpy(), 0 if done else 1, action_mask.numpy())
            
            # Update state and turn counter
            s = next_s
            turn += 1
            real_traj_len = t
            
            if done:
                break

        sampled_num += real_traj_len
        sampled_traj_num += 1

    # Push data to queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num, seed, user_reward=False):
    """
    Multi-process sampling for DyBBT
    """
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    train_seeds = random.sample(range(0, 1000), process_num)
    queue = mp.Queue()
    evt = mp.Event()
    
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz, train_seeds[i], user_reward)
        processes.append(mp.Process(target=sampler, args=process_args))
    
    for p in processes:
        p.daemon = True
        p.start()

    # Merge data from all processes
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)
    
    evt.set()
    return buff0.get_batch()


def sample_single(env, policy, batchsz, process_num, seed, user_reward=False):
    """
    Single-process sampling for DyBBT
    """
    buff = Memory()
    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    set_seed(seed)

    while sampled_num < batchsz:
        s = env.reset()
        turn = 0
        
        for t in range(traj_len):
            # Use DyBBT policy
            a = policy.predict(s, turn)
            
            next_s, r, done = env.step(a, user_reward=user_reward)
            
            s_vec, action_mask = policy.vector.state_vectorize(s)
            s_vec = torch.Tensor(s_vec)
            action_mask = torch.Tensor(action_mask)
            
            next_s_vec, next_action_mask = policy.vector.state_vectorize(next_s)
            next_s_vec = torch.Tensor(next_s_vec)
            
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, 
                     next_s_vec.numpy(), 0 if done else 1, action_mask.numpy())
            
            s = next_s
            turn += 1
            real_traj_len = t
            
            if done:
                break

        sampled_num += real_traj_len
        sampled_traj_num += 1

    return buff.get_batch()


def update(env, policy, batchsz, epoch, process_num, seed=0, user_reward=False):
    """
    Update function for DyBBT training
    """
    # Sample data
    batch = sample_single(env, policy, batchsz, process_num, seed, user_reward)
    
    # Convert to tensors
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    action_mask = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
    batchsz_real = s.size(0)
    
    # Update policy (placeholder - would be implemented for PPO)
    # policy.update(epoch, batchsz_real, s, a, r, mask, action_mask)
    
    # Log training progress
    logging.info(f"Epoch {epoch}: Sampled {batchsz_real} experiences")


if __name__ == '__main__':
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    begin_time = datetime.now()
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./configs/dybbt.json",
                        help="Load path for DyBBT config file")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the policy parameter initialization")
    parser.add_argument("--mode", type=str, default='info',
                        help="Set level for logger")
    parser.add_argument("--save_eval_dials", type=bool, default=False,
                        help="Flag for saving dialogue_info during evaluation")
    parser.add_argument("--user-reward", action="store_true")

    args = parser.parse_args()
    path = args.path
    seed = args.seed
    mode = args.mode
    save_eval = args.save_eval_dials
    use_user_reward = args.user_reward

    # Initialize logging
    logger, tb_writer, current_time, save_path, config_save_path, dir_path, log_save_path = \
        init_logging(os.path.dirname(os.path.abspath(__file__)), mode)

    # Load and save config
    environment_config = load_config_file(path)
    save_config(vars(args), environment_config, config_save_path)

    conf = get_config(path, [('model', 'seed', seed)] if seed is not None else [])
    seed = conf['model']['seed']
    logging.info('DyBBT Train seed is ' + str(seed))
    set_seed(seed)

    # Initialize DyBBT policy
    policy_sys = DyBBT(conf, is_train=True, seed=conf['model']['seed'])

    # Load model if specified
    if conf['model']['use_pretrained_initialisation']:
        logging.info("Loading supervised model checkpoint.")
        policy_sys.load_from_pretrained(conf['model'].get('pretrained_load_path', ""))
    elif conf['model']['load_path']:
        try:
            policy_sys.load(conf['model']['load_path'])
        except Exception as e:
            logging.info(f"Could not load a policy: {e}")
    else:
        logging.info("DyBBT Policy initialised from scratch")

    log_start_args(conf)
    logging.info(f"New episodes per epoch: {conf['model']['batchsz']}")

    # Configure environment
    env, sess = env_config(conf, policy_sys)

    policy_sys.current_time = current_time
    policy_sys.log_dir = config_save_path.replace('configs', 'logs')
    policy_sys.save_dir = save_path
    policy_sys.sess = sess

    # Initial evaluation
    logging.info(f"Evaluating at start - {time_now}" + '-'*60)
    time_now = time.time()
    eval_dict = eval_policy(conf, policy_sys, env, sess, save_eval, log_save_path)
    logging.info(f"Finished evaluating, time spent: {time.time() - time_now}")

    for key in eval_dict:
        tb_writer.add_scalar(key, eval_dict[key], 0)
    
    best_complete_rate = eval_dict['complete_rate']
    best_success_rate = eval_dict['success_rate_strict']
    best_return = eval_dict['avg_return']

    # Main training loop
    logging.info("Start of DyBBT Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    for i in range(conf['model']['epoch']):
        idx = i + 1
        
        # Update policy
        update(env, policy_sys, conf['model']['batchsz'],
               idx, conf['model']['process_num'], seed=seed, user_reward=use_user_reward)

        # Periodic evaluation
        if idx % conf['model']['eval_frequency'] == 0 and idx != 0:
            time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info(
                f"Evaluating after Dialogues: {idx * conf['model']['batchsz']} - {time_now}" + '-' * 60)

            eval_dict = eval_policy(
                conf, policy_sys, env, sess, save_eval, log_save_path)

            best_complete_rate, best_success_rate, best_return = \
                save_best(policy_sys, best_complete_rate, best_success_rate, best_return,
                          eval_dict["complete_rate"], eval_dict["success_rate_strict"],
                          eval_dict["avg_return"], save_path)
            
            policy_sys.save(save_path, "last")
            for key in eval_dict:
                tb_writer.add_scalar(
                    key, eval_dict[key], idx * conf['model']['batchsz'])

        # Periodic knowledge distillation
        if idx % conf['model']['distillation_frequency'] == 0 and idx != 0:
            logging.info(f"Performing knowledge distillation at epoch {idx}")
            policy_sys.distill_system2_knowledge()
            buffer_size = policy_sys.get_distillation_buffer_size()
            logging.info(f"Distillation buffer size: {buffer_size}")
            
            # Log additional distillation stats if available
            if hasattr(policy_sys, 'distillation_buffer') and hasattr(policy_sys.distillation_buffer, 'get_stats'):
                buffer_stats = policy_sys.distillation_buffer.get_stats()
                logging.info(f"Distillation buffer stats: {buffer_stats}")

    logging.info("End of DyBBT Training: " +
                 time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    # Save training time
    f = open(os.path.join(dir_path, "time.txt"), "a")
    f.write(str(datetime.now() - begin_time))
    f.close()

    move_finished_training(dir_path, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "finished_experiments"))
