import torch
import numpy as np
import json
import logging
import copy
import os
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from convlab.policy.policy import Policy
from collections import deque
import gc
import torch.nn as nn
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from DyBBT.utils import CustomVectorBase, parse_model_output, get_domains_from_action, get_domains_belief_state, get_available_actions
from torch.distributions import Categorical
# Import custom PPO trainer
from ppo import PPOTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class System1Policy(Policy):
    def __init__(self, config: Dict, is_train: bool = False):
        super().__init__()

        self.config = config
        self.is_train = is_train
        self.sess = None

        self.model_name = "LLM-Research/Llama-3.2-1B-Instruct"
        self.sft_checkpoint_dir = "../sft_checkpoints_Llama-3.2-1B-Instruct"
        self.ppo_checkpoint_dir = "./ppo_checkpoints_Llama-3.2-1B-Instruct"

        # Configure LoRA
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        
        self.load()
        
        # 初始化PPO训练器
        if self.is_train:
            self.init_ppo_trainer()
        
        # Print model information
        if hasattr(self.ppo_trainer, 'print_trainable_parameters'):
            self.ppo_trainer.print_trainable_parameters()
        else:
            self.model.print_trainable_parameters()
        self.vector = CustomVectorBase()

    def get_action_and_value(self, query_tensor, response_tensor, scores, model_outputs):
        """
        Calculate action, action probability and state value
        
        Args:
            query_tensor: Query tensor
            response_tensor: Response tensor
            scores: Scores during generation process
            model_outputs: Model outputs (generation result object)
        """
        # Ensure query_tensor and response_tensor dimensions match
        # query_tensor is 2D [batch_size, seq_len], response_tensor is 1D [seq_len] or 2D
        if query_tensor.dim() == 2 and response_tensor.dim() == 1:
            # Expand response_tensor to 2D
            response_tensor = response_tensor.unsqueeze(0)
        elif query_tensor.dim() == 1 and response_tensor.dim() == 1:
            # If both are 1D, expand both to 2D
            query_tensor = query_tensor.unsqueeze(0)
            response_tensor = response_tensor.unsqueeze(0)
        
        # Get last layer hidden states
        hidden_states = model_outputs.hidden_states[-1]  # Hidden states of the last layer
        
        # Get last token's hidden state
        if isinstance(hidden_states, (list, tuple)):
            # If it's a tuple or list, take the last element
            last_hidden_state = hidden_states[-1]
        else:
            # If it's a tensor, use it directly
            last_hidden_state = hidden_states
        
        # Ensure last_hidden_state is 2D [batch_size, hidden_size]
        if last_hidden_state.dim() > 2:
            # If it's 3D [batch_size, seq_len, hidden_size], take the last token
            last_hidden_state = last_hidden_state[:, -1, :]
        
        # Calculate value
        value = self.model.value_head(last_hidden_state).squeeze(-1)
        
        # Use the passed scores parameter to get logits
        logits = scores[-1]  # Logits of the last token
        
        # Get the actual generated token from response as action
        action = response_tensor[:, -1]  # Last generated token
        
        # Calculate action probability distribution
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        # Calculate log probability
        log_prob = action_dist.log_prob(action)
        
        # Calculate entropy
        entropy = action_dist.entropy()
        
        # Ensure return value dimensions are consistent
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return action, log_prob, value, entropy

    def get_action_and_value_for_ppo(self, query_tensor, attention_mask, action_tensor):
        """
        Method specifically for PPO updates, recalculates action, log probability, value and entropy
        
        Args:
            query_tensor: Query tensor
            attention_mask: Attention mask
            action_tensor: Action tensor (action to evaluate)
        """
        # Ensure query_tensor and action_tensor dimensions match
        if query_tensor.dim() == 2 and action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        elif query_tensor.dim() == 1 and action_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
        
        # Forward pass to get hidden states
        outputs = self.model(
            input_ids=query_tensor,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Get last token's hidden state
        if hidden_states.dim() > 2:
            last_hidden_state = hidden_states[:, -1, :]
        else:
            last_hidden_state = hidden_states
        
        # Calculate value
        value = self.model.value_head(last_hidden_state).squeeze(-1)
        
        # Get logits (last token's logits)
        logits = outputs.logits[:, -1, :]
        
        # Clip logits to prevent numerical instability
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # Get action (passed action)
        action = action_tensor[:, -1]
        
        # Calculate action probability distribution (using more stable softmax)
        # First ensure logits don't have extreme values
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        # Use log_softmax then exp for better numerical stability
        log_action_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        action_probs = torch.exp(log_action_probs)
        
        # Check again and handle NaN values
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            # If NaN or inf appears, use uniform distribution as fallback
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
            action_probs = action_probs.to(logits.device)
        
        # Ensure probabilities are within valid range and satisfy simplex constraint
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        
        # Ensure probability sum is 1 (satisfy simplex constraint)
        action_probs_sum = action_probs.sum(dim=-1, keepdim=True)
        
        # Check if probability sum is 0 or invalid
        if torch.isnan(action_probs_sum).any() or torch.isinf(action_probs_sum).any() or (action_probs_sum == 0).any():
            # If probability sum is invalid, use uniform distribution
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        else:
            action_probs = action_probs / action_probs_sum
        
        # Final check to ensure probabilities are valid
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or (action_probs < 0).any():
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        action_dist = Categorical(action_probs)
        
        # Calculate log probability
        log_prob = action_dist.log_prob(action)
        
        # Calculate entropy
        entropy = action_dist.entropy()
        
        # Ensure return value dimensions are consistent
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        return action, log_prob, value, entropy

    def predict(self, state):
        """
        System 1 prediction (fast intuitive response)
        In practice, this would call the LoRA-finetuned LLM model
        """
        # 构建查询prompt
        domains = get_domains_from_action(s['user_action'])
        belief_state = get_domains_belief_state(domains, s['belief_state'])
        available_actions = get_available_actions(self.vector, domains, belief_state)
        system1_prompt = [
            {"role": "system", "content": "You are the fast, intuitive component (System 1) of a dialogue system. Your task is to generate the next system action based solely on the current belief state. Do not reason step-by-step. Output your first, most intuitive response in the exact JSON format specified."},
            {"role": "user", "content": f"""
**Current Belief State:**
{belief_state}

**Available Actions:**
{available_actions}

Based on the above, output ONLY a valid JSON object with your predicted action and its confidence. Do not output any other text.

{{"action": [["<act_type>", "<domain>", "<slot>"], ["<act_type>", "<domain>", "<slot>"], ...],"confidence": <confidence_score>}}
"""}
        ]
        # 编码查询
        formatted_input = self.tokenizer.apply_chat_template(system1_prompt, tokenize=False)
        encoding = self.tokenizer(formatted_input, truncation=True, max_length=833, return_tensors="pt")
        query_tensor = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # 生成响应
        with torch.no_grad():
            response_tensor = self.model.generate(
                query_tensor,
                attention_mask=attention_mask,
                max_new_tokens=100,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 获取生成的token序列
        generated_tokens = response_tensor.sequences[0]
        
        # 确保generated_tokens维度与query_tensor匹配
        if generated_tokens.dim() == 1:
            generated_tokens = generated_tokens.unsqueeze(0)

        
        # 解码响应
        response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        actions = parse_model_output(response_text)
        actions = self.vector.form_actions(actions)
        
        return actions
    
    def init_ppo_trainer(self):
        """初始化PPO训练器"""
        # 创建PPO配置
        ppo_config = {
            'learning_rate': 1e-5,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'value_coef': 0.1,
            'entropy_coef': 0.01,
            'batch_size': 1,
            'ppo_epochs': 4,
            'max_grad_norm': 0.5,
            'gae_lambda': 0.95,
            'cosine_t_max': 1000,
            'cosine_eta_min': 1e-7,
            'gradient_accumulation_steps': 32
        }
        
        # 初始化自定义PPO训练器
        self.ppo_trainer = PPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ppo_config,
            policy=self  # 传递policy对象引用
        )

    def train(self, env, batchsz, tb_writer=None, global_step=0):
        sampled_traj_num = 0
        traj_len = 40
        
        # 使用更高效的数据结构存储轨迹数据
        # Use deque to limit maximum length, automatically clean old data
        max_buffer_size = self.ppo_trainer.batch_size * 2  # Set buffer size
        queries = deque(maxlen=max_buffer_size)
        responses = deque(maxlen=max_buffer_size)
        rewards = deque(maxlen=max_buffer_size)
        ppo_actions = deque(maxlen=max_buffer_size)
        ppo_log_probs = deque(maxlen=max_buffer_size)
        ppo_values = deque(maxlen=max_buffer_size)
        ppo_entropies = deque(maxlen=max_buffer_size)
        
        # Periodic cleanup counter
        cleanup_counter = 0
        cleanup_interval = 5  # Clean memory every 5 trajectories processed
        
        pbar = tqdm(total=batchsz, desc="Sampling trajectories", unit="traj")
        while sampled_traj_num < batchsz:
            s = env.reset()
            done = False
            
            for t in range(traj_len):
                # Build query prompt
                domains = get_domains_from_action(s['user_action'])
                belief_state = get_domains_belief_state(domains, s['belief_state'])
                available_actions = get_available_actions(self.vector, domains, belief_state)

                filled1, total1 = self.get_state_filled_num(s['belief_state'], domains)
                
                system1_prompt = [
            {"role": "system", "content": "You are the fast, intuitive component (System 1) of a dialogue system. Your task is to generate the next system action based solely on the current belief state. Do not reason step-by-step. Output your first, most intuitive response in the exact JSON format specified."},
            {"role": "user", "content": f"""
**Current Belief State:**
{belief_state}

**Available Actions:**
{available_actions}

Based on the above, output ONLY a valid JSON object with your predicted action and its confidence. Do not output any other text.

{{"action": [["<act_type>", "<domain>", "<slot>"], ["<act_type>", "<domain>", "<slot>"], ...],"confidence": <confidence_score>}}
"""}
        ]
                
                # Encode query
                formatted_input = self.tokenizer.apply_chat_template(system1_prompt, tokenize=False)
                encoding = self.tokenizer(formatted_input, truncation=True, max_length=833, return_tensors="pt")
                query_tensor = encoding['input_ids'].to(DEVICE)
                attention_mask = encoding['attention_mask'].to(DEVICE)
                
                # Generate response and get model output
                with torch.no_grad():
                    response_tensor = self.model.generate(
                        query_tensor,
                        attention_mask=attention_mask,
                        max_new_tokens=100,
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=True
                    )
                
                # Get generated token sequence
                generated_tokens = response_tensor.sequences[0]
                
                # Ensure generated_tokens dimensions match query_tensor
                if generated_tokens.dim() == 1:
                    generated_tokens = generated_tokens.unsqueeze(0)
                
                # Calculate action, log probability, and value for generated sequence
                ppo_action, log_prob, ppo_value, entropy = self.get_action_and_value(
                    query_tensor, generated_tokens, response_tensor.scores, response_tensor)
                
                # Decode response
                response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                actions = parse_model_output(response_text)

                invalid_action = False
                if actions is None:
                    invalid_action = True
                
                # print(f"Belief State: {belief_state}")
                # print(f"SYS Actions: {actions}")
                
                actions = self.vector.form_actions(actions)

                # print(f"SYS Actions After: {actions}")
                
                # Execute action
                s, r, done = env.step(actions, user_reward=True)

                # print(f"Domains: {domains}")
                # print(f"USER Actions: {s['user_action']}")
                
                # Calculate reward (based on state fill ratio change)
                domains = get_domains_from_action(s['user_action'])
                filled2, total2 = self.get_state_filled_num(s['belief_state'], domains)             
                
                if t > traj_len * 0.6:
                    r = -0.2 * (t - traj_len * 0.6)
                else:
                    r = -0.05

                if done:
                    r = 20
                elif invalid_action:
                    r += -5
                elif filled2 > filled1: 
                    r += (filled2 - filled1) * 1.5
                elif filled2 == filled1: 
                    r += -0.05
                else:
                    r += (filled2 - filled1) * 2

                # print(f"Reward After: {r}")
                # print(f"BS After: {get_domains_belief_state(domains, s['belief_state'])}")
                print(f"idx: {t}, filled1: {filled1}, filled2: {filled2}, total1={total1}, total2={total2}, reward={r}")
                print('-'*100)
                
                # Store trajectory data
                queries.append(query_tensor)
                responses.append(generated_tokens)
                rewards.append(r)
                
                ppo_actions.append(ppo_action)
                ppo_log_probs.append(log_prob)
                ppo_values.append(ppo_value)
                ppo_entropies.append(entropy)
        
                # Collect PPO training data
                if len(queries) >= self.ppo_trainer.batch_size:
                    # Prepare PPO training data
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(DEVICE)

                    # Execute PPO training step
                    self.ppo_trainer.train_step(queries, responses, rewards, ppo_actions, ppo_log_probs, ppo_values, ppo_entropies)
                    
                    logging.info(f"Collected {len(queries)} samples for PPO training")
                    logging.info(f"Average reward: {rewards_tensor.mean().item():.3f}")
                    
                    if tb_writer is not None:
                        tb_writer.add_scalar("PPO/average_reward", rewards_tensor.mean().item(), global_step)
                    global_step += 1
                    
                    # Clear trajectory data to avoid memory leaks
                    queries.clear()
                    responses.clear()
                    rewards.clear()
                    ppo_actions.clear()
                    ppo_log_probs.clear()
                    ppo_values.clear()
                    ppo_entropies.clear()
                
                if done:
                    break
                
            sampled_traj_num += 1
            pbar.update(1)
            pbar.set_postfix({"current": sampled_traj_num, "total": batchsz})
            
            # Periodically clean memory
            cleanup_counter += 1
            if cleanup_counter >= cleanup_interval:
                cleanup_counter = 0                
                # Force garbage collection
                gc.collect()                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        pbar.close()
            
        return global_step

    def save(self, save_dir, pattern):
        """
        Save model weights
        pattern: best-return, best-success, best-complete, #last
        """
        self.tokenizer.save_pretrained(f"{save_dir}/{pattern}")
        self.ppo_trainer.save_model(save_dir, pattern)

    def load(self):
        # 从ModelScope下载模型
        model_dir = snapshot_download(self.model_name)

        # 加载分词器和基础模型
        #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=bnb_config)
        
        # 只在模型还没有peft配置时才转换为PEFT模型
        if not hasattr(self.model, 'peft_config'):
            self.model = get_peft_model(self.model, self.peft_config)
        
        # 确保模型在正确的设备上
        self.model = self.model.to(DEVICE)
        
        if self.is_train:
            # 加载SFT检查点的LoRA权重
            self.model.load_adapter(self.sft_checkpoint_dir, adapter_name="default")
            # 添加价值头，使用固定的隐藏层维度                   
            if not hasattr(self.model, 'value_head'):
                # 使用模型配置中的隐藏层大小，确保维度固定
                hidden_size = self.model.config.hidden_size
                self.model.value_head = nn.Linear(hidden_size, 1).to(DEVICE)
                # 初始化value_head权重
                nn.init.orthogonal_(self.model.value_head.weight, gain=0.01)
                nn.init.constant_(self.model.value_head.bias, 0.0)
        else:
            self.model.load_adapter(f"{self.ppo_checkpoint_dir}/best-return", adapter_name="default")
    
    def get_state_filled_num(self, belief_state: Dict[str, Dict[str, str]], domains: List[str]):
        """
        计算指定 domain 列表中非空值的比例
        
        Args:
            belief_state: 包含各个domain状态的字典
            domains: 需要计算比例的domain列表
            
        Returns:
            int: 非空值数量, 总值数量
        """
        total_slots = 0
        filled_slots = 0
        
        for domain in domains:
            if domain in belief_state:
                domain_state = belief_state[domain]
                for slot, value in domain_state.items():
                    total_slots += 1
                    if value != '':
                        filled_slots += 1
            
        return filled_slots, total_slots